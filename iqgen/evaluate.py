"""SNR / SIR sweep evaluator.

Usage:
    # AWGN sweep on a SigMF recording, expected bits supplied
    python -m iqgen.evaluate clean.sigmf-meta \\
        --bits-file expected.txt \\
        --interferer awgn \\
        --sweep -20:20:2

    # SIR sweep with a tone, with framing
    python -m iqgen.evaluate clean.sigmf-meta \\
        --bits-file payload.txt \\
        --framing \\
        --interferer tone --tone-hz 50000 \\
        --sweep -10:30:2

    # Run a single point (no sweep) with a file interferer
    python -m iqgen.evaluate clean.cf32 -m qpsk -s 1e6 -b 100e3 \\
        -f root_raised_cosine \\
        --interferer file --interferer-file jammer.cf32 \\
        --sir 6

Outputs:
    - CSV to stdout (or --csv path)
    - Optional PNG plot of BER vs target_db via --plot
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .channel import (add_awgn, add_tone, awgn, from_file as load_interferer,
                       mix, tone)
from .framing import FrameConfig
from .verifier import (ReceiveParams, compare_bits, demodulate,
                        demodulate_frame, detect_format,
                        find_sigmf_data_for_meta, load_iq,
                        params_from_sigmf_meta, parse_bits)
from .verify_cli import _offsets, build_parser as _build_demod_parser

log = logging.getLogger(__name__)


@dataclass
class SweepPoint:
    target_db: float
    sync_found: bool
    sync_distance: int
    sync_pattern_bits: int
    fec_corrections: int
    crc_scheme: str
    crc_ok: Optional[bool]
    n_compared: int
    n_errors: int
    ber: float


CSV_FIELDS = ["target_db", "sync_found", "sync_distance", "sync_pattern_bits",
              "fec_corrections", "crc_scheme", "crc_ok", "n_compared",
              "n_errors", "ber"]


def _parse_sweep(spec: str) -> list[float]:
    """Parse 'start:stop:step' or a single value or 'v1,v2,v3'."""
    if "," in spec:
        return [float(x) for x in spec.split(",") if x.strip()]
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) == 2:
            start, stop = float(parts[0]), float(parts[1])
            step = 1.0
        elif len(parts) == 3:
            start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        else:
            raise ValueError(f"Bad sweep spec: {spec}")
        if step == 0:
            raise ValueError("sweep step cannot be zero")
        # inclusive of stop within float precision
        n = int(round((stop - start) / step)) + 1
        return [start + i * step for i in range(n)]
    return [float(spec)]


def _build_interferer(args, n_samples: int, sample_rate: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Build a length-`n_samples` unit-power interferer template; mix()
    will scale it. Some interferers are inherently signal-dependent
    (AWGN needs a fresh draw each point) — caller decides if/when to
    regenerate."""
    kind = args.interferer
    if kind == "awgn":
        return awgn(n_samples, rng)
    if kind == "tone":
        if args.tone_hz is None:
            raise SystemExit("--tone-hz required for tone interferer")
        return tone(n_samples, args.tone_hz, sample_rate, args.tone_phase)
    if kind == "file":
        if not args.interferer_file:
            raise SystemExit("--interferer-file required for file interferer")
        return load_interferer(args.interferer_file)
    raise SystemExit(f"unknown interferer: {kind}")


def _build_frame_config(args) -> FrameConfig:
    kwargs = {}
    if args.preamble_hex:
        kwargs["preamble"] = bytes.fromhex(args.preamble_hex.replace(" ", ""))
    if args.syncword_hex:
        kwargs["syncword"] = bytes.fromhex(args.syncword_hex.replace(" ", ""))
    if args.header_format:
        kwargs["header_format"] = tuple(
            (n.strip(), int(b)) for n, b in (
                p.split(":", 1) for p in args.header_format.split(","))
        )
    if args.crc:
        kwargs["crc"] = args.crc
    if args.fec:
        kwargs["fec"] = args.fec
    return FrameConfig(**kwargs)


def _resolve_signal_and_params(args) -> tuple[np.ndarray, ReceiveParams]:
    """Borrow the verify_cli logic to load the IQ + ReceiveParams."""
    fmt = detect_format(args.input)
    if fmt == "sigmf-meta":
        params = params_from_sigmf_meta(args.input)
        iq = load_iq(find_sigmf_data_for_meta(args.input))
    elif fmt == "sigmf-data":
        meta = args.input.with_suffix(".sigmf-meta")
        params = params_from_sigmf_meta(meta) if meta.exists() else None
        iq = load_iq(args.input)
    else:
        params = None
        iq = load_iq(args.input)

    if params is None:
        required = ("modulation", "sample_rate", "bitrate")
        missing = [k for k in required if getattr(args, k, None) is None]
        if missing:
            raise SystemExit(
                "Raw cf32 input requires manual params; missing: "
                + ", ".join(f"--{k.replace('_','-')}" for k in missing))
        params = ReceiveParams(
            modulation=args.modulation, sample_rate=args.sample_rate,
            bitrate=args.bitrate, filter_type=args.filter_type or "none",
            span_symbols=args.span_symbols, roll_off=args.roll_off,
            bt_product=args.bt_product,
            gray_coding=(args.gray_coding != "false"),
            initial_phase=args.initial_phase,
            channel_mode=args.channel_mode,
            channel_offsets_hz=_offsets(args.offsets_hz),
            hop_duration_sec=args.hop_duration,
        )
    else:
        if args.modulation:  params.modulation = args.modulation
        if args.sample_rate: params.sample_rate = args.sample_rate
        if args.bitrate:     params.bitrate = args.bitrate
        if args.filter_type: params.filter_type = args.filter_type
        if args.gray_coding: params.gray_coding = args.gray_coding != "false"
    return iq, params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iqgen.evaluate",
        description="SNR/SIR sweep — measure receiver performance vs "
                    "interference power.",
    )
    # Reuse most demod args by parenting on verify_cli's parser.
    p.add_argument("input", type=Path,
                   help=".cf32, .sigmf-data, or .sigmf-meta of CLEAN signal.")
    p.add_argument("-v", "--verbose", action="store_true")

    # Demod params (same as verify_cli)
    from .config import BITS_PER_SYMBOL, VALID_FILTERS
    mp = p.add_argument_group("demod parameters (required for cf32; override SigMF)")
    mp.add_argument("-m", "--modulation", choices=sorted(BITS_PER_SYMBOL))
    mp.add_argument("-s", "--sample-rate", type=float)
    mp.add_argument("-b", "--bitrate", type=float)
    mp.add_argument("-f", "--filter", dest="filter_type",
                     choices=sorted(VALID_FILTERS), default=None)
    mp.add_argument("--span-symbols", type=int, default=10)
    mp.add_argument("--roll-off", type=float, default=0.35)
    mp.add_argument("--bt-product", type=float, default=0.35)
    mp.add_argument("--gray-coding", choices=("true", "false"), default=None)
    mp.add_argument("--initial-phase", type=float, default=0.0)
    mp.add_argument("--offsets-hz", default=None)
    mp.add_argument("--channel-mode", choices=("concurrent", "hopping"),
                     default="concurrent")
    mp.add_argument("--hop-duration", type=float, default=None)

    # Expected bits
    eb = p.add_argument_group("expected bits")
    g = eb.add_mutually_exclusive_group()
    g.add_argument("--bits", help="expected bits string")
    g.add_argument("--bits-file", type=Path,
                    help="file containing expected bits")
    eb.add_argument("--expected-is-payload", action="store_true",
                     help="(with --framing) compare expected against payload only")

    # Interferer
    iv = p.add_argument_group("interferer")
    iv.add_argument("--interferer", choices=("awgn", "tone", "file"),
                     default="awgn")
    iv.add_argument("--interferer-file", type=Path)
    iv.add_argument("--tone-hz", type=float)
    iv.add_argument("--tone-phase", type=float, default=0.0)
    iv.add_argument("--align", choices=("truncate", "tile", "pad"),
                     default="truncate")
    iv.add_argument("--offset-samples", type=int, default=0)
    iv.add_argument("--trim-samples", type=int, default=0,
                     help="exclude leading/trailing samples from power estimation")
    iv.add_argument("--seed", type=int, default=None,
                     help="RNG seed (mainly for AWGN reproducibility)")

    # Sweep
    sw = p.add_argument_group("sweep")
    sw.add_argument("--sweep", default=None,
                     help="dB values: 'start:stop:step', 'v1,v2,...', or single value")
    sw.add_argument("--snr", type=float, default=None,
                     help="single SNR/SIR dB (shorthand for --sweep <value>)")
    sw.add_argument("--mode", choices=("snr", "sir"), default="snr",
                     help="report label only — math is identical")

    # Framing
    fr = p.add_argument_group("framing (optional)")
    fr.add_argument("--framing", action="store_true",
                     help="parse demod output as a framed packet")
    fr.add_argument("--preamble-hex")
    fr.add_argument("--syncword-hex")
    fr.add_argument("--header-format",
                     help='ordered fields, e.g. "length:16,seq:8,type:8"')
    fr.add_argument("--crc", choices=("none", "crc16-ccitt-false", "crc32"))
    fr.add_argument("--fec", choices=("none", "repetition-3", "hamming-7-4"))
    fr.add_argument("--max-sync-distance", type=int, default=None)

    # Output
    out = p.add_argument_group("output")
    out.add_argument("--csv", type=Path, help="write CSV to this path")
    out.add_argument("--plot", type=Path,
                      help="save BER waterfall PNG to this path")
    return p


def run_point(signal: np.ndarray, params: ReceiveParams,
               target_db: float, args,
               rng: np.random.Generator,
               expected: Optional[np.ndarray],
               frame_cfg: Optional[FrameConfig]) -> SweepPoint:
    interferer = _build_interferer(args, signal.size, params.sample_rate, rng)
    mixed, _ = mix(signal, interferer, target_db, mode=args.mode,
                    align=args.align, offset_samples=args.offset_samples,
                    trim_samples=args.trim_samples)

    if frame_cfg is not None:
        exp_payload = expected if (
            expected is not None and args.expected_is_payload) else None
        _, rep = demodulate_frame(
            mixed, params, frame_cfg,
            expected_payload_bits=exp_payload,
            max_sync_distance=args.max_sync_distance)
        n_compared = rep.n_payload_compared
        n_errors = rep.n_payload_errors if rep.n_payload_errors is not None else 0
        ber = (n_errors / n_compared) if n_compared else float("nan")
        return SweepPoint(
            target_db=target_db,
            sync_found=rep.sync_found,
            sync_distance=rep.sync_distance,
            sync_pattern_bits=rep.sync_pattern_bits,
            fec_corrections=rep.fec_corrections,
            crc_scheme=rep.crc_scheme,
            crc_ok=rep.crc_ok if rep.crc_scheme != "none" else None,
            n_compared=n_compared,
            n_errors=n_errors,
            ber=ber,
        )

    # Unframed: just compare bits
    recovered = demodulate(mixed, params)
    if expected is None:
        return SweepPoint(target_db, True, 0, 0, 0, "none", None,
                           0, 0, float("nan"))
    cmp_rep = compare_bits(recovered, expected)
    return SweepPoint(
        target_db=target_db,
        sync_found=True, sync_distance=0, sync_pattern_bits=0,
        fec_corrections=0, crc_scheme="none", crc_ok=None,
        n_compared=cmp_rep.n_compared,
        n_errors=cmp_rep.n_errors,
        ber=cmp_rep.ber,
    )


def _to_row(pt: SweepPoint) -> dict:
    return {
        "target_db": f"{pt.target_db:.3f}",
        "sync_found": int(bool(pt.sync_found)),
        "sync_distance": pt.sync_distance,
        "sync_pattern_bits": pt.sync_pattern_bits,
        "fec_corrections": pt.fec_corrections,
        "crc_scheme": pt.crc_scheme,
        "crc_ok": "" if pt.crc_ok is None else int(bool(pt.crc_ok)),
        "n_compared": pt.n_compared,
        "n_errors": pt.n_errors,
        "ber": (f"{pt.ber:.6e}" if not np.isnan(pt.ber) else "nan"),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    # Sweep values
    if args.sweep and args.snr is not None:
        raise SystemExit("Pass --sweep OR --snr, not both.")
    if args.sweep:
        sweep_vals = _parse_sweep(args.sweep)
    elif args.snr is not None:
        sweep_vals = [args.snr]
    else:
        raise SystemExit("Specify --sweep or --snr.")

    signal, params = _resolve_signal_and_params(args)
    if params.sample_rate <= 0 or params.bitrate <= 0:
        raise SystemExit("sample-rate and bitrate must be > 0")

    expected: Optional[np.ndarray] = None
    if args.bits:
        expected = parse_bits(args.bits)
    elif args.bits_file:
        expected = parse_bits(args.bits_file.read_text())

    frame_cfg = _build_frame_config(args) if args.framing else None

    rng = np.random.default_rng(args.seed)

    points: list[SweepPoint] = []
    log.info("Running %d sweep point(s) on %d-sample signal", len(sweep_vals),
             signal.size)
    for db in sweep_vals:
        # For AWGN draw fresh noise per point (so each SNR is independent).
        pt = run_point(signal, params, db, args, rng, expected, frame_cfg)
        points.append(pt)
        crc_str = "" if pt.crc_ok is None else (" CRC=PASS" if pt.crc_ok else " CRC=FAIL")
        sync_str = (f" sync={pt.sync_distance}/{pt.sync_pattern_bits}"
                    if pt.sync_pattern_bits else "")
        log.info("%6.2f dB:%s%s  errs=%d/%d  BER=%.3e  fec_corr=%d",
                 db, sync_str, crc_str, pt.n_errors, pt.n_compared,
                 pt.ber, pt.fec_corrections)

    # CSV
    out_path = args.csv
    writer = csv.DictWriter(sys.stdout if out_path is None else out_path.open("w"),
                              fieldnames=CSV_FIELDS)
    writer.writeheader()
    for pt in points:
        writer.writerow(_to_row(pt))

    # Plot
    if args.plot:
        _save_plot(points, args.plot, args.mode)

    return 0


def _save_plot(points: list[SweepPoint], path: Path, mode: str) -> None:
    """BER waterfall + optional sync-distance / fec-correction overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db = [p.target_db for p in points]
    ber = [p.ber if not np.isnan(p.ber) else None for p in points]

    fig, ax = plt.subplots(figsize=(8, 5))
    # Filter Nones (sync-not-found points have undefined BER)
    plot_db = [d for d, b in zip(db, ber) if b is not None]
    plot_ber = [b if b > 0 else 1e-6 for b in ber if b is not None]
    if plot_db:
        ax.semilogy(plot_db, plot_ber, marker="o", label="payload BER")

    fec = [p.fec_corrections for p in points]
    ax2 = ax.twinx()
    ax2.plot(db, fec, marker="s", color="tab:orange", alpha=0.5,
              label="FEC corrections", linestyle="--")
    # Highlight CRC fails / sync losses
    fail_db = [p.target_db for p in points
                if (p.crc_ok is False) or (not p.sync_found
                                            and p.sync_pattern_bits)]
    for d in fail_db:
        ax.axvline(d, color="red", alpha=0.15, linewidth=2)

    ax.set_xlabel(f"target {mode.upper()} (dB)")
    ax.set_ylabel("payload BER")
    ax.set_title(f"BER waterfall vs {mode.upper()}")
    ax.grid(True, which="both", alpha=0.3)
    ax2.set_ylabel("FEC corrections", color="tab:orange")
    ax.set_ylim(bottom=1e-6)
    fig.tight_layout()
    fig.savefig(path)
    log.info("saved plot to %s", path)


if __name__ == "__main__":
    sys.exit(main())
