"""CLI for the iqgen verifier.

Usage examples:
    # SigMF input (auto-reads parameters from .sigmf-meta):
    python -m iqgen.verify_cli path/to/recording.sigmf-meta --bits "10110010..."
    python -m iqgen.verify_cli path/to/recording.sigmf-data --bits-file bits.txt

    # Raw cf32 (no metadata) — supply parameters manually:
    python -m iqgen.verify_cli sig.cf32 --modulation qpsk --sample-rate 1e6 \\
        --bitrate 100e3 --filter root_raised_cosine --roll-off 0.35

    # Multi-frequency hopping:
    python -m iqgen.verify_cli sig.cf32 -m qpsk -s 1e6 -b 100e3 -f none \\
        --offsets-hz -100e3,100e3 --channel-mode hopping --hop-duration 0.001
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .verifier import (ReceiveParams, compare_bits, demodulate, detect_format,
                        find_sigmf_data_for_meta, load_iq, params_from_sigmf_meta,
                        parse_bits)
from .config import BITS_PER_SYMBOL, VALID_FILTERS

log = logging.getLogger(__name__)


def _offsets(raw: str | None) -> list[float]:
    if not raw:
        return [0.0]
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iqgen.verify_cli",
        description="Recover bits from a .cf32 or .sigmf-data file produced by iqgen.",
    )
    p.add_argument("input", type=Path,
                   help="Path to .cf32, .sigmf-data, or .sigmf-meta file.")
    p.add_argument("-v", "--verbose", action="store_true")

    # Comparison
    cmp = p.add_argument_group("expected bits (optional, for BER report)")
    g = cmp.add_mutually_exclusive_group()
    g.add_argument("--bits", help="Expected bits as a string of 0/1.")
    g.add_argument("--bits-file", type=Path,
                   help="Path to a file containing expected bits (0/1, "
                        "whitespace ignored).")
    cmp.add_argument("--output-bits", type=Path,
                     help="Write recovered bits to this file as a 0/1 string.")

    # Manual params (used when input is cf32 OR to override SigMF metadata)
    mp = p.add_argument_group("parameters (required for cf32; override SigMF)")
    mp.add_argument("-m", "--modulation", choices=sorted(BITS_PER_SYMBOL))
    mp.add_argument("-s", "--sample-rate", type=float, help="Hz")
    mp.add_argument("-b", "--bitrate", type=float, help="bits/sec")
    mp.add_argument("-f", "--filter", dest="filter_type",
                    choices=sorted(VALID_FILTERS), default=None)
    mp.add_argument("--span-symbols", type=int, default=10)
    mp.add_argument("--roll-off", type=float, default=0.35)
    mp.add_argument("--bt-product", type=float, default=0.35)
    mp.add_argument("--gray-coding", choices=("true", "false"), default=None)
    mp.add_argument("--initial-phase", type=float, default=0.0,
                    help="radians (used by differential modulations)")

    mf = p.add_argument_group("multi-frequency (manual params only)")
    mf.add_argument("--offsets-hz",
                    help="Comma-separated baseband carrier offsets. Default 0.")
    mf.add_argument("--channel-mode", choices=("concurrent", "hopping"),
                    default="concurrent")
    mf.add_argument("--hop-duration", type=float,
                    help="seconds; required when channel-mode=hopping")
    return p


def _resolve(args) -> tuple[np.ndarray, ReceiveParams]:
    """Load the IQ file and build the ReceiveParams. SigMF metadata is
    preferred when available; CLI flags override individual fields."""
    fmt = detect_format(args.input)
    if fmt == "sigmf-meta":
        params = params_from_sigmf_meta(args.input)
        iq = load_iq(find_sigmf_data_for_meta(args.input))
    elif fmt == "sigmf-data":
        meta = args.input.with_suffix(".sigmf-meta")
        if meta.exists():
            params = params_from_sigmf_meta(meta)
        else:
            params = None
        iq = load_iq(args.input)
    else:
        params = None
        iq = load_iq(args.input)

    # Build / override from CLI
    if params is None:
        required = ("modulation", "sample_rate", "bitrate")
        missing = [k for k in required if getattr(args, k, None) is None]
        if missing:
            raise SystemExit(
                "Raw .cf32 input requires manual parameters; missing: "
                + ", ".join(f"--{k.replace('_','-')}" for k in missing)
            )
        params = ReceiveParams(
            modulation=args.modulation,
            sample_rate=args.sample_rate,
            bitrate=args.bitrate,
            filter_type=args.filter_type or "none",
            span_symbols=args.span_symbols,
            roll_off=args.roll_off,
            bt_product=args.bt_product,
            gray_coding=(args.gray_coding != "false"),
            initial_phase=args.initial_phase,
            channel_mode=args.channel_mode,
            channel_offsets_hz=_offsets(args.offsets_hz),
            hop_duration_sec=args.hop_duration,
        )
    else:
        # SigMF gave us a baseline; let any CLI override stick.
        if args.modulation:      params.modulation = args.modulation
        if args.sample_rate:     params.sample_rate = args.sample_rate
        if args.bitrate:         params.bitrate = args.bitrate
        if args.filter_type:     params.filter_type = args.filter_type
        if args.gray_coding:     params.gray_coding = args.gray_coding != "false"
        if args.offsets_hz:
            params.channel_offsets_hz = _offsets(args.offsets_hz)
            params.channel_mode = args.channel_mode
            params.hop_duration_sec = args.hop_duration

    return iq, params


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    iq, params = _resolve(args)
    log.info("Loaded %d samples; demodulating as %s @ %g bps (sps=%d)",
             iq.size, params.modulation, params.bitrate, params.samples_per_symbol)

    recovered = demodulate(iq, params)

    if args.output_bits:
        s = "".join(str(int(b)) for b in recovered)
        args.output_bits.write_text(s)
        log.info("Wrote recovered bits to %s (%d bits)", args.output_bits, recovered.size)

    expected = None
    if args.bits:
        expected = parse_bits(args.bits)
    elif args.bits_file:
        expected = parse_bits(args.bits_file.read_text())

    if expected is not None:
        report = compare_bits(recovered, expected)
        print(report)
        return 0 if report.n_errors == 0 else 1

    # No expected bits: print a short preview
    preview = "".join(str(int(b)) for b in recovered[:64])
    suffix = "…" if recovered.size > 64 else ""
    print(f"Recovered {recovered.size} bits: {preview}{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
