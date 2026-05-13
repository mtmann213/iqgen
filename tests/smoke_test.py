"""Smoke test: every modulation × every filter, both output formats.

Run from the project root:
    python -m tests.smoke_test
or:
    PYTHONPATH=. python tests/smoke_test.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

# Make package importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iqgen.config import SignalConfig
from iqgen.generator import IQGenerator
from iqgen.verifier import ReceiveParams, compare_bits, demodulate
from iqgen.writers import Cf32Writer, SigMFWriter


MODS = ["bpsk", "dbpsk", "pi2_bpsk",
        "qpsk", "dqpsk", "pi4_qpsk", "oqpsk",
        "8psk", "d8psk", "pi4_8psk"]
FILTERS = ["none", "root_raised_cosine", "raised_cosine", "gaussian", "rectangular"]


def base_config(out_dir: Path, mod: str, filt: str, fmt: str = "cf32") -> dict:
    return {
        "signal": {
            "name": "smoke",
            "center_frequency_hz": 915e6,
            "sample_rate": 1e6,
            "output_dir": str(out_dir),
            "timestamp": False,
            "normalization": "peak",
        },
        "source": {"type": "random", "bit_count": 2048, "seed": 1},
        "modulation": {"type": mod, "gray_coding": True, "initial_phase": 0.0},
        "rate": {"bitrate": 100000},
        "pulse_shaping": {
            "filter_type": filt, "span_symbols": 8,
            "roll_off": 0.35, "bt_product": 0.5,
        },
        "output": {"format": fmt, "sigmf": {
            "author": "smoke", "description": "automated test",
            "license": "CC0", "hardware": "synthetic",
        }},
    }


def run() -> int:
    logging.basicConfig(level=logging.ERROR,
                        format="%(levelname)s %(name)s: %(message)s")
    out_dir = Path("smoke_output")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    failures = []
    n_ok = 0
    for mod in MODS:
        for filt in FILTERS:
            for fmt in ("cf32", "sigmf"):
                label = f"{mod:10s} + {filt:20s} + {fmt}"
                try:
                    cfg = SignalConfig.from_dict(base_config(out_dir, mod, filt, fmt))
                    signal = IQGenerator(cfg).generate()
                    assert signal.size > 0, "empty signal"
                    assert signal.dtype == np.complex64, f"bad dtype {signal.dtype}"
                    # peak normalization invariant
                    if cfg.normalization == "peak":
                        assert np.max(np.abs(signal)) <= 1.0 + 1e-5, "peak > 1"
                    writer = SigMFWriter() if fmt == "sigmf" else Cf32Writer()
                    paths = writer.write(signal, cfg)
                    if fmt == "sigmf":
                        data_p, meta_p = paths
                        meta = json.loads(meta_p.read_text())
                        assert meta["annotations"][0]["iqgen:modulation"] == mod
                        assert meta["annotations"][0]["iqgen:pulse_shaping"] == filt
                    n_ok += 1
                    print(f"  OK    {label}")
                except Exception as e:
                    failures.append((label, e))
                    print(f"  FAIL  {label}: {e}")

    # --- Additional edge-case tests ----------------------------------------
    print("\nEdge cases:")
    # Bitstring source
    try:
        cfg = SignalConfig.from_dict({
            **base_config(out_dir, "qpsk", "root_raised_cosine"),
            "source": {"type": "bitstring", "bits": "10110010" * 64},
        })
        s = IQGenerator(cfg).generate()
        assert s.size > 0
        print("  OK    bitstring source")
        n_ok += 1
    except Exception as e:
        failures.append(("bitstring source", e))
        print(f"  FAIL  bitstring source: {e}")

    # File source LSB / MSB
    bits_bin = out_dir / "bits.bin"
    bits_bin.write_bytes(bytes([0xA5, 0x3C, 0xFF, 0x00] * 16))
    for order in ("msb_first", "lsb_first"):
        try:
            cfg = SignalConfig.from_dict({
                **base_config(out_dir, "8psk", "gaussian"),
                "source": {"type": "file", "input_file": str(bits_bin), "bit_order": order},
            })
            s = IQGenerator(cfg).generate()
            assert s.size > 0
            print(f"  OK    file source ({order})")
            n_ok += 1
        except Exception as e:
            failures.append((f"file source {order}", e))
            print(f"  FAIL  file source ({order}): {e}")

    # duration_sec
    try:
        cfg = SignalConfig.from_dict({
            **base_config(out_dir, "bpsk", "none"),
            "source": {"type": "random", "duration_sec": 0.01, "seed": 7},
        })
        s = IQGenerator(cfg).generate()
        # 0.01 s * 100000 bps = 1000 bits, 1 sps... no wait, sps depends on bitrate vs sample_rate.
        # bitrate=100000, bits_per_symbol=1 -> symbol_rate=100000, sps=10. Output ≈ 10000 samples.
        assert s.size > 0
        print("  OK    duration_sec source")
        n_ok += 1
    except Exception as e:
        failures.append(("duration_sec source", e))
        print(f"  FAIL  duration_sec source: {e}")

    # Non-integer sps auto-adjust
    try:
        cfg = SignalConfig.from_dict({
            **base_config(out_dir, "qpsk", "none"),
            "signal": {**base_config(out_dir, "qpsk", "none")["signal"],
                       "sample_rate": 1.234e6},
        })
        assert cfg.samples_per_symbol == round(1.234e6 / 50000)
        assert np.isclose(cfg.sample_rate, cfg.samples_per_symbol * cfg.symbol_rate)
        print(f"  OK    sample_rate auto-adjust (sps={cfg.samples_per_symbol}, "
              f"sample_rate={cfg.sample_rate:g})")
        n_ok += 1
    except Exception as e:
        failures.append(("sample_rate auto-adjust", e))
        print(f"  FAIL  sample_rate auto-adjust: {e}")

    # OQPSK odd-sps auto-bump
    try:
        cfg = SignalConfig.from_dict({
            **base_config(out_dir, "oqpsk", "none"),
            "signal": {**base_config(out_dir, "oqpsk", "none")["signal"],
                       "sample_rate": 550000},  # 550000/50000 = 11 -> bump to 12
        })
        assert cfg.samples_per_symbol % 2 == 0
        assert cfg.samples_per_symbol == 12
        print(f"  OK    oqpsk odd-sps auto-bump (sps={cfg.samples_per_symbol})")
        n_ok += 1
    except Exception as e:
        failures.append(("oqpsk odd-sps", e))
        print(f"  FAIL  oqpsk odd-sps: {e}")

    # Partial-symbol padding
    try:
        cfg = SignalConfig.from_dict({
            **base_config(out_dir, "8psk", "none"),
            "source": {"type": "bitstring", "bits": "1" * 100},  # 100 % 3 = 1
        })
        s = IQGenerator(cfg).generate()
        assert s.size > 0
        print("  OK    partial-symbol zero-pad")
        n_ok += 1
    except Exception as e:
        failures.append(("partial-symbol pad", e))
        print(f"  FAIL  partial-symbol pad: {e}")

    # Multi-frequency: concurrent (FDM)
    try:
        cfg = SignalConfig.from_dict({
            **base_config(out_dir, "bpsk", "none", "sigmf"),
            "channels": {
                "mode": "concurrent",
                "offsets_hz": [-200e3, 0.0, 200e3],
            },
        })
        s = IQGenerator(cfg).generate()
        assert s.size > 0
        # Peak normalization should still hold after FDM sum
        assert np.max(np.abs(s)) <= 1.0 + 1e-5
        _, meta_p = SigMFWriter().write(s, cfg)
        meta = json.loads(meta_p.read_text())
        assert len(meta["annotations"]) == 3, "expected 3 annotations"
        offs = sorted(a["iqgen:offset_hz"] for a in meta["annotations"])
        assert offs == [-200e3, 0.0, 200e3]
        print("  OK    multi-freq concurrent (3 carriers)")
        n_ok += 1
    except Exception as e:
        failures.append(("multi-freq concurrent", e))
        print(f"  FAIL  multi-freq concurrent: {e}")

    # Multi-frequency: hopping
    try:
        cfg = SignalConfig.from_dict({
            **base_config(out_dir, "qpsk", "none", "sigmf"),
            "channels": {
                "mode": "hopping",
                "offsets_hz": [-100e3, 100e3],
                "hop_duration_sec": 0.001,  # 1 ms hops at 1 MHz => 1000 samples/hop
            },
        })
        s = IQGenerator(cfg).generate()
        assert s.size > 0
        _, meta_p = SigMFWriter().write(s, cfg)
        meta = json.loads(meta_p.read_text())
        # We expect at least 2 annotations (one per hop), each with iqgen:offset_hz
        assert len(meta["annotations"]) >= 2
        assert all("iqgen:offset_hz" in a for a in meta["annotations"])
        # Hops should alternate between the two offsets
        seen = {a["iqgen:offset_hz"] for a in meta["annotations"]}
        assert seen == {-100e3, 100e3}
        print(f"  OK    multi-freq hopping ({len(meta['annotations'])} hops)")
        n_ok += 1
    except Exception as e:
        failures.append(("multi-freq hopping", e))
        print(f"  FAIL  multi-freq hopping: {e}")

    # Round-trip: generate, then verify — every modulation × every filter
    print("\nRound-trip (generate -> verify):")
    rng = np.random.default_rng(2024)
    src_bits = rng.integers(0, 2, 600, dtype=np.uint8)
    src_bitstring = "".join(str(int(b)) for b in src_bits.tolist())
    for mod in MODS:
        for filt in FILTERS:
            label = f"{mod:10s} + {filt:20s}"
            try:
                cfg = SignalConfig.from_dict({
                    **base_config(out_dir, mod, filt, "cf32"),
                    "source": {"type": "bitstring", "bits": src_bitstring},
                })
                signal = IQGenerator(cfg).generate()
                params = ReceiveParams(
                    modulation=mod, sample_rate=cfg.sample_rate,
                    bitrate=cfg.bitrate, filter_type=filt,
                    span_symbols=cfg.span_symbols, roll_off=cfg.roll_off,
                    bt_product=cfg.bt_product, gray_coding=cfg.gray_coding,
                    initial_phase=cfg.initial_phase,
                )
                recovered = demodulate(signal, params)
                report = compare_bits(recovered, src_bits)
                if report.n_errors == 0:
                    n_ok += 1
                    print(f"  OK    {label}  recovered {report.n_recovered_bits}, "
                          f"errors 0/{report.n_compared}")
                else:
                    failures.append((f"round-trip {label}",
                                     RuntimeError(f"{report.n_errors} bit errors")))
                    print(f"  FAIL  {label}  {report.n_errors} errors")
            except Exception as e:
                failures.append((f"round-trip {label}", e))
                print(f"  FAIL  {label}: {e}")

    # Multi-frequency: Nyquist violation must raise
    try:
        SignalConfig.from_dict({
            **base_config(out_dir, "bpsk", "none"),
            "channels": {"mode": "concurrent", "offsets_hz": [600e3]},
            # sample_rate=1e6 => Nyquist=500kHz; offset=600kHz must fail
        })
        failures.append(("nyquist guard", AssertionError("expected ValueError")))
        print("  FAIL  multi-freq nyquist guard: no error raised")
    except ValueError as e:
        assert "Nyquist" in str(e), f"unexpected message: {e}"
        print("  OK    multi-freq Nyquist guard")
        n_ok += 1
    except Exception as e:
        failures.append(("nyquist guard", e))
        print(f"  FAIL  multi-freq nyquist guard: {e}")

    # ---------------- Framing tests ----------------
    print("\nFraming:")
    from iqgen.framing import FrameConfig, build_frame, parse_frame
    from iqgen.verifier import demodulate_frame

    payload_seed = np.random.default_rng(99).integers(0, 2, 240, dtype=np.uint8)

    # Bit-level round-trips (no modulator in the loop)
    for fec in ("none", "repetition-3", "hamming-7-4"):
        for crc in ("none", "crc16-ccitt-false", "crc32"):
            label = f"frame-bitlevel {fec:<13s} + {crc:<18s}"
            try:
                fc = FrameConfig(fec=fec, crc=crc)
                framed = build_frame(payload_seed, fc)
                rep = parse_frame(framed, fc, expected_payload_bits=payload_seed)
                assert rep.sync_found, "sync not found"
                assert rep.crc_ok, "crc unexpectedly failed"
                assert rep.n_payload_errors == 0, f"errors={rep.n_payload_errors}"
                n_ok += 1
                print(f"  OK    {label}  {rep.short_summary()}")
            except Exception as e:
                failures.append((label, e))
                print(f"  FAIL  {label}: {e}")

    # FEC-correctable single-bit error
    try:
        fc = FrameConfig(fec="hamming-7-4", crc="crc16-ccitt-false")
        framed = build_frame(payload_seed, fc)
        corrupted = framed.copy()
        coded_start = 32 + 32  # past preamble+sync
        corrupted[coded_start + 5] ^= 1  # flip a bit inside first Hamming codeword
        rep = parse_frame(corrupted, fc, expected_payload_bits=payload_seed)
        assert rep.fec_corrections >= 1
        assert rep.crc_ok
        assert rep.n_payload_errors == 0
        print(f"  OK    hamming corrects 1-bit error  "
              f"({rep.fec_corrections} corrections, CRC PASS, 0 payload errors)")
        n_ok += 1
    except Exception as e:
        failures.append(("hamming 1-flip", e))
        print(f"  FAIL  hamming 1-flip: {e}")

    # FEC-uncorrectable; CRC catches the residual error
    try:
        fc = FrameConfig(fec="hamming-7-4", crc="crc16-ccitt-false")
        framed = build_frame(payload_seed, fc)
        corrupted = framed.copy()
        coded_start = 32 + 32
        corrupted[coded_start + 5] ^= 1
        corrupted[coded_start + 6] ^= 1   # 2 errors in same codeword: Hamming miscorrects
        rep = parse_frame(corrupted, fc, expected_payload_bits=payload_seed)
        # The CRC should fail OR the payload should have errors — at minimum
        # one of the two diagnostics must surface the problem.
        assert (not rep.crc_ok) or rep.n_payload_errors > 0, \
            "expected CRC fail or payload errors with 2-bit codeword damage"
        print(f"  OK    hamming uncorrectable detected  "
              f"(CRC={'OK' if rep.crc_ok else 'FAIL'}, "
              f"payload errors={rep.n_payload_errors})")
        n_ok += 1
    except Exception as e:
        failures.append(("hamming 2-flip", e))
        print(f"  FAIL  hamming 2-flip: {e}")

    # ---------------- Channel / interference tests ----------------
    print("\nChannel / interference:")
    from iqgen.channel import (add_awgn, add_tone, awgn, measure_power, mix,
                                 tone)

    # SNR accuracy: AWGN at known dB targets
    try:
        rng_ch = np.random.default_rng(11)
        bpsk_like = (rng_ch.choice([-1, 1], 8000).astype(np.float32) + 0j
                     ).astype(np.complex64)
        for target in (20.0, 10.0, 0.0, -10.0):
            mixed, mrep = add_awgn(bpsk_like, snr_db=target, rng=rng_ch)
            noise_only = mixed - bpsk_like
            measured = 10.0 * np.log10(
                measure_power(bpsk_like) / measure_power(noise_only))
            assert abs(measured - target) < 0.3, \
                f"target {target} dB, measured {measured} dB"
        print("  OK    AWGN SNR accuracy (±0.3 dB across +20, +10, 0, -10)")
        n_ok += 1
    except Exception as e:
        failures.append(("awgn snr accuracy", e))
        print(f"  FAIL  awgn snr accuracy: {e}")

    # Tone SIR accuracy
    try:
        sig_test = (np.ones(4000, dtype=np.float32) + 0j).astype(np.complex64)
        mixed, mrep = add_tone(sig_test, sir_db=6.0, freq_hz=2000.0,
                                sample_rate=100000.0)
        tone_only = mixed - sig_test
        measured = 10.0 * np.log10(
            measure_power(sig_test) / measure_power(tone_only))
        assert abs(measured - 6.0) < 0.01
        print(f"  OK    tone SIR accuracy ({measured:+.3f} dB vs target +6 dB)")
        n_ok += 1
    except Exception as e:
        failures.append(("tone sir", e))
        print(f"  FAIL  tone sir: {e}")

    # Channel + framing degradation pattern: pick three SNR points and
    # check that the diagnostics tell the expected story.
    from iqgen.framing import FrameConfig as _FC
    from iqgen.verifier import demodulate_frame as _df
    try:
        cfg_ch = SignalConfig.from_dict({
            **base_config(out_dir, "qpsk", "root_raised_cosine"),
            "source": {"type": "framed",
                        "payload": {"type": "random", "bit_count": 4000,
                                     "seed": 1},
                        "framing": {}},
        })
        gen_ch = IQGenerator(cfg_ch)
        sig_ch = gen_ch.generate()
        exp_ch = gen_ch.source._last_payload
        params_ch = ReceiveParams(
            modulation="qpsk", sample_rate=cfg_ch.sample_rate,
            bitrate=cfg_ch.bitrate, filter_type="root_raised_cosine",
            span_symbols=cfg_ch.span_symbols, roll_off=cfg_ch.roll_off,
            gray_coding=True)
        fc_ch = _FC()
        rng_w = np.random.default_rng(7)
        # 1. High SNR: clean (sync 0, no FEC corrections, no errors)
        mixed_hi, _ = add_awgn(sig_ch, snr_db=20.0, rng=rng_w)
        _, rep_hi = _df(mixed_hi, params_ch, fc_ch, expected_payload_bits=exp_ch,
                         max_sync_distance=16)
        assert rep_hi.sync_found and rep_hi.crc_ok and rep_hi.n_payload_errors == 0
        # 2. Moderate (~-5 dB): FEC engages but still clean
        mixed_md, _ = add_awgn(sig_ch, snr_db=-5.0, rng=rng_w)
        _, rep_md = _df(mixed_md, params_ch, fc_ch, expected_payload_bits=exp_ch,
                         max_sync_distance=16)
        assert rep_md.sync_found, "expected sync at -5 dB"
        # 3. Severe (-14 dB): payload BER should be > 0 and CRC should fail
        mixed_lo, _ = add_awgn(sig_ch, snr_db=-14.0, rng=rng_w)
        _, rep_lo = _df(mixed_lo, params_ch, fc_ch, expected_payload_bits=exp_ch,
                         max_sync_distance=24)
        # We expect either CRC fail or non-zero payload errors at this SNR.
        assert (not rep_lo.crc_ok) or rep_lo.n_payload_errors > 0, \
            "expected detected damage at -14 dB"
        print(f"  OK    AWGN+framing waterfall  "
              f"(hi: CRC PASS, 0 err | md: sync OK, FEC corr {rep_md.fec_corrections}"
              f" | lo: CRC {'PASS' if rep_lo.crc_ok else 'FAIL'}, "
              f"errors={rep_lo.n_payload_errors})")
        n_ok += 1
    except Exception as e:
        failures.append(("awgn+framing waterfall", e))
        print(f"  FAIL  awgn+framing waterfall: {e}")

    # Modulated round-trip: QPSK + RRC + framed source + Hamming + CRC-16
    try:
        cfg = SignalConfig.from_dict({
            **base_config(out_dir, "qpsk", "root_raised_cosine"),
            "source": {
                "type": "framed",
                "payload": {"type": "random", "bit_count": 240, "seed": 11},
                "framing": {"fec": "hamming-7-4", "crc": "crc16-ccitt-false"},
            },
        })
        gen = IQGenerator(cfg)
        signal = gen.generate()
        expected = gen.source._last_payload
        params = ReceiveParams(
            modulation="qpsk", sample_rate=cfg.sample_rate, bitrate=cfg.bitrate,
            filter_type="root_raised_cosine", span_symbols=cfg.span_symbols,
            roll_off=cfg.roll_off, gray_coding=True, initial_phase=0.0)
        _, rep = demodulate_frame(signal, params, FrameConfig(),
                                   expected_payload_bits=expected)
        assert rep.sync_found and rep.crc_ok and rep.n_payload_errors == 0
        print(f"  OK    modulated framed round-trip (QPSK+RRC)  {rep.short_summary()}")
        n_ok += 1
    except Exception as e:
        failures.append(("framed modulated round-trip", e))
        print(f"  FAIL  framed modulated round-trip: {e}")

    print(f"\n{n_ok} passed, {len(failures)} failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(run())
