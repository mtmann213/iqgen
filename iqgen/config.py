from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger(__name__)


BITS_PER_SYMBOL = {
    "bpsk": 1, "dbpsk": 1,
    "qpsk": 2, "dqpsk": 2, "pi4_qpsk": 2, "oqpsk": 2,
    "8psk": 3, "d8psk": 3, "pi4_8psk": 3,
}

VALID_NORMALIZATIONS = {"peak", "rms", "none"}
VALID_FORMATS = {"cf32", "sigmf"}
VALID_FILTERS = {"none", "root_raised_cosine", "raised_cosine", "gaussian", "rectangular"}


def _num(v: Any) -> float:
    if isinstance(v, str):
        return float(v)
    return float(v)


@dataclass
class SignalConfig:
    """Validated, derived configuration. Construct via `from_yaml`."""

    raw: dict = field(repr=False)

    # Populated by _validate_and_derive
    name: str = ""
    center_frequency_hz: float = 0.0
    sample_rate: float = 0.0
    normalization: str = "peak"
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    timestamp: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"

    modulation: str = ""
    bits_per_symbol: int = 0
    gray_coding: bool = True
    initial_phase: float = 0.0

    bitrate: float = 0.0
    symbol_rate: float = 0.0
    samples_per_symbol: int = 0

    filter_type: str = "none"
    span_symbols: int = 10
    roll_off: float = 0.35
    bt_product: float = 0.35
    num_taps: int = 0

    format: str = "cf32"
    sigmf_meta: dict = field(default_factory=dict)
    source_cfg: dict = field(default_factory=dict)

    # Multi-frequency (absent `channels:` block => defaults = single carrier at offset 0)
    channel_mode: str = "concurrent"          # "concurrent" | "hopping"
    channel_offsets_hz: list = field(default_factory=lambda: [0.0])
    hop_duration_sec: float | None = None     # only used when channel_mode == "hopping"

    @classmethod
    def from_yaml(cls, path) -> "SignalConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")
        cfg = cls(raw=raw)
        cfg._validate_and_derive()
        return cfg

    @classmethod
    def from_dict(cls, raw: dict) -> "SignalConfig":
        cfg = cls(raw=raw)
        cfg._validate_and_derive()
        return cfg

    def _validate_and_derive(self) -> None:
        s = self.raw.get("signal") or {}
        r = self.raw.get("rate") or {}
        m = self.raw.get("modulation") or {}
        ps = self.raw.get("pulse_shaping") or {}
        out = self.raw.get("output") or {}

        if "sample_rate" not in s:
            raise ValueError("signal.sample_rate is required")
        if "bitrate" not in r:
            raise ValueError("rate.bitrate is required")
        if "type" not in m:
            raise ValueError("modulation.type is required")

        # --- signal ---
        self.name = str(s.get("name", "signal"))
        self.center_frequency_hz = _num(s.get("center_frequency_hz", 0.0))
        self.sample_rate = _num(s["sample_rate"])
        norm = str(s.get("normalization", "peak")).lower()
        if norm not in VALID_NORMALIZATIONS:
            raise ValueError(f"signal.normalization must be one of {VALID_NORMALIZATIONS}")
        self.normalization = norm
        self.output_dir = Path(s.get("output_dir", "./output"))
        self.timestamp = bool(s.get("timestamp", True))
        self.timestamp_format = str(s.get("timestamp_format", "%Y%m%d_%H%M%S"))

        # --- modulation ---
        self.modulation = str(m["type"]).lower()
        if self.modulation not in BITS_PER_SYMBOL:
            raise ValueError(
                f"Unknown modulation '{self.modulation}'. "
                f"Valid: {sorted(BITS_PER_SYMBOL)}"
            )
        self.bits_per_symbol = BITS_PER_SYMBOL[self.modulation]
        self.gray_coding = bool(m.get("gray_coding", True))
        self.initial_phase = float(m.get("initial_phase", 0.0))

        # --- rate (interconnected) ---
        self.bitrate = _num(r["bitrate"])
        if self.bitrate <= 0:
            raise ValueError("rate.bitrate must be positive")

        derived_sr = self.bitrate / self.bits_per_symbol
        if "symbol_rate" in r:
            log.warning(
                "rate.symbol_rate is calculated from bitrate/bits_per_symbol; "
                "ignoring manual value %s and using %g",
                r["symbol_rate"], derived_sr,
            )
        self.symbol_rate = derived_sr

        sps_float = self.sample_rate / self.symbol_rate
        sps = int(round(sps_float))
        if sps < 1:
            raise ValueError(
                f"samples_per_symbol={sps_float:g} < 1; "
                f"sample_rate ({self.sample_rate:g}) must exceed symbol_rate ({self.symbol_rate:g})"
            )
        if not np.isclose(sps_float, sps):
            new_sr = sps * self.symbol_rate
            log.warning(
                "samples_per_symbol = sample_rate/symbol_rate = %.6f is non-integer; "
                "auto-adjusting sample_rate %g -> %g so samples_per_symbol = %d",
                sps_float, self.sample_rate, new_sr, sps,
            )
            self.sample_rate = new_sr
        if "samples_per_symbol" in r:
            log.warning(
                "rate.samples_per_symbol is calculated from sample_rate/symbol_rate; "
                "ignoring manual value %s and using %d",
                r["samples_per_symbol"], sps,
            )

        if self.modulation == "oqpsk" and sps % 2 != 0:
            new_sps = sps + 1
            new_sr = new_sps * self.symbol_rate
            log.warning(
                "OQPSK requires even samples_per_symbol; bumping %d -> %d "
                "(sample_rate %g -> %g)",
                sps, new_sps, self.sample_rate, new_sr,
            )
            sps = new_sps
            self.sample_rate = new_sr
        self.samples_per_symbol = sps

        # --- pulse shaping (depends on sps) ---
        ft = str(ps.get("filter_type", "none")).lower()
        if ft not in VALID_FILTERS:
            raise ValueError(f"Unknown pulse_shaping.filter_type '{ft}'. Valid: {sorted(VALID_FILTERS)}")
        self.filter_type = ft
        self.span_symbols = int(ps.get("span_symbols", 10))
        self.roll_off = float(ps.get("roll_off", 0.35))
        self.bt_product = float(ps.get("bt_product", 0.35))
        self.num_taps = self.span_symbols * self.samples_per_symbol + 1

        # --- output ---
        fmt = str(out.get("format", "cf32")).lower()
        if fmt not in VALID_FORMATS:
            raise ValueError(f"output.format must be one of {VALID_FORMATS}")
        self.format = fmt
        self.sigmf_meta = out.get("sigmf") or {}

        # --- source ---
        self.source_cfg = self.raw.get("source") or {"type": "random", "bit_count": 1024}

        # --- multi-frequency channels (optional) ---
        ch = self.raw.get("channels") or {}
        mode = str(ch.get("mode", "concurrent")).lower()
        if mode not in ("concurrent", "hopping"):
            raise ValueError(
                f"channels.mode must be 'concurrent' or 'hopping' (got '{mode}')"
            )
        self.channel_mode = mode

        offsets_raw = ch.get("offsets_hz", [0.0])
        if not isinstance(offsets_raw, (list, tuple)) or len(offsets_raw) == 0:
            raise ValueError("channels.offsets_hz must be a non-empty list")
        self.channel_offsets_hz = [float(x) for x in offsets_raw]

        # Bandwidth budget per carrier (single-sided, conservative)
        if self.filter_type in ("root_raised_cosine", "raised_cosine"):
            bw_single = self.symbol_rate * (1.0 + self.roll_off) / 2.0
        else:
            # NRZ rectangular / Gaussian: main lobe is ~symbol_rate
            bw_single = self.symbol_rate
        nyq = self.sample_rate / 2.0
        for f in self.channel_offsets_hz:
            if abs(f) + bw_single > nyq + 1e-9:
                min_sr = 2.0 * (abs(f) + bw_single)
                raise ValueError(
                    f"Channel offset {f:g} Hz with single-sided bandwidth "
                    f"{bw_single:g} Hz exceeds Nyquist (sample_rate/2 = "
                    f"{nyq:g} Hz). Increase signal.sample_rate to at least "
                    f"{min_sr:g} Hz."
                )

        if mode == "hopping":
            hd = ch.get("hop_duration_sec")
            if hd is None:
                raise ValueError(
                    "channels.hop_duration_sec is required when channels.mode=hopping"
                )
            hd = float(hd)
            if hd <= 0:
                raise ValueError("channels.hop_duration_sec must be positive")
            self.hop_duration_sec = hd
