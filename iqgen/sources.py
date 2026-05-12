from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class DataSource:
    """Abstract source of bits. Subclasses must implement get_bits()."""

    def get_bits(self) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def from_config(cfg: dict, bitrate: Optional[float] = None) -> "DataSource":
        src_type = str(cfg.get("type", "random")).lower()
        if src_type == "random":
            n = cfg.get("bit_count")
            dur = cfg.get("duration_sec")
            if dur is not None:
                if bitrate is None:
                    raise ValueError("source.duration_sec requires bitrate to compute bit count")
                n = int(round(float(dur) * float(bitrate)))
                log.info("Computed bit_count=%d from duration_sec=%s @ bitrate=%g",
                         n, dur, bitrate)
            if n is None:
                raise ValueError("random source needs bit_count or duration_sec")
            return RandomSource(int(n), seed=cfg.get("seed"))
        if src_type == "file":
            if "input_file" not in cfg:
                raise ValueError("file source requires input_file")
            return FileSource(cfg["input_file"], cfg.get("bit_order", "msb_first"))
        if src_type == "bitstring":
            if "bits" not in cfg:
                raise ValueError("bitstring source requires bits")
            return BitstringSource(cfg["bits"])
        raise ValueError(f"Unknown source type '{src_type}' (random|file|bitstring)")


class RandomSource(DataSource):
    def __init__(self, n_bits: int, seed: Optional[int] = None):
        if n_bits < 0:
            raise ValueError("bit_count must be non-negative")
        self.n_bits = n_bits
        self.seed = seed

    def get_bits(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.integers(0, 2, size=self.n_bits, dtype=np.uint8)


class FileSource(DataSource):
    def __init__(self, path, bit_order: str = "msb_first"):
        self.path = Path(path)
        bit_order = bit_order.lower()
        if bit_order not in ("msb_first", "lsb_first"):
            raise ValueError(f"bit_order must be msb_first|lsb_first, got {bit_order!r}")
        self.bit_order = bit_order

    def get_bits(self) -> np.ndarray:
        if not self.path.exists():
            raise FileNotFoundError(f"input_file not found: {self.path}")
        data = np.fromfile(self.path, dtype=np.uint8)
        order = "big" if self.bit_order == "msb_first" else "little"
        return np.unpackbits(data, bitorder=order)


class BitstringSource(DataSource):
    def __init__(self, s):
        if not isinstance(s, str):
            s = str(s)
        cleaned = [c for c in s if c in "01"]
        self.bits = np.array([int(c) for c in cleaned], dtype=np.uint8)

    def get_bits(self) -> np.ndarray:
        return self.bits
