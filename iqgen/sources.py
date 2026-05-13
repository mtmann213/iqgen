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
            return _random_source(cfg, bitrate)
        if src_type == "file":
            if "input_file" not in cfg:
                raise ValueError("file source requires input_file")
            return FileSource(cfg["input_file"], cfg.get("bit_order", "msb_first"))
        if src_type == "bitstring":
            if "bits" not in cfg:
                raise ValueError("bitstring source requires bits")
            return BitstringSource(cfg["bits"])
        if src_type == "framed":
            return _framed_source(cfg, bitrate)
        raise ValueError(
            f"Unknown source type '{src_type}' (random|file|bitstring|framed)")


def _random_source(cfg: dict, bitrate: Optional[float]) -> "RandomSource":
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


def _framed_source(cfg: dict, bitrate: Optional[float]) -> "FramedSource":
    """Wraps another source as the payload and emits a fully framed bit
    stream (preamble + sync + FEC(header + payload + CRC)).

    Config layout:
      source:
        type: framed
        payload: { ...nested source config (random/file/bitstring)... }
        framing:
          preamble_hex: "AAAAAAAA"        # optional
          syncword_hex: "1ACFFC1D"        # optional
          header_format: [[length,16],[seq,8],[type,8]]  # optional
          header_values: {seq: 0, type: 0}                # optional
          crc: "crc16-ccitt-false"        # none | crc16-ccitt-false | crc32
          fec: "hamming-7-4"              # none | repetition-3 | hamming-7-4
    """
    from .framing import FrameConfig
    payload_cfg = cfg.get("payload")
    if payload_cfg is None:
        raise ValueError("framed source requires 'payload' (nested source config)")
    payload_src = DataSource.from_config(payload_cfg, bitrate=bitrate)

    f = cfg.get("framing") or {}
    kwargs = {}
    if "preamble_hex" in f:
        kwargs["preamble"] = bytes.fromhex(f["preamble_hex"])
    if "syncword_hex" in f:
        kwargs["syncword"] = bytes.fromhex(f["syncword_hex"])
    if "header_format" in f:
        kwargs["header_format"] = tuple((str(n), int(b)) for n, b in f["header_format"])
    if "crc" in f:
        kwargs["crc"] = str(f["crc"])
    if "fec" in f:
        kwargs["fec"] = str(f["fec"])
    fc = FrameConfig(**kwargs)
    header_values = dict(f.get("header_values") or {})
    return FramedSource(payload_src, fc, header_values)


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


class FramedSource(DataSource):
    """Wraps any DataSource as the payload of a packet frame."""

    def __init__(self, payload_source: "DataSource", frame_config,
                 header_values: Optional[dict] = None):
        self.payload_source = payload_source
        self.frame_config = frame_config
        self.header_values = dict(header_values or {})
        self._last_payload: Optional[np.ndarray] = None
        self._last_frame: Optional[np.ndarray] = None

    def get_bits(self) -> np.ndarray:
        from .framing import build_frame
        payload = self.payload_source.get_bits()
        framed = build_frame(payload, self.frame_config, self.header_values)
        self._last_payload = payload
        self._last_frame = framed
        return framed
