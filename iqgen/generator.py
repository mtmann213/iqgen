from __future__ import annotations

import logging

import numpy as np

from .config import SignalConfig
from .filters import PulseShaper
from .mappers import OQPSKMapper, create_mapper
from .sources import DataSource

log = logging.getLogger(__name__)


class IQGenerator:
    """Orchestrates the bits -> symbols -> upsample -> filter -> normalize pipeline."""

    def __init__(self, config: SignalConfig):
        self.cfg = config

    def generate(self) -> np.ndarray:
        cfg = self.cfg

        # 1. Bits
        source = DataSource.from_config(cfg.source_cfg, bitrate=cfg.bitrate)
        bits = source.get_bits()
        log.info("Source produced %d bits", bits.size)

        # 2. Zero-pad to a whole symbol
        rem = bits.size % cfg.bits_per_symbol
        if rem != 0:
            pad = cfg.bits_per_symbol - rem
            log.warning(
                "Bit count %d is not divisible by bits_per_symbol=%d; "
                "zero-padding with %d bit(s) at the end",
                bits.size, cfg.bits_per_symbol, pad,
            )
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

        if bits.size == 0:
            log.warning("No bits to modulate; returning empty signal")
            return np.zeros(0, dtype=np.complex64)

        # 3. Map to symbols
        mapper = create_mapper(cfg.modulation, cfg.gray_coding, cfg.initial_phase)
        symbols = mapper.map(bits)

        # 4. Upsample (and offset Q for OQPSK).
        # Two modes:
        #   - filter_type == "none": sample-and-hold (NRZ rectangular pulse
        #     of width one symbol). The user gets a usable baseband signal
        #     with the expected sinc(f/Rs) spectrum, instead of an impulse
        #     train (whose spectrum is flat across ±Fs/2).
        #   - any actual filter: zero-stuff, then convolve. The filter's
        #     impulse response IS the pulse shape (RRC/RC/Gaussian/Rect),
        #     so impulse-train input is required.
        sps = cfg.samples_per_symbol
        if cfg.filter_type == "none":
            up = _hold_upsample
        else:
            up = _zero_upsample

        if isinstance(mapper, OQPSKMapper):
            I, Q = symbols
            I_up = up(I.astype(np.float32), sps)
            Q_up = up(Q.astype(np.float32), sps)
            half = sps // 2
            I_up = np.concatenate([I_up, np.zeros(half, dtype=np.float32)])
            Q_up = np.concatenate([np.zeros(half, dtype=np.float32), Q_up])
            signal = (I_up + 1j * Q_up).astype(np.complex64)
        else:
            signal = up(symbols.astype(np.complex64), sps)

        # 5. Pulse-shape (no-op when filter_type == "none")
        shaper = PulseShaper(
            cfg.filter_type, cfg.span_symbols, cfg.samples_per_symbol,
            cfg.roll_off, cfg.bt_product,
        )
        signal = shaper.apply(signal).astype(np.complex64)

        # 6. Normalize the baseband
        signal = _normalize(signal, cfg.normalization)

        # 7. Multi-frequency: mix the baseband to one or more offsets.
        #    Single carrier at offset 0 short-circuits to the original signal.
        signal = _apply_channels(signal, cfg)

        log.info(
            "Generated %d IQ samples (mod=%s, sps=%d, filter=%s, norm=%s, "
            "channels=%s mode=%s)",
            signal.size, cfg.modulation, sps, cfg.filter_type, cfg.normalization,
            len(cfg.channel_offsets_hz), cfg.channel_mode,
        )
        return signal


def _apply_channels(signal: np.ndarray, cfg) -> np.ndarray:
    """Apply the multi-frequency stage.

    concurrent: sum copies of the baseband at each offset (FDM). The result
        is re-normalized so peak/RMS still match cfg.normalization.
    hopping:    each successive `hop_duration_sec` window of the baseband is
        mixed to the next offset in round-robin. Phase is NOT carried across
        hops on purpose — that's how a real hopper looks.
    A single offset of 0.0 is a no-op fast path.
    """
    offsets = cfg.channel_offsets_hz
    if signal.size == 0:
        return signal
    if len(offsets) == 1 and offsets[0] == 0.0:
        return signal

    fs = cfg.sample_rate
    n = signal.size

    if cfg.channel_mode == "concurrent":
        out = np.zeros(n, dtype=np.complex64)
        t = np.arange(n, dtype=np.float64) / fs
        for f in offsets:
            if f == 0.0:
                out += signal
            else:
                lo = np.exp(2j * np.pi * f * t).astype(np.complex64)
                out += (signal * lo).astype(np.complex64)
        # FDM sum can exceed unit amplitude — renormalize.
        return _normalize(out, cfg.normalization)

    # hopping
    spc = max(1, int(round(cfg.hop_duration_sec * fs)))
    out = signal.copy()
    for hop_idx, start in enumerate(range(0, n, spc)):
        end = min(start + spc, n)
        f = offsets[hop_idx % len(offsets)]
        if f == 0.0:
            continue
        t_block = np.arange(start, end, dtype=np.float64) / fs
        lo = np.exp(2j * np.pi * f * t_block).astype(np.complex64)
        out[start:end] = (signal[start:end] * lo).astype(np.complex64)
    return out


def _zero_upsample(sig: np.ndarray, sps: int) -> np.ndarray:
    """Insert sps-1 zeros between samples. Used before a pulse-shaping filter
    convolution: the filter's impulse response becomes the pulse shape."""
    if sig.size == 0:
        return sig
    out = np.zeros(sig.size * sps, dtype=sig.dtype)
    out[::sps] = sig
    return out


def _hold_upsample(sig: np.ndarray, sps: int) -> np.ndarray:
    """Sample-and-hold: repeat each symbol sps times. Equivalent to applying
    a rectangular pulse of width one symbol period. Used when no separate
    pulse-shaping filter is requested."""
    if sig.size == 0:
        return sig
    return np.repeat(sig, sps)


def _normalize(signal: np.ndarray, mode: str) -> np.ndarray:
    if signal.size == 0 or mode == "none":
        return signal
    if mode == "peak":
        peak = float(np.max(np.abs(signal)))
        if peak > 0:
            return (signal / peak).astype(np.complex64)
        return signal
    if mode == "rms":
        rms = float(np.sqrt(np.mean(np.abs(signal) ** 2)))
        if rms > 0:
            return (signal / rms).astype(np.complex64)
        return signal
    raise ValueError(f"Unknown normalization mode: {mode}")
