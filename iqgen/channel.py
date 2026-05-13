"""Channel / interference layer for iqgen.

Mixes a clean iqgen recording with an interferer (AWGN, CW tone, or
another IQ file) at a target power ratio (SNR or SIR in dB). The
output is `signal + scaled_interferer`, where the interferer is scaled
so that 10·log10(P_signal / P_interferer) == target_db.

Power model: average power over the entire signal, P = mean(|x|²). For
pulse-shaped signals this is slightly pessimistic near the filter tails,
but for typical capture lengths the bias is negligible. Use
`measure_power_active()` if you need to exclude leading/trailing tails.

Conventions:
- All IQ is complex64.
- "Interferer" is the generic name; "noise" (AWGN) and "interference"
  (tone, file) are both treated the same way mathematically — the only
  difference is how the interferer waveform is constructed.
- target_db > 0 means signal is *stronger* than the interferer.
- target_db < 0 means interferer is *stronger* than the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# Power measurement
# =============================================================================

def measure_power(x: np.ndarray) -> float:
    """Average power: mean(|x|²). Returns 0.0 for an empty array."""
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) ** 2))


def measure_power_active(x: np.ndarray, trim_samples: int = 0) -> float:
    """Average power excluding the first/last `trim_samples` samples
    (useful for ignoring pulse-shape filter tails). Returns 0.0 if the
    trim would leave fewer than 1 sample."""
    if x.size <= 2 * trim_samples:
        return 0.0
    body = x[trim_samples:x.size - trim_samples] if trim_samples else x
    return float(np.mean(np.abs(body) ** 2))


# =============================================================================
# Interferer constructors
# =============================================================================

def awgn(n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Unit-power complex AWGN: (randn + j·randn) / √2 → E[|z|²] = 1.
    The mix() function then scales this to the target power."""
    if rng is None:
        rng = np.random.default_rng()
    re = rng.standard_normal(n_samples).astype(np.float32)
    im = rng.standard_normal(n_samples).astype(np.float32)
    return ((re + 1j * im) / np.sqrt(2.0)).astype(np.complex64)


def tone(n_samples: int, freq_hz: float, sample_rate: float,
         phase: float = 0.0) -> np.ndarray:
    """Complex exponential CW tone at `freq_hz`. Unit power (|exp(jθ)|² = 1)."""
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    return np.exp(2j * np.pi * freq_hz * t + 1j * phase).astype(np.complex64)


def from_file(path) -> np.ndarray:
    """Load an interferer from a .cf32 or .sigmf-data file."""
    p = Path(path)
    if p.suffix == ".sigmf-meta":
        # Look for paired .sigmf-data
        data = p.with_suffix(".sigmf-data")
        if not data.exists():
            raise FileNotFoundError(f"No .sigmf-data next to {p}")
        p = data
    return np.fromfile(p, dtype=np.complex64)


# =============================================================================
# Alignment
# =============================================================================

def _align(interferer: np.ndarray, n: int, mode: str,
           offset_samples: int = 0) -> np.ndarray:
    """Resize `interferer` to length `n`. Modes:
      - "truncate": cut to n (or zero-pad if shorter)
      - "tile":     repeat to fill n
      - "pad":      zero-pad to n (only the part of the signal that
                    overlaps with the interferer sees interference)

    `offset_samples` shifts the interferer to start `offset` samples
    into the output (negative = wrap to the right; usually 0).
    """
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    if interferer.size == 0:
        return np.zeros(n, dtype=np.complex64)

    if mode == "tile":
        reps = int(np.ceil(n / interferer.size))
        body = np.tile(interferer, reps)[:n]
    elif mode == "pad":
        body = np.zeros(n, dtype=np.complex64)
        m = min(n, interferer.size)
        body[:m] = interferer[:m]
    elif mode == "truncate":
        if interferer.size >= n:
            body = interferer[:n]
        else:
            body = np.zeros(n, dtype=np.complex64)
            body[:interferer.size] = interferer
    else:
        raise ValueError(f"Unknown align mode: {mode}")

    if offset_samples == 0:
        return body.astype(np.complex64)
    out = np.zeros(n, dtype=np.complex64)
    if offset_samples > 0:
        m = min(n - offset_samples, body.size)
        if m > 0:
            out[offset_samples:offset_samples + m] = body[:m]
    else:  # negative: drop the leading samples
        skip = -offset_samples
        m = min(body.size - skip, n)
        if m > 0:
            out[:m] = body[skip:skip + m]
    return out.astype(np.complex64)


# =============================================================================
# Mix
# =============================================================================

@dataclass
class MixReport:
    signal_power: float
    interferer_power_raw: float       # before scaling
    interferer_power_applied: float   # after scaling
    scale_factor: float
    achieved_db: float                # the actual P_signal/P_interferer in dB
    target_db: float
    mode: str                         # "snr" or "sir" (label only)
    align: str
    n_samples: int

    def __str__(self) -> str:
        return (f"mix: target_{self.mode}={self.target_db:+.2f} dB, "
                f"achieved={self.achieved_db:+.2f} dB, "
                f"signal_P={self.signal_power:.3e}, "
                f"interferer_P={self.interferer_power_applied:.3e}, "
                f"scale={self.scale_factor:.3e}, align={self.align}, "
                f"n={self.n_samples}")


def mix(signal: np.ndarray, interferer: np.ndarray,
        target_db: float,
        mode: str = "snr",
        align: str = "truncate",
        offset_samples: int = 0,
        trim_samples: int = 0) -> tuple[np.ndarray, MixReport]:
    """Combine `signal` + scaled `interferer` so that the resulting
    P_signal / P_interferer == 10^(target_db / 10).

    Returns (mixed, report). The mixed signal has the same length as
    `signal`.

    Parameters
    ----------
    target_db
        Desired SNR (or SIR) in dB. Positive → signal stronger than
        interferer. -inf → no interferer (returns `signal` unchanged).
    mode
        "snr" or "sir" — purely a label for the report; the math is
        identical.
    align
        How to size the interferer to match the signal length: see
        `_align()`.
    offset_samples
        Shift the interferer in time by this many samples (positive =
        delayed; negative = leading samples dropped).
    trim_samples
        For power estimation only: exclude this many leading/trailing
        samples from each side (pulse-shape filter tails).
    """
    if signal.size == 0:
        return signal.copy(), MixReport(0, 0, 0, 0, float("nan"),
                                          target_db, mode, align, 0)

    if np.isneginf(target_db):
        # Convention: -inf means "no interference"
        return signal.astype(np.complex64), MixReport(
            measure_power_active(signal, trim_samples), 0.0, 0.0, 0.0,
            float("inf"), target_db, mode, align, signal.size)

    aligned = _align(interferer, signal.size, align, offset_samples)

    P_sig = measure_power_active(signal, trim_samples)
    P_int_raw = measure_power(aligned)
    if P_sig <= 0:
        raise ValueError("Signal has zero power; can't set a ratio against zero.")
    if P_int_raw <= 0:
        raise ValueError("Interferer has zero power; can't scale to a ratio.")

    # Target: P_sig / (scale² · P_int_raw) == 10^(target_db/10)
    # => scale² = P_sig / (P_int_raw · 10^(target_db/10))
    target_lin = 10.0 ** (target_db / 10.0)
    scale = float(np.sqrt(P_sig / (P_int_raw * target_lin)))

    scaled = (aligned * scale).astype(np.complex64)
    P_int_applied = measure_power(scaled)
    achieved = 10.0 * np.log10(P_sig / P_int_applied) if P_int_applied > 0 else float("inf")

    mixed = (signal + scaled).astype(np.complex64)
    return mixed, MixReport(
        signal_power=P_sig,
        interferer_power_raw=P_int_raw,
        interferer_power_applied=P_int_applied,
        scale_factor=scale,
        achieved_db=achieved,
        target_db=target_db,
        mode=mode,
        align=align,
        n_samples=signal.size,
    )


# =============================================================================
# Convenience: build common interferers, length-matched to a signal
# =============================================================================

def add_awgn(signal: np.ndarray, snr_db: float,
             rng: Optional[np.random.Generator] = None,
             trim_samples: int = 0) -> tuple[np.ndarray, MixReport]:
    """Convenience: add complex AWGN at a target SNR."""
    if rng is None:
        rng = np.random.default_rng()
    noise = awgn(signal.size, rng)
    return mix(signal, noise, snr_db, mode="snr",
               align="truncate", trim_samples=trim_samples)


def add_tone(signal: np.ndarray, sir_db: float, freq_hz: float,
             sample_rate: float, phase: float = 0.0,
             trim_samples: int = 0) -> tuple[np.ndarray, MixReport]:
    """Convenience: add a CW tone interferer at a target SIR."""
    t = tone(signal.size, freq_hz, sample_rate, phase)
    return mix(signal, t, sir_db, mode="sir",
               align="truncate", trim_samples=trim_samples)


def add_file_interferer(signal: np.ndarray, sir_db: float, path,
                         align: str = "tile",
                         offset_samples: int = 0,
                         trim_samples: int = 0) -> tuple[np.ndarray, MixReport]:
    """Convenience: add an IQ file as the interferer at a target SIR."""
    interferer = from_file(path)
    return mix(signal, interferer, sir_db, mode="sir", align=align,
               offset_samples=offset_samples, trim_samples=trim_samples)
