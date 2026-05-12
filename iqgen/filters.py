"""Pulse-shaping filters. Hand-rolled where scipy doesn't provide them directly.

All filters return a 1-D real tap array. Convolution is done by the generator.
Normalization: RRC and RC are normalized to unit energy (sum(h**2)=1);
Gaussian and rectangular to unit DC gain (sum(h)=1). Final amplitude is
fixed by the generator's normalization step.
"""

from __future__ import annotations

import numpy as np


class PulseShaper:
    def __init__(self, filter_type: str, span_symbols: int,
                 samples_per_symbol: int, roll_off: float = 0.35,
                 bt_product: float = 0.35):
        self.filter_type = filter_type.lower()
        self.span = int(span_symbols)
        self.sps = int(samples_per_symbol)
        self.roll_off = float(roll_off)
        self.bt_product = float(bt_product)
        self.num_taps = self.span * self.sps + 1
        self.taps = self._build()

    def _build(self):
        ft = self.filter_type
        if ft == "none":
            return None
        if ft == "rectangular":
            return self._rectangular()
        if ft == "root_raised_cosine":
            return self._rrc()
        if ft == "raised_cosine":
            return self._rc()
        if ft == "gaussian":
            return self._gaussian()
        raise ValueError(f"Unknown filter: {self.filter_type}")

    def _rectangular(self):
        # NRZ pulse spanning one symbol; pad to num_taps so behavior is consistent
        taps = np.zeros(self.num_taps)
        center = self.num_taps // 2
        half = self.sps // 2
        start = center - half
        taps[start:start + self.sps] = 1.0
        taps = taps / np.sum(taps)
        return taps

    def _rrc(self):
        beta = self.roll_off
        sps = self.sps
        N = self.num_taps
        # t expressed in symbol periods, centered
        t = (np.arange(N) - (N - 1) / 2.0) / sps
        taps = np.zeros(N)
        for i, ti in enumerate(t):
            if np.isclose(ti, 0.0):
                taps[i] = 1.0 - beta + 4.0 * beta / np.pi
            elif beta > 0 and np.isclose(abs(ti), 1.0 / (4.0 * beta)):
                taps[i] = (beta / np.sqrt(2.0)) * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                )
            else:
                num = (np.sin(np.pi * ti * (1.0 - beta))
                       + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
                den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
                taps[i] = num / den
        energy = np.sqrt(np.sum(taps ** 2))
        if energy > 0:
            taps = taps / energy
        return taps

    def _rc(self):
        beta = self.roll_off
        sps = self.sps
        N = self.num_taps
        t = (np.arange(N) - (N - 1) / 2.0) / sps
        taps = np.zeros(N)
        for i, ti in enumerate(t):
            if beta > 0 and np.isclose(abs(2.0 * beta * ti), 1.0):
                taps[i] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * beta))
            else:
                denom = 1.0 - (2.0 * beta * ti) ** 2
                taps[i] = np.sinc(ti) * np.cos(np.pi * beta * ti) / denom
        energy = np.sqrt(np.sum(taps ** 2))
        if energy > 0:
            taps = taps / energy
        return taps

    def _gaussian(self):
        # sigma_t in symbol periods: sigma_t = sqrt(ln 2)/(2*pi*BT)
        # convert to samples
        sps = self.sps
        N = self.num_taps
        sigma_t = np.sqrt(np.log(2.0)) / (2.0 * np.pi * self.bt_product)
        sigma_n = sigma_t * sps
        t = (np.arange(N) - (N - 1) / 2.0)
        taps = np.exp(-0.5 * (t / sigma_n) ** 2)
        s = np.sum(taps)
        if s > 0:
            taps = taps / s
        return taps

    def apply(self, signal: np.ndarray) -> np.ndarray:
        if self.taps is None:
            return signal
        if signal.size == 0:
            return signal
        return np.convolve(signal, self.taps.astype(signal.real.dtype if np.iscomplexobj(signal) else signal.dtype),
                           mode="same")
