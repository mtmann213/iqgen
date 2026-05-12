"""Plotting for IQ signals. Used by the GUI (embedded) and for headless PNG
exports written alongside generated data files.

This module never imports pyplot or touches the matplotlib backend, so it
plays nicely with the Tk-embedded canvas in gui.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal as sps


def _constellation(ax, iq, max_points=5000):
    if iq.size == 0:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Constellation")
        return
    step = max(1, iq.size // max_points)
    pts = iq[::step][:max_points]
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.4)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("Constellation")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    lim = 1.1 * float(np.max(np.abs(iq)))
    lim = max(lim, 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)


def _iq_vs_time(ax, iq, sample_rate, max_samples=2000):
    if iq.size == 0:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("IQ vs Time")
        return
    n = min(max_samples, iq.size)
    t_ms = np.arange(n) / sample_rate * 1e3
    ax.plot(t_ms, iq[:n].real, label="I", linewidth=0.8)
    ax.plot(t_ms, iq[:n].imag, label="Q", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"IQ vs Time (first {n} of {iq.size} samples)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def _psd(ax, iq, sample_rate, center_freq=0.0):
    if iq.size < 32:
        ax.text(0.5, 0.5, "signal too short for PSD",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("PSD")
        return
    nper = min(2048, iq.size)
    f, Pxx = sps.welch(iq, fs=sample_rate, nperseg=nper,
                        return_onesided=False, detrend=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    if center_freq:
        f_disp = (f + center_freq) / 1e6
        xlabel = "Frequency (MHz)"
    else:
        # auto-scale baseband units
        if sample_rate >= 1e6:
            f_disp, xlabel = f / 1e6, "Baseband freq (MHz)"
        elif sample_rate >= 1e3:
            f_disp, xlabel = f / 1e3, "Baseband freq (kHz)"
        else:
            f_disp, xlabel = f, "Baseband freq (Hz)"
    Pxx_db = 10.0 * np.log10(np.maximum(Pxx, 1e-20))
    ax.plot(f_disp, Pxx_db, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title("Power Spectral Density (Welch)")
    ax.grid(True, alpha=0.3)


def _spectrogram(ax, iq, sample_rate):
    if iq.size < 256:
        ax.text(0.5, 0.5, "signal too short for spectrogram",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Spectrogram")
        return
    nper = min(256, max(64, iq.size // 8))
    f, t, Sxx = sps.spectrogram(iq, fs=sample_rate, nperseg=nper,
                                  return_onesided=False)
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-20))
    if sample_rate >= 1e6:
        f_disp, ylabel = f / 1e6, "Baseband freq (MHz)"
    elif sample_rate >= 1e3:
        f_disp, ylabel = f / 1e3, "Baseband freq (kHz)"
    else:
        f_disp, ylabel = f, "Baseband freq (Hz)"
    ax.pcolormesh(t * 1e3, f_disp, Sxx_db, shading="auto", cmap="viridis")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title("Spectrogram")


def render(fig, iq, sample_rate, center_freq=0.0, title=""):
    """Render the 4-panel figure onto an existing matplotlib Figure."""
    fig.clear()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    _constellation(ax1, iq)
    _iq_vs_time(ax2, iq, sample_rate)
    _psd(ax3, iq, sample_rate, center_freq)
    _spectrogram(ax4, iq, sample_rate)
    if title:
        fig.suptitle(title, fontsize=10)
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    except Exception:
        # tight_layout occasionally complains about colorbar/text overlap
        pass


def save_png(path, iq, sample_rate, center_freq=0.0, title="") -> Path:
    """Headless PNG export. Does not require a display."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, 8), dpi=100)
    FigureCanvasAgg(fig)  # attach canvas
    render(fig, iq, sample_rate, center_freq, title)
    out = Path(path)
    fig.savefig(out, dpi=100)
    return out
