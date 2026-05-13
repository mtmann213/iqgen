"""Tkinter GUI for the iqgen verifier.

Launch:
    python -m iqgen.verifier_gui

Workflow:
  1. Pick an IQ file (.cf32, .sigmf-meta, or .sigmf-data).
  2. If SigMF, parameters auto-populate from the metadata; with cf32, fill
     them in manually.
  3. Paste expected bits (or load from a file) to get a BER report. Leave
     blank to just see the recovered bits.
  4. Click Verify. A constellation plot of the recovered symbols is drawn.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from iqgen.config import BITS_PER_SYMBOL, VALID_FILTERS
    from iqgen.verifier import (ReceiveParams, compare_bits, demodulate,
                                detect_format, find_sigmf_data_for_meta,
                                load_iq, params_from_sigmf_meta, parse_bits)
else:
    from .config import BITS_PER_SYMBOL, VALID_FILTERS
    from .verifier import (ReceiveParams, compare_bits, demodulate,
                            detect_format, find_sigmf_data_for_meta, load_iq,
                            params_from_sigmf_meta, parse_bits)

log = logging.getLogger(__name__)

MODULATIONS = sorted(BITS_PER_SYMBOL)
FILTERS = sorted(VALID_FILTERS)
CHANNEL_MODES = ["concurrent", "hopping"]


class VerifierGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("iqgen — IQ Verifier")
        self.root.geometry("1200x800")

        self.v: dict[str, tk.Variable] = {
            "input_path": tk.StringVar(value=""),
            "modulation": tk.StringVar(value="bpsk"),
            "sample_rate": tk.StringVar(value=""),
            "bitrate": tk.StringVar(value=""),
            "filter_type": tk.StringVar(value="none"),
            "span_symbols": tk.StringVar(value="10"),
            "roll_off": tk.StringVar(value="0.35"),
            "bt_product": tk.StringVar(value="0.35"),
            "gray_coding": tk.BooleanVar(value=True),
            "initial_phase": tk.StringVar(value="0.0"),
            "offsets_hz": tk.StringVar(value="0"),
            "channel_mode": tk.StringVar(value="concurrent"),
            "hop_duration_sec": tk.StringVar(value=""),
            "expected_bits_path": tk.StringVar(value=""),
            "_status": tk.StringVar(value="Pick an IQ file to begin."),
        }

        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        left = ttk.Frame(main, width=520)
        right = ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)

        self._build_form(left)
        self._build_plot(right)
        self._build_log(left)

    # ---------- form ----------
    def _build_form(self, parent):
        # File picker
        fp = ttk.LabelFrame(parent, text="Input IQ file", padding=6)
        fp.pack(fill=tk.X, padx=4, pady=4)
        ttk.Entry(fp, textvariable=self.v["input_path"]).grid(
            row=0, column=0, sticky="ew", padx=2)
        ttk.Button(fp, text="Browse…",
                    command=self._browse_input).grid(row=0, column=1, padx=2)
        ttk.Button(fp, text="Auto-fill from SigMF",
                    command=self._autofill).grid(row=0, column=2, padx=2)
        fp.grid_columnconfigure(0, weight=1)

        # Parameters
        pf = ttk.LabelFrame(parent, text="Parameters", padding=6)
        pf.pack(fill=tk.X, padx=4, pady=4)
        r = 0
        self._combo(pf, r, "Modulation", self.v["modulation"], MODULATIONS); r += 1
        self._entry(pf, r, "Sample rate (Hz)", self.v["sample_rate"]); r += 1
        self._entry(pf, r, "Bitrate (bps)", self.v["bitrate"]); r += 1
        self._combo(pf, r, "Filter", self.v["filter_type"], FILTERS); r += 1
        self._entry(pf, r, "Span (symbols)", self.v["span_symbols"]); r += 1
        self._entry(pf, r, "Roll-off (RRC/RC)", self.v["roll_off"]); r += 1
        self._entry(pf, r, "BT product (Gaussian)", self.v["bt_product"]); r += 1
        ttk.Checkbutton(pf, text="Gray coding",
                         variable=self.v["gray_coding"]).grid(
            row=r, column=1, sticky="w"); r += 1
        self._entry(pf, r, "Initial phase (rad)", self.v["initial_phase"]); r += 1

        # Multi-frequency
        mf = ttk.LabelFrame(parent, text="Multi-frequency", padding=6)
        mf.pack(fill=tk.X, padx=4, pady=4)
        self._entry(mf, 0, "Offsets (Hz, comma-sep)", self.v["offsets_hz"])
        self._combo(mf, 1, "Channel mode", self.v["channel_mode"], CHANNEL_MODES)
        self._entry(mf, 2, "Hop duration (s)", self.v["hop_duration_sec"])

        # Expected bits
        eb = ttk.LabelFrame(parent, text="Expected bits (optional)", padding=6)
        eb.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(eb, text="File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(eb, textvariable=self.v["expected_bits_path"]).grid(
            row=0, column=1, sticky="ew", padx=2)
        ttk.Button(eb, text="Browse…",
                    command=self._browse_expected).grid(row=0, column=2, padx=2)
        ttk.Label(eb, text="Or paste:").grid(row=1, column=0, sticky="nw", pady=4)
        self.expected_text = tk.Text(eb, height=4, wrap="word",
                                       font=("TkFixedFont", 9))
        self.expected_text.grid(row=1, column=1, columnspan=2,
                                 sticky="ew", padx=2, pady=4)
        eb.grid_columnconfigure(1, weight=1)

        # Actions
        af = ttk.Frame(parent)
        af.pack(fill=tk.X, padx=4, pady=6)
        ttk.Button(af, text="Verify", command=self.verify).pack(side=tk.LEFT)
        ttk.Button(af, text="Save recovered bits…",
                    command=self.save_recovered).pack(side=tk.LEFT, padx=4)
        ttk.Label(af, textvariable=self.v["_status"],
                   foreground="#444").pack(side=tk.LEFT, padx=8)

    def _build_log(self, parent):
        ttk.Label(parent, text="Recovered bits / report").pack(
            anchor="w", padx=4, pady=(8, 0))
        self.result_text = tk.Text(parent, height=14, wrap="word",
                                     font=("TkFixedFont", 9))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _build_plot(self, parent):
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.text(0.5, 0.5, "Run Verify to see recovered constellation",
                       ha="center", va="center", color="#888")
        self.canvas.draw()

    # ---------- helpers ----------
    def _entry(self, parent, r, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(parent, textvariable=var).grid(
            row=r, column=1, columnspan=2, sticky="ew", padx=2, pady=2)
        parent.grid_columnconfigure(1, weight=1)

    def _combo(self, parent, r, label, var, values):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=2, pady=2)
        ttk.Combobox(parent, textvariable=var, values=values,
                       state="readonly").grid(
            row=r, column=1, columnspan=2, sticky="ew", padx=2, pady=2)
        parent.grid_columnconfigure(1, weight=1)

    def _browse_input(self):
        p = filedialog.askopenfilename(
            title="Choose IQ file",
            filetypes=[("IQ files", "*.cf32 *.sigmf-data *.sigmf-meta"),
                       ("All", "*")],
        )
        if p:
            self.v["input_path"].set(p)
            if detect_format(p) in ("sigmf-meta", "sigmf-data"):
                self._autofill()

    def _browse_expected(self):
        p = filedialog.askopenfilename(
            title="Choose expected-bits file",
            filetypes=[("Text", "*.txt"), ("All", "*")],
        )
        if p:
            self.v["expected_bits_path"].set(p)

    def _autofill(self):
        path = self.v["input_path"].get().strip()
        if not path:
            return
        p = Path(path)
        meta_path = p if p.suffix == ".sigmf-meta" else p.with_suffix(".sigmf-meta")
        if not meta_path.exists():
            self.v["_status"].set(f"No .sigmf-meta next to {p.name}")
            return
        try:
            params = params_from_sigmf_meta(meta_path)
        except Exception as e:
            messagebox.showerror("Could not read SigMF metadata", str(e))
            return
        self.v["modulation"].set(params.modulation)
        self.v["sample_rate"].set(f"{params.sample_rate:g}")
        self.v["bitrate"].set(f"{params.bitrate:g}")
        self.v["filter_type"].set(params.filter_type)
        self.v["roll_off"].set(f"{params.roll_off:g}")
        self.v["bt_product"].set(f"{params.bt_product:g}")
        self.v["gray_coding"].set(params.gray_coding)
        self.v["offsets_hz"].set(
            ",".join(f"{o:g}" for o in params.channel_offsets_hz))
        self.v["channel_mode"].set(params.channel_mode)
        self.v["hop_duration_sec"].set(
            "" if params.hop_duration_sec is None else f"{params.hop_duration_sec:g}")
        self.v["_status"].set(f"Auto-filled from {meta_path.name}")

    def _build_params(self) -> ReceiveParams:
        def _f(name, default=None):
            s = self.v[name].get().strip()
            return float(s) if s else default
        offsets_raw = self.v["offsets_hz"].get().strip()
        offsets = [float(x) for x in offsets_raw.split(",")] if offsets_raw else [0.0]
        return ReceiveParams(
            modulation=self.v["modulation"].get(),
            sample_rate=_f("sample_rate") or 0.0,
            bitrate=_f("bitrate") or 0.0,
            filter_type=self.v["filter_type"].get(),
            span_symbols=int(_f("span_symbols", 10) or 10),
            roll_off=_f("roll_off", 0.35) or 0.35,
            bt_product=_f("bt_product", 0.35) or 0.35,
            gray_coding=bool(self.v["gray_coding"].get()),
            initial_phase=_f("initial_phase", 0.0) or 0.0,
            channel_mode=self.v["channel_mode"].get() or "concurrent",
            channel_offsets_hz=offsets,
            hop_duration_sec=_f("hop_duration_sec"),
        )

    def _resolve_iq(self) -> np.ndarray:
        path = self.v["input_path"].get().strip()
        if not path:
            raise ValueError("Pick an IQ file first.")
        p = Path(path)
        fmt = detect_format(p)
        if fmt == "sigmf-meta":
            return load_iq(find_sigmf_data_for_meta(p))
        return load_iq(p)

    def _get_expected(self) -> Optional[np.ndarray]:
        pasted = self.expected_text.get("1.0", "end").strip()
        if pasted:
            arr = parse_bits(pasted)
            return arr if arr.size else None
        path = self.v["expected_bits_path"].get().strip()
        if path:
            return parse_bits(Path(path).read_text())
        return None

    # ---------- actions ----------
    def verify(self):
        try:
            iq = self._resolve_iq()
            params = self._build_params()
        except Exception as e:
            messagebox.showerror("Bad input", str(e))
            return
        if params.sample_rate <= 0 or params.bitrate <= 0:
            messagebox.showerror("Missing parameters",
                                  "Sample rate and bitrate are required.")
            return

        try:
            recovered = demodulate(iq, params)
        except Exception as e:
            messagebox.showerror("Demodulation failed", str(e))
            return
        self._recovered = recovered

        expected = self._get_expected()
        lines = []
        if expected is not None:
            report = compare_bits(recovered, expected)
            lines.append(str(report))
            self.v["_status"].set(
                f"BER {report.ber:.3e}" if report.n_compared else "no comparison")
        else:
            self.v["_status"].set(f"Recovered {recovered.size} bits")

        s = "".join(str(int(b)) for b in recovered)
        lines.append("")
        lines.append(f"Recovered bits ({recovered.size}):")
        lines.append(s)
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "\n".join(lines))

        self._draw_constellation(iq, params, recovered)

    def save_recovered(self):
        recovered = getattr(self, "_recovered", None)
        if recovered is None or recovered.size == 0:
            messagebox.showinfo("Nothing to save", "Run Verify first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".txt",
                                           filetypes=[("Text", "*.txt"), ("All", "*")])
        if not p:
            return
        Path(p).write_text("".join(str(int(b)) for b in recovered))

    def _draw_constellation(self, iq: np.ndarray, params: ReceiveParams,
                             recovered: np.ndarray):
        # Re-run the demod pipeline to expose the sampled symbols (cheap).
        from .verifier import (_downconvert, _sample_single, _sample_oqpsk)
        from .filters import PulseShaper
        z = _downconvert(iq, params)
        if params.filter_type == "root_raised_cosine":
            z = PulseShaper(params.filter_type, params.span_symbols,
                              params.samples_per_symbol, params.roll_off,
                              params.bt_product).apply(z).astype(np.complex64)
        sps = params.samples_per_symbol
        if params.modulation == "oqpsk":
            sym = _sample_oqpsk(z, sps, params.filter_type)
        else:
            sym = _sample_single(z, sps, params.filter_type)

        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        if sym.size:
            step = max(1, sym.size // 5000)
            pts = sym[::step]
            ax.scatter(pts.real, pts.imag, s=6, alpha=0.5)
            lim = 1.2 * float(np.max(np.abs(pts)))
            lim = max(lim, 1.2)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_title(f"Recovered symbols — {params.modulation} "
                      f"({sym.size} symbols, {recovered.size} bits)")
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw()


def main():
    root = tk.Tk()
    VerifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
