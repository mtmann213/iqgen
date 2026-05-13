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
                                demodulate_frame, detect_format,
                                find_sigmf_data_for_meta, load_iq,
                                params_from_sigmf_meta, parse_bits)
    from iqgen.framing import FrameConfig
else:
    from .config import BITS_PER_SYMBOL, VALID_FILTERS
    from .verifier import (ReceiveParams, compare_bits, demodulate,
                            demodulate_frame, detect_format,
                            find_sigmf_data_for_meta, load_iq,
                            params_from_sigmf_meta, parse_bits)
    from .framing import FrameConfig

log = logging.getLogger(__name__)

MODULATIONS = sorted(BITS_PER_SYMBOL)
FILTERS = sorted(VALID_FILTERS)
CHANNEL_MODES = ["concurrent", "hopping"]
CRC_OPTIONS = ["none", "crc16-ccitt-false", "crc32"]
FEC_OPTIONS = ["none", "repetition-3", "hamming-7-4"]


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
            # Framing
            "frame_enabled": tk.BooleanVar(value=False),
            "frame_preamble_hex": tk.StringVar(value="AAAAAAAA"),
            "frame_sync_hex": tk.StringVar(value="1ACFFC1D"),
            "frame_header_format": tk.StringVar(value="length:16,seq:8,type:8"),
            "frame_crc": tk.StringVar(value="crc16-ccitt-false"),
            "frame_fec": tk.StringVar(value="hamming-7-4"),
            "frame_max_sync_dist": tk.StringVar(value=""),
            "expected_is_payload": tk.BooleanVar(value=True),
            "_status": tk.StringVar(value="Pick an IQ file to begin."),
        }

        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        left_wrap = ttk.Frame(main, width=560)
        right = ttk.Frame(main)
        main.add(left_wrap, weight=0)
        main.add(right, weight=1)

        # Scrollable left panel — framing section adds height.
        left_canvas = tk.Canvas(left_wrap, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(left_wrap, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)
        left = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_left_configure(_evt):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfigure(left_window, width=left_canvas.winfo_width())

        left.bind("<Configure>", _on_left_configure)
        left_canvas.bind("<Configure>", _on_left_configure)
        # Mouse-wheel scroll on the left pane
        left_canvas.bind_all(
            "<MouseWheel>",
            lambda e: left_canvas.yview_scroll(int(-e.delta / 120), "units"))
        left_canvas.bind_all(
            "<Button-4>", lambda e: left_canvas.yview_scroll(-1, "units"))
        left_canvas.bind_all(
            "<Button-5>", lambda e: left_canvas.yview_scroll(+1, "units"))

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

        # Framing
        ff = ttk.LabelFrame(parent, text="Framing", padding=6)
        ff.pack(fill=tk.X, padx=4, pady=4)
        ttk.Checkbutton(ff, text="Enable framing (parse packet structure)",
                         variable=self.v["frame_enabled"]).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self._entry(ff, 1, "Preamble (hex)", self.v["frame_preamble_hex"])
        self._entry(ff, 2, "Syncword (hex)", self.v["frame_sync_hex"])
        self._entry(ff, 3, "Header format (name:bits,…)", self.v["frame_header_format"])
        self._combo(ff, 4, "CRC", self.v["frame_crc"], CRC_OPTIONS)
        self._combo(ff, 5, "FEC", self.v["frame_fec"], FEC_OPTIONS)
        self._entry(ff, 6, "Max sync dist (blank=auto)",
                     self.v["frame_max_sync_dist"])
        ttk.Button(ff, text="Reset to defaults",
                    command=self._reset_framing_defaults).grid(
            row=7, column=1, sticky="w", pady=(4, 0))

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
        ttk.Checkbutton(eb,
                         text="When framing is on, treat expected as PAYLOAD bits only",
                         variable=self.v["expected_is_payload"]).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))
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

    def _reset_framing_defaults(self):
        self.v["frame_preamble_hex"].set("AAAAAAAA")
        self.v["frame_sync_hex"].set("1ACFFC1D")
        self.v["frame_header_format"].set("length:16,seq:8,type:8")
        self.v["frame_crc"].set("crc16-ccitt-false")
        self.v["frame_fec"].set("hamming-7-4")
        self.v["frame_max_sync_dist"].set("")

    def _parse_header_format(self, s: str) -> tuple:
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(
                    f"Header field {part!r} missing ':bits' (e.g. 'length:16')")
            name, bits = part.split(":", 1)
            out.append((name.strip(), int(bits.strip())))
        if not out:
            raise ValueError("Header format must have at least one field.")
        return tuple(out)

    def _build_frame_config(self) -> FrameConfig:
        preamble = bytes.fromhex(
            self.v["frame_preamble_hex"].get().strip().replace(" ", ""))
        sync = bytes.fromhex(
            self.v["frame_sync_hex"].get().strip().replace(" ", ""))
        hdr_fmt = self._parse_header_format(self.v["frame_header_format"].get())
        return FrameConfig(
            preamble=preamble,
            syncword=sync,
            header_format=hdr_fmt,
            crc=self.v["frame_crc"].get() or "none",
            fec=self.v["frame_fec"].get() or "none",
        )

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

        framing_on = bool(self.v["frame_enabled"].get())
        expected = self._get_expected()

        try:
            if framing_on:
                fc = self._build_frame_config()
                msd_raw = self.v["frame_max_sync_dist"].get().strip()
                max_sync = int(msd_raw) if msd_raw else None
                exp_payload = expected if (
                    expected is not None
                    and self.v["expected_is_payload"].get()) else None
                recovered, report = demodulate_frame(
                    iq, params, fc,
                    expected_payload_bits=exp_payload,
                    max_sync_distance=max_sync)
                self._frame_report = report
                self._fc = fc
            else:
                recovered = demodulate(iq, params)
                self._frame_report = None
        except Exception as e:
            messagebox.showerror("Demodulation failed", str(e))
            return
        self._recovered = recovered

        # Build the result panel text
        lines: list[str] = []
        if framing_on:
            lines.extend(self._format_frame_diagnostics(self._frame_report,
                                                         expected))
            self._set_status_from_frame(self._frame_report)
        else:
            if expected is not None:
                rep = compare_bits(recovered, expected)
                lines.append(str(rep))
                self.v["_status"].set(
                    f"BER {rep.ber:.3e}" if rep.n_compared else "no comparison")
            else:
                self.v["_status"].set(f"Recovered {recovered.size} bits")
            lines.append("")
            lines.append(f"Recovered bits ({recovered.size}):")
            lines.append("".join(str(int(b)) for b in recovered))

        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "\n".join(lines))
        self._draw_constellation(iq, params, recovered)

    def _set_status_from_frame(self, rep):
        if not rep.sync_found:
            self.v["_status"].set(
                f"NO SYNC (best distance {rep.sync_distance}/"
                f"{rep.sync_pattern_bits})")
            return
        crc_part = ("CRC PASS" if rep.crc_ok
                    else ("CRC FAIL" if rep.crc_scheme != "none" else "no CRC"))
        ber_part = ""
        if rep.n_payload_compared:
            ber_part = (f", payload errors {rep.n_payload_errors}/"
                        f"{rep.n_payload_compared}")
        self.v["_status"].set(
            f"sync@{rep.sync_offset} (dist {rep.sync_distance}); "
            f"FEC corrections {rep.fec_corrections}; {crc_part}{ber_part}")

    def _format_frame_diagnostics(self, rep, expected) -> list[str]:
        """Format a FrameReport into the diagnostic text block."""
        L: list[str] = []
        L.append("=" * 60)
        L.append("FRAME DIAGNOSTICS")
        L.append("=" * 60)

        # SYNC
        L.append("[SYNC]")
        L.append(f"  preamble offset (bit) : {rep.preamble_offset}")
        L.append(f"  syncword offset (bit) : {rep.sync_offset}")
        L.append(f"  hamming distance      : {rep.sync_distance}"
                 f" / {rep.sync_pattern_bits}")
        L.append(f"  result                : "
                 f"{'FOUND' if rep.sync_found else 'NOT FOUND'}")
        if not rep.sync_found:
            if rep.notes:
                L.append("  notes:")
                for n in rep.notes:
                    L.append(f"    - {n}")
            return L

        # HEADER
        L.append("")
        L.append("[HEADER]")
        for k, v in rep.header_fields.items():
            L.append(f"  {k:<22s}: {v}  (0x{v:X})")
        if rep.header_bits is not None and rep.header_bits.size <= 128:
            L.append(f"  raw bits              : "
                     f"{''.join(str(int(b)) for b in rep.header_bits)}")

        # FEC
        L.append("")
        L.append(f"[FEC: {rep.fec_scheme}]")
        L.append(f"  codewords             : {rep.fec_codewords}")
        L.append(f"  corrections           : {rep.fec_corrections}")
        if rep.fec_corrected_positions:
            head = rep.fec_corrected_positions[:32]
            tail = "" if len(rep.fec_corrected_positions) <= 32 else \
                   f" … (+{len(rep.fec_corrected_positions) - 32} more)"
            L.append(f"  corrected positions   : "
                     f"{', '.join(str(i) for i in head)}{tail}")

        # CRC
        L.append("")
        L.append(f"[CRC: {rep.crc_scheme}]")
        if rep.crc_scheme == "none":
            L.append("  (no CRC configured)")
        else:
            L.append(f"  expected (from frame) : "
                     f"0x{rep.crc_expected:0{(rep.crc_expected.bit_length()+3)//4 or 1}X}"
                     if rep.crc_expected is not None else "  expected              : n/a")
            L.append(f"  computed              : "
                     f"0x{rep.crc_computed:0{(rep.crc_computed.bit_length()+3)//4 or 1}X}"
                     if rep.crc_computed is not None else "  computed              : n/a")
            L.append(f"  result                : "
                     f"{'PASS' if rep.crc_ok else 'FAIL'}")

        # PAYLOAD
        L.append("")
        L.append("[PAYLOAD]")
        n = 0 if rep.payload_bits is None else rep.payload_bits.size
        L.append(f"  declared length (bits): {rep.payload_length_declared}")
        L.append(f"  recovered length      : {n}")
        if rep.n_payload_compared:
            ber = (rep.n_payload_errors / rep.n_payload_compared
                   if rep.n_payload_compared else float("nan"))
            L.append(f"  compared              : {rep.n_payload_compared} bits")
            L.append(f"  bit errors            : {rep.n_payload_errors}")
            L.append(f"  BER                   : {ber:.3e}")
            # locate the first few error positions for diagnostics
            if rep.n_payload_errors and rep.payload_bits is not None:
                diff = (rep.payload_bits[:rep.n_payload_compared]
                        != expected[:rep.n_payload_compared].astype("uint8"))
                first = list(np.where(diff)[0][:16])
                L.append(f"  first error positions : "
                         f"{', '.join(str(i) for i in first)}")
        elif expected is not None and not self.v["expected_is_payload"].get():
            L.append("  (expected was compared against full demod bits; "
                     "see 'recovered bits' below)")

        if rep.payload_bits is not None:
            L.append("")
            L.append(f"  payload bits (first 256):")
            preview = "".join(str(int(b)) for b in rep.payload_bits[:256])
            suffix = "" if rep.payload_bits.size <= 256 else \
                     f" … (+{rep.payload_bits.size - 256} more)"
            L.append("  " + preview + suffix)

        if rep.notes:
            L.append("")
            L.append("[NOTES]")
            for n in rep.notes:
                L.append(f"  - {n}")
        return L

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
