"""Tkinter GUI for iqgen.

Launch:
    python -m iqgen.gui

Additive only — imports from the existing pipeline without modifying it.

Features:
  - All YAML options exposed via tabs.
  - Tooltips on every field describing constraints.
  - Source-type fields are gated: only the active type's inputs are shown.
  - Generate runs in a worker thread so the UI stays responsive; a Stop
    button discards the in-flight result (cannot actually interrupt the
    numerical work, but no files are written if cancelled in time).
  - Generate writes data file (.cf32 or SigMF pair) plus a matching .png
    rendered from the same Figure as the embedded view (one render, not two).
  - Save/Load presets round-trip the form to/from YAML.
  - Pipeline log mirrored into a bottom pane.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import SignalConfig
from .generator import IQGenerator
from .plotting import render
from .writers import Cf32Writer, SigMFWriter

log = logging.getLogger(__name__)


def _fmt_num(x) -> str:
    """Compact float repr for the GUI entry — drops trailing .0 on integers."""
    f = float(x)
    if f == int(f):
        return str(int(f))
    return repr(f)


MODULATIONS = ["bpsk", "dbpsk", "qpsk", "dqpsk", "pi4_qpsk",
               "oqpsk", "8psk", "d8psk", "pi4_8psk"]
FILTERS = ["none", "root_raised_cosine", "raised_cosine",
           "gaussian", "rectangular"]
NORMALIZATIONS = ["peak", "rms", "none"]
FORMATS = ["cf32", "sigmf"]
SOURCE_TYPES = ["random", "file", "bitstring"]
BIT_ORDERS = ["msb_first", "lsb_first"]
CHANNEL_MODES = ["concurrent", "hopping"]


# =============================================================================
# Tooltip helper
# =============================================================================

class Tooltip:
    """Tk has no native tooltip widget. This attaches one to any widget,
    appearing after a short hover delay and disappearing on leave/click.
    """

    def __init__(self, widget: tk.Misc, text: str, delay_ms: int = 450,
                 wraplength: int = 360):
        self.widget = widget
        self.text = text
        self.delay = delay_ms
        self.wraplength = wraplength
        self._tip: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _evt=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self._tip is not None:
            return
        # Place to the right of the widget, just below its top
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 8
        y = self.widget.winfo_rooty() + 4
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("TkDefaultFont", 9), wraplength=self.wraplength,
        ).pack(ipadx=6, ipady=3)

    def _hide(self, _evt=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None


def _attach_tooltip(text: str, *widgets: tk.Misc):
    for w in widgets:
        if w is not None:
            Tooltip(w, text)


# =============================================================================
# Logging handler that pipes records into a Tk Text widget
# =============================================================================

class _TextHandler(logging.Handler):
    def __init__(self, text_widget: tk.Text):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        try:
            self.text_widget.after(0, self._append, msg)
        except RuntimeError:
            pass  # widget destroyed during teardown

    def _append(self, msg):
        try:
            self.text_widget.configure(state="normal")
            self.text_widget.insert("end", msg + "\n")
            self.text_widget.see("end")
            self.text_widget.configure(state="disabled")
        except tk.TclError:
            pass  # widget already destroyed


# =============================================================================
# Tooltip text — central catalogue so wording stays consistent
# =============================================================================

T = {
    "name": (
        "Signal identifier. Appears in the output filename. Avoid spaces "
        "and path separators."
    ),
    "center_frequency_hz": (
        "Single value: nominal RF carrier frequency in Hz, stored in SigMF "
        "captures[].core:frequency. The baseband contains one carrier at "
        "offset 0. Scientific notation OK (e.g. 915e6).\n\n"
        "Multiple space-separated values: each value is a baseband carrier "
        "offset (in Hz); a signal is produced at every listed offset. SigMF "
        "core:frequency is set to 0 in this mode (you're specifying the "
        "baseband directly).\n"
        "Example: '1000 3000 5000' → three carriers at 1, 3, 5 kHz.\n"
        "Example: '-100e3 100e3'   → two carriers, ±100 kHz."
    ),
    "channel_mode": (
        "Only used when multiple frequencies are entered above.\n"
        "  concurrent: all offsets transmit simultaneously (FDM). The same "
        "baseband symbol stream is summed at each offset, then renormalized.\n"
        "  hopping:    one offset at a time, cycling round-robin through "
        "the list every 'Hop duration'. Phase resets at each boundary."
    ),
    "hop_duration_sec": (
        "Time in seconds per hop, used only when channel mode = hopping. "
        "Each hop occupies hop_duration · sample_rate samples; the last "
        "hop is truncated if the signal ends mid-hop."
    ),
    "sample_rate": (
        "Sample rate in Hz. May be auto-bumped upward so that "
        "samples_per_symbol = sample_rate / symbol_rate is an integer "
        "(and even, for OQPSK). The effective rate shown on the Rate tab "
        "after Generate is what actually gets written."
    ),
    "normalization": (
        "peak: max |sample| = 1.0  (default)\n"
        "rms:  sqrt(mean(|sample|^2)) = 1.0\n"
        "none: leave amplitudes as produced by filter normalization"
    ),
    "output_dir": (
        "Directory for output files (.cf32 / .sigmf-* / .png). "
        "Created automatically if it does not exist."
    ),
    "timestamp": (
        "If checked, the current UTC timestamp is prepended to the "
        "filename. Useful for archiving multiple runs."
    ),
    "timestamp_format": (
        "Python strftime format applied in UTC. Default %Y%m%d_%H%M%S → "
        "'20240115_120000'."
    ),

    "source_type": (
        "random:   generate uniform-random bits in-process\n"
        "file:     read packed bits from a binary file\n"
        "bitstring: literal '0'/'1' characters in this form\n"
        "Only the inputs for the selected type are used."
    ),
    "bit_count": (
        "Number of bits to generate. If not divisible by bits/symbol the "
        "stream is zero-padded (logged as a WARNING)."
    ),
    "duration_sec": (
        "Alternative to bit_count: generates int(round(duration·bitrate)) "
        "bits. Overrides bit_count when both are provided. Leave blank to "
        "use bit_count."
    ),
    "seed": (
        "Optional integer for reproducible random bits. Leave blank for "
        "nondeterministic output (seeded from OS entropy)."
    ),
    "input_file": (
        "Binary file containing packed bits (8 bits per byte). Read in "
        "full. Use bit_order to control how each byte is unpacked."
    ),
    "bit_order": (
        "msb_first: byte 0xA5 → 1,0,1,0,0,1,0,1\n"
        "lsb_first: byte 0xA5 → 1,0,1,0,0,1,0,1 reversed"
    ),
    "bits": (
        "Literal '0' and '1' characters. Whitespace and any other "
        "characters are silently stripped. An empty result is rejected."
    ),

    "modulation": (
        "Bits/symbol — BPSK/DBPSK: 1, QPSK/DQPSK/π4-QPSK/OQPSK: 2, "
        "8PSK/D8PSK/π4-8PSK: 3.\n"
        "Differential variants encode each symbol as a rotation from the "
        "previous output (no absolute phase reference required).\n"
        "OQPSK delays the Q channel by half a symbol; requires even sps."
    ),
    "gray_coding": (
        "If enabled, adjacent constellation points differ by exactly one "
        "bit → lower BER at high SNR. Has no effect for BPSK (trivially "
        "Gray). For π/4-DQPSK the Gray mapping is baked into the delta "
        "table and this flag is informational."
    ),
    "initial_phase": (
        "Reference phase (radians) for differential and π/4-DQPSK "
        "modulations. No effect on non-differential BPSK/QPSK/8PSK."
    ),

    "bitrate": (
        "REQUIRED. Information bitrate in bits/sec. Source of truth — "
        "symbol_rate = bitrate / bits_per_symbol and "
        "samples_per_symbol = sample_rate / symbol_rate are derived. "
        "Setting symbol_rate or sps in YAML logs a WARNING and is "
        "overridden."
    ),

    "filter_type": (
        "Pulse-shaping filter applied after upsampling.\n"
        "  none: sample-and-hold NRZ — each symbol becomes a rectangular "
        "pulse of one symbol period. Spectrum is sinc(f/Rs).\n"
        "  rectangular: same shape as 'none' but routed through the "
        "convolution path (use when you want span_symbols transient "
        "behavior to match the other filters).\n"
        "  root_raised_cosine: Tx-matched filter (uses roll_off)\n"
        "  raised_cosine: full Nyquist response (uses roll_off)\n"
        "  gaussian: GMSK/FSK-style (uses bt_product)"
    ),
    "span_symbols": (
        "Filter span in symbol periods. Larger → sharper spectrum and "
        "longer transients. num_taps = span · sps + 1. Default 10."
    ),
    "roll_off": (
        "RRC / RC excess bandwidth β. Range (0, 1]. Typical 0.2–0.5. "
        "Smaller = narrower spectrum but longer impulse response. "
        "Default 0.35. Ignored for other filter types."
    ),
    "bt_product": (
        "Gaussian bandwidth-time product. Smaller = narrower spectrum + "
        "more ISI. GSM uses 0.3. Default 0.35. Ignored for other filter "
        "types."
    ),

    "format": (
        "cf32:  single .cf32 file (raw interleaved IQ as complex64 LE)\n"
        "sigmf: .sigmf-data (cf32_le) + .sigmf-meta (JSON, SigMF 1.0.0)"
    ),
    "sigmf_author": "SigMF core:author. Free text.",
    "sigmf_description": "SigMF core:description. Free text.",
    "sigmf_license": "SigMF core:license. Free text (e.g. CC0, MIT).",
    "sigmf_hardware": (
        "SigMF core:hardware. Free text — for synthetic signals use "
        "something like 'iqgen synthetic'."
    ),
    "sigmf_recorder": "SigMF core:recorder. Free text (defaults to 'iqgen').",
    "sigmf_comment": (
        "Placed in the annotation's core:description field. Free text — "
        "use to record what's special about this run."
    ),

    "symbol_rate": "Derived = bitrate / bits_per_symbol.",
    "sps": (
        "Derived = sample_rate / symbol_rate (rounded). Must be ≥1 and "
        "even for OQPSK."
    ),
    "effective_sample_rate": (
        "What was actually written to disk after any auto-bump."
    ),
    "num_taps": "Filter length = span_symbols · sps + 1.",
}


# =============================================================================
# Main GUI
# =============================================================================

class IQGenGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("iqgen — IQ Signal Generator")
        self.root.geometry("1400x900")

        self.v = self._make_vars()
        self._gen_id = 0
        self._busy = False
        # Worker thread results land here; the main thread polls.
        # Tkinter is NOT thread-safe — workers never touch widgets.
        self._result_queue: queue.Queue = queue.Queue()

        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(main, width=500)
        right = ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)

        self._build_form(left)
        self._build_bottom(left)
        self._build_plot_area(right)

        self._install_log_handler()
        self._poll_result_queue()
        log.info("Ready. Set parameters and click Generate.")

    # ---------------- variable definitions ----------------
    def _make_vars(self) -> dict[str, tk.Variable]:
        return {
            # signal
            "name": tk.StringVar(value="test"),
            "center_frequency_hz": tk.StringVar(value="915e6"),
            "sample_rate": tk.StringVar(value="8e3"),
            "normalization": tk.StringVar(value="peak"),
            "output_dir": tk.StringVar(value="./output"),
            "timestamp": tk.BooleanVar(value=True),
            "timestamp_format": tk.StringVar(value="%Y%m%d_%H%M%S"),
            # multi-frequency
            "channel_mode": tk.StringVar(value="concurrent"),
            "hop_duration_sec": tk.StringVar(value=""),
            # source
            "source_type": tk.StringVar(value="random"),
            "bit_count": tk.StringVar(value="10000"),
            "duration_sec": tk.StringVar(value=""),
            "seed": tk.StringVar(value=""),
            "input_file": tk.StringVar(value=""),
            "bit_order": tk.StringVar(value="msb_first"),
            # modulation
            "modulation": tk.StringVar(value="bpsk"),
            "gray_coding": tk.BooleanVar(value=True),
            "initial_phase": tk.StringVar(value="0.0"),
            # rate
            "bitrate": tk.StringVar(value="1000"),
            # pulse shaping
            "filter_type": tk.StringVar(value="none"),
            "span_symbols": tk.StringVar(value="10"),
            "roll_off": tk.StringVar(value="0.35"),
            "bt_product": tk.StringVar(value="0.35"),
            # output
            "format": tk.StringVar(value="sigmf"),
            "sigmf_author": tk.StringVar(value=""),
            "sigmf_description": tk.StringVar(value=""),
            "sigmf_license": tk.StringVar(value=""),
            "sigmf_hardware": tk.StringVar(value="iqgen synthetic"),
            "sigmf_recorder": tk.StringVar(value="iqgen"),
            "sigmf_comment": tk.StringVar(value=""),
            # derived (read-only display)
            "_symbol_rate": tk.StringVar(value="—"),
            "_sps": tk.StringVar(value="—"),
            "_effective_sample_rate": tk.StringVar(value="—"),
            "_num_taps": tk.StringVar(value="—"),
        }

    # ---------------- form ----------------
    def _build_form(self, parent: ttk.Frame):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)
        self._tab_signal(nb)
        self._tab_source(nb)
        self._tab_modulation(nb)
        self._tab_rate(nb)
        self._tab_filter(nb)
        self._tab_output(nb)

    def _tab_signal(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Signal")
        self._row(f, 0, "Name", self.v["name"], tip=T["name"])
        self._row(f, 1, "Center freq (Hz)", self.v["center_frequency_hz"],
                  tip=T["center_frequency_hz"])
        ttk.Label(
            f,
            text=("Tip: enter 2+ space-separated values to get a carrier "
                  "at each one, e.g. '1000 3000 5000'."),
            foreground="#666", font=("TkDefaultFont", 8, "italic"),
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=4)
        self._row(f, 3, "Sample rate (Hz)", self.v["sample_rate"],
                  tip=T["sample_rate"])
        self._combo_row(f, 4, "Normalization", self.v["normalization"],
                         NORMALIZATIONS, tip=T["normalization"])
        self._dir_row(f, 5, "Output dir", self.v["output_dir"],
                       tip=T["output_dir"])
        cb = ttk.Checkbutton(f, text="Prepend timestamp to filename",
                              variable=self.v["timestamp"])
        cb.grid(row=6, column=1, sticky="w", pady=2)
        _attach_tooltip(T["timestamp"], cb)
        self._row(f, 7, "Timestamp format", self.v["timestamp_format"],
                  tip=T["timestamp_format"])

        # Multi-frequency subsection — only meaningful when center freq has
        # 2+ space-delimited values, but the controls are always visible so
        # the user sees the option exists.
        mf = ttk.LabelFrame(f, text="Multi-frequency (used when 2+ freqs above)",
                             padding=6)
        mf.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        f.grid_columnconfigure(1, weight=1)
        self._combo_row(mf, 0, "Channel mode", self.v["channel_mode"],
                         CHANNEL_MODES, tip=T["channel_mode"])
        self._row(mf, 1, "Hop duration (s)", self.v["hop_duration_sec"],
                  tip=T["hop_duration_sec"])

    def _tab_source(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Source")
        self._combo_row(f, 0, "Source type", self.v["source_type"],
                         SOURCE_TYPES, tip=T["source_type"])

        # One LabelFrame per source type, only the active one is gridded
        self.source_frames: dict[str, ttk.LabelFrame] = {}

        rf = ttk.LabelFrame(f, text="random", padding=8)
        self.source_frames["random"] = rf
        self._row(rf, 0, "Bit count", self.v["bit_count"], tip=T["bit_count"])
        self._row(rf, 1, "Duration sec (overrides bit_count)",
                  self.v["duration_sec"], tip=T["duration_sec"])
        self._row(rf, 2, "Seed (optional)", self.v["seed"], tip=T["seed"])

        ff = ttk.LabelFrame(f, text="file", padding=8)
        self.source_frames["file"] = ff
        self._file_row(ff, 0, "Input file", self.v["input_file"],
                        tip=T["input_file"])
        self._combo_row(ff, 1, "Bit order", self.v["bit_order"], BIT_ORDERS,
                         tip=T["bit_order"])

        bf = ttk.LabelFrame(f, text="bitstring", padding=8)
        self.source_frames["bitstring"] = bf
        lbl = ttk.Label(bf, text="Bits")
        lbl.grid(row=0, column=0, sticky="nw", padx=4, pady=2)
        self.bits_text = tk.Text(bf, height=10, wrap="word",
                                   font=("TkFixedFont", 9))
        self.bits_text.grid(row=0, column=1, sticky="nsew", padx=4, pady=2)
        bf.grid_columnconfigure(1, weight=1)
        bf.grid_rowconfigure(0, weight=1)
        _attach_tooltip(T["bits"], lbl, self.bits_text)

        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)

        self.v["source_type"].trace_add(
            "write", lambda *_: self._update_source_visibility())
        self._update_source_visibility()

    def _update_source_visibility(self):
        active = self.v["source_type"].get()
        for name, frame in self.source_frames.items():
            if name == active:
                frame.grid(row=1, column=0, columnspan=3,
                            sticky="nsew", pady=4)
            else:
                frame.grid_remove()

    def _tab_modulation(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Modulation")
        self._combo_row(f, 0, "Type", self.v["modulation"], MODULATIONS,
                         tip=T["modulation"])
        cb = ttk.Checkbutton(f, text="Gray coding",
                              variable=self.v["gray_coding"])
        cb.grid(row=1, column=1, sticky="w", pady=2)
        _attach_tooltip(T["gray_coding"], cb)
        self._row(f, 2, "Initial phase (rad)", self.v["initial_phase"],
                  tip=T["initial_phase"])

    def _tab_rate(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Rate")
        ttk.Label(
            f,
            text=("bitrate is the source of truth.\n"
                  "symbol_rate and samples_per_symbol are derived.\n"
                  "sample_rate may be auto-bumped to keep sps integer "
                  "(even for OQPSK)."),
            foreground="#666",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self._row(f, 1, "Bitrate (bps)", self.v["bitrate"], tip=T["bitrate"])
        ttk.Separator(f).grid(row=2, column=0, columnspan=3,
                               sticky="ew", pady=6)
        ttk.Label(f, text="Derived (updated after Generate)",
                  font=("TkDefaultFont", 9, "bold")).grid(
            row=3, column=0, columnspan=3, sticky="w")
        self._ro_row(f, 4, "Symbol rate (sym/s)", self.v["_symbol_rate"],
                      tip=T["symbol_rate"])
        self._ro_row(f, 5, "Samples per symbol", self.v["_sps"],
                      tip=T["sps"])
        self._ro_row(f, 6, "Effective sample rate (Hz)",
                      self.v["_effective_sample_rate"],
                      tip=T["effective_sample_rate"])
        self._ro_row(f, 7, "Filter taps", self.v["_num_taps"],
                      tip=T["num_taps"])

    def _tab_filter(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Pulse Shaping")
        self._combo_row(f, 0, "Filter type", self.v["filter_type"], FILTERS,
                         tip=T["filter_type"])
        self._row(f, 1, "Span symbols", self.v["span_symbols"],
                  tip=T["span_symbols"])
        self._row(f, 2, "Roll-off (RRC / RC)", self.v["roll_off"],
                  tip=T["roll_off"])
        self._row(f, 3, "BT product (Gaussian)", self.v["bt_product"],
                  tip=T["bt_product"])

    def _tab_output(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Output")
        self._combo_row(f, 0, "Format", self.v["format"], FORMATS,
                         tip=T["format"])
        ttk.Separator(f).grid(row=1, column=0, columnspan=3,
                               sticky="ew", pady=6)
        ttk.Label(f, text="SigMF metadata (only used when format=sigmf)",
                  font=("TkDefaultFont", 9, "bold")).grid(
            row=2, column=0, columnspan=3, sticky="w")
        self._row(f, 3, "Author", self.v["sigmf_author"],
                   tip=T["sigmf_author"])
        self._row(f, 4, "Description", self.v["sigmf_description"],
                   tip=T["sigmf_description"])
        self._row(f, 5, "License", self.v["sigmf_license"],
                   tip=T["sigmf_license"])
        self._row(f, 6, "Hardware", self.v["sigmf_hardware"],
                   tip=T["sigmf_hardware"])
        self._row(f, 7, "Recorder", self.v["sigmf_recorder"],
                   tip=T["sigmf_recorder"])
        self._row(f, 8, "Comment (annotation)", self.v["sigmf_comment"],
                   tip=T["sigmf_comment"])

    # ---------------- row helpers (tip = optional tooltip text) ----------------
    def _row(self, parent, r, label, var, tip: str = ""):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ent = ttk.Entry(parent, textvariable=var)
        ent.grid(row=r, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        parent.grid_columnconfigure(1, weight=1)
        if tip:
            _attach_tooltip(tip, lbl, ent)

    def _ro_row(self, parent, r, label, var, tip: str = ""):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ent = ttk.Entry(parent, textvariable=var, state="readonly")
        ent.grid(row=r, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        parent.grid_columnconfigure(1, weight=1)
        if tip:
            _attach_tooltip(tip, lbl, ent)

    def _combo_row(self, parent, r, label, var, values, tip: str = ""):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=r, column=0, sticky="w", padx=4, pady=2)
        cb = ttk.Combobox(parent, textvariable=var, values=values,
                            state="readonly")
        cb.grid(row=r, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        parent.grid_columnconfigure(1, weight=1)
        if tip:
            _attach_tooltip(tip, lbl, cb)

    def _dir_row(self, parent, r, label, var, tip: str = ""):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ent = ttk.Entry(parent, textvariable=var)
        ent.grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        btn = ttk.Button(parent, text="Browse…",
                          command=lambda: self._browse_dir(var))
        btn.grid(row=r, column=2, padx=2)
        parent.grid_columnconfigure(1, weight=1)
        if tip:
            _attach_tooltip(tip, lbl, ent)

    def _file_row(self, parent, r, label, var, tip: str = ""):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ent = ttk.Entry(parent, textvariable=var)
        ent.grid(row=r, column=1, sticky="ew", padx=4, pady=2)
        btn = ttk.Button(parent, text="Browse…",
                          command=lambda: self._browse_file(var))
        btn.grid(row=r, column=2, padx=2)
        parent.grid_columnconfigure(1, weight=1)
        if tip:
            _attach_tooltip(tip, lbl, ent)

    def _browse_dir(self, var: tk.StringVar):
        initial = var.get() or "."
        if not Path(initial).is_dir():
            initial = "."
        path = filedialog.askdirectory(initialdir=initial,
                                         title="Choose output directory")
        if path:
            var.set(path)

    def _browse_file(self, var: tk.StringVar):
        initial = Path(var.get()).parent if var.get() else Path(".")
        if not initial.is_dir():
            initial = Path(".")
        path = filedialog.askopenfilename(initialdir=str(initial),
                                            title="Choose input bits file")
        if path:
            var.set(path)

    # ---------------- plot area ----------------
    def _build_plot_area(self, parent: ttk.Frame):
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.text(0.5, 0.5, "Click Generate to produce a signal",
                       ha="center", va="center", fontsize=12, color="#888")
        self.canvas.draw()

    # ---------------- bottom (buttons + log) ----------------
    def _build_bottom(self, parent: ttk.Frame):
        btn = ttk.Frame(parent)
        btn.pack(fill=tk.X, padx=4, pady=(8, 4))
        self.btn_generate = ttk.Button(btn, text="Generate",
                                         command=self.generate)
        self.btn_generate.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(btn, text="Stop", command=self.stop,
                                     state="disabled")
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        _attach_tooltip(
            "Discards the result of an in-flight Generate (no files are "
            "written). Cannot interrupt numpy/scipy work already running, "
            "but the next interactive update will be responsive again.",
            self.btn_stop,
        )

        ttk.Separator(btn, orient="vertical").pack(side=tk.LEFT, fill="y",
                                                      padx=8)
        ttk.Button(btn, text="Save preset…",
                    command=self.save_preset).pack(side=tk.LEFT)
        ttk.Button(btn, text="Load preset…",
                    command=self.load_preset).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn, text="Clear log",
                    command=self.clear_log).pack(side=tk.RIGHT)

        ttk.Label(parent, text="Log").pack(anchor="w", padx=4)
        self.log_text = tk.Text(parent, height=10, state="disabled",
                                  wrap="word", font=("TkFixedFont", 9))
        self.log_text.pack(fill=tk.BOTH, expand=False, padx=4, pady=(0, 4))

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _install_log_handler(self):
        handler = _TextHandler(self.log_text)
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger = logging.getLogger("iqgen")
        logger.setLevel(logging.INFO)
        if not any(isinstance(h, _TextHandler) for h in logger.handlers):
            logger.addHandler(handler)

    # ---------------- config build / restore ----------------
    @staticmethod
    def _fnum(s: str, default=None):
        s = (s or "").strip()
        if not s:
            return default
        return float(s)

    @staticmethod
    def _fint(s: str, default=None):
        s = (s or "").strip()
        if not s:
            return default
        return int(float(s))

    def _build_config_dict(self) -> dict[str, Any]:
        v = self.v
        # Center frequency field:
        #   1 value  → single-carrier RF mode. That value is the SigMF
        #              center frequency; the baseband contains one carrier
        #              at offset 0.
        #   2+ values → multi-carrier mode. EVERY value is a baseband
        #              carrier offset. SigMF center frequency defaults to 0
        #              because the user is specifying baseband directly.
        cf_tokens = (v["center_frequency_hz"].get() or "").split()
        try:
            cf_values = [float(t) for t in cf_tokens]
        except ValueError as e:
            raise ValueError(
                f"Could not parse center frequency field "
                f"'{v['center_frequency_hz'].get()}': {e}"
            ) from None

        if len(cf_values) <= 1:
            center_freq = cf_values[0] if cf_values else 0.0
            extra_offsets: list[float] = []
        else:
            center_freq = 0.0
            extra_offsets = cf_values

        signal = {
            "name": v["name"].get().strip() or "signal",
            "center_frequency_hz": center_freq,
            "sample_rate": self._fnum(v["sample_rate"].get()),
            "normalization": v["normalization"].get(),
            "output_dir": v["output_dir"].get().strip() or "./output",
            "timestamp": bool(v["timestamp"].get()),
            "timestamp_format": (v["timestamp_format"].get()
                                 or "%Y%m%d_%H%M%S"),
        }

        src_type = v["source_type"].get()
        source: dict[str, Any] = {"type": src_type}
        if src_type == "random":
            n = self._fint(v["bit_count"].get())
            if n is not None:
                source["bit_count"] = n
            dur = self._fnum(v["duration_sec"].get())
            if dur is not None:
                source["duration_sec"] = dur
            seed = self._fint(v["seed"].get())
            if seed is not None:
                source["seed"] = seed
        elif src_type == "file":
            path = v["input_file"].get().strip()
            if not path:
                raise ValueError(
                    "file source: choose an input file (Source tab)")
            source["input_file"] = path
            source["bit_order"] = v["bit_order"].get()
        elif src_type == "bitstring":
            raw = self.bits_text.get("1.0", "end")
            cleaned = [c for c in raw if c in "01"]
            if not cleaned:
                raise ValueError(
                    "bitstring source: enter at least one '0' or '1' "
                    "character (Source tab)")
            source["bits"] = raw

        modulation = {
            "type": v["modulation"].get(),
            "gray_coding": bool(v["gray_coding"].get()),
            "initial_phase": self._fnum(v["initial_phase"].get(), 0.0),
        }
        rate = {"bitrate": self._fnum(v["bitrate"].get())}
        pulse_shaping = {
            "filter_type": v["filter_type"].get(),
            "span_symbols": self._fint(v["span_symbols"].get(), 10),
            "roll_off": self._fnum(v["roll_off"].get(), 0.35),
            "bt_product": self._fnum(v["bt_product"].get(), 0.35),
        }

        sigmf: dict[str, Any] = {}
        for short, key in [
            ("author", "sigmf_author"),
            ("description", "sigmf_description"),
            ("license", "sigmf_license"),
            ("hardware", "sigmf_hardware"),
            ("recorder", "sigmf_recorder"),
            ("comment", "sigmf_comment"),
        ]:
            val = v[key].get().strip()
            if val:
                sigmf[short] = val
        output = {"format": v["format"].get()}
        if sigmf:
            output["sigmf"] = sigmf

        cfg_dict: dict[str, Any] = {
            "signal": signal,
            "source": source,
            "modulation": modulation,
            "rate": rate,
            "pulse_shaping": pulse_shaping,
            "output": output,
        }

        # Only emit a `channels:` block when extra offsets were given —
        # otherwise stay backwards-compatible with single-carrier presets.
        if extra_offsets:
            mode = v["channel_mode"].get() or "concurrent"
            channels: dict[str, Any] = {
                "mode": mode,
                "offsets_hz": extra_offsets,
            }
            if mode == "hopping":
                hd = self._fnum(v["hop_duration_sec"].get())
                if hd is None:
                    raise ValueError(
                        "Hop duration is required when channel mode = hopping "
                        "(Signal tab)"
                    )
                channels["hop_duration_sec"] = hd
            cfg_dict["channels"] = channels

        return cfg_dict

    def _populate_from_dict(self, raw: dict):
        v = self.v
        s = raw.get("signal") or {}
        v["name"].set(str(s.get("name", "")))

        # Rebuild the space-delimited center-freq string. With a `channels:`
        # block present, every offset is a carrier — show them as the
        # space-delimited list (no separate center-freq slot in that mode).
        ch = raw.get("channels") or {}
        offsets = list(ch.get("offsets_hz") or [])
        if offsets:
            cf_str = " ".join(_fmt_num(o) for o in offsets)
        else:
            cf_str = str(s.get("center_frequency_hz", ""))
        v["center_frequency_hz"].set(cf_str)
        v["channel_mode"].set(str(ch.get("mode", "concurrent")))
        hd = ch.get("hop_duration_sec")
        v["hop_duration_sec"].set("" if hd is None else _fmt_num(hd))
        v["sample_rate"].set(str(s.get("sample_rate", "")))
        v["normalization"].set(str(s.get("normalization", "peak")))
        v["output_dir"].set(str(s.get("output_dir", "./output")))
        v["timestamp"].set(bool(s.get("timestamp", True)))
        v["timestamp_format"].set(str(s.get("timestamp_format",
                                              "%Y%m%d_%H%M%S")))

        src = raw.get("source") or {}
        v["source_type"].set(str(src.get("type", "random")))
        v["bit_count"].set(str(src["bit_count"]) if "bit_count" in src else "")
        v["duration_sec"].set(str(src["duration_sec"])
                                if "duration_sec" in src else "")
        v["seed"].set(str(src["seed"]) if "seed" in src else "")
        v["input_file"].set(str(src.get("input_file", "")))
        v["bit_order"].set(str(src.get("bit_order", "msb_first")))
        self.bits_text.delete("1.0", "end")
        if "bits" in src and src["bits"] is not None:
            self.bits_text.insert("1.0", str(src["bits"]))

        m = raw.get("modulation") or {}
        v["modulation"].set(str(m.get("type", "bpsk")))
        v["gray_coding"].set(bool(m.get("gray_coding", True)))
        v["initial_phase"].set(str(m.get("initial_phase", 0.0)))

        r = raw.get("rate") or {}
        v["bitrate"].set(str(r.get("bitrate", "")))

        ps = raw.get("pulse_shaping") or {}
        v["filter_type"].set(str(ps.get("filter_type", "none")))
        v["span_symbols"].set(str(ps.get("span_symbols", 10)))
        v["roll_off"].set(str(ps.get("roll_off", 0.35)))
        v["bt_product"].set(str(ps.get("bt_product", 0.35)))

        out = raw.get("output") or {}
        v["format"].set(str(out.get("format", "cf32")))
        sigmf = out.get("sigmf") or {}
        v["sigmf_author"].set(str(sigmf.get("author", "")))
        v["sigmf_description"].set(str(sigmf.get("description", "")))
        v["sigmf_license"].set(str(sigmf.get("license", "")))
        v["sigmf_hardware"].set(str(sigmf.get("hardware",
                                                "iqgen synthetic")))
        v["sigmf_recorder"].set(str(sigmf.get("recorder", "iqgen")))
        v["sigmf_comment"].set(str(sigmf.get("comment", "")))

    # ---------------- actions ----------------
    def save_preset(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*")],
            title="Save preset as YAML",
        )
        if not path:
            return
        try:
            raw = self._build_config_dict()
            with open(path, "w") as fp:
                yaml.safe_dump(raw, fp, sort_keys=False)
            log.info("Saved preset: %s", path)
        except Exception as e:  # noqa: BLE001
            log.exception("Save preset failed")
            messagebox.showerror("Save failed", str(e))

    def load_preset(self):
        path = filedialog.askopenfilename(
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*")],
            title="Load preset from YAML",
        )
        if not path:
            return
        try:
            with open(path) as fp:
                raw = yaml.safe_load(fp) or {}
            if not isinstance(raw, dict):
                raise ValueError("Preset YAML must be a mapping at the root")
            self._populate_from_dict(raw)
            log.info("Loaded preset: %s", path)
        except Exception as e:  # noqa: BLE001
            log.exception("Load preset failed")
            messagebox.showerror("Load failed", str(e))

    # ---- threaded generate ----
    def generate(self):
        if self._busy:
            return
        try:
            raw = self._build_config_dict()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Bad configuration", str(e))
            return

        self._gen_id += 1
        my_id = self._gen_id
        self._set_busy(True)
        log.info("Generating (run %d)...", my_id)

        # Worker pushes a tuple onto the queue; the main thread picks it up
        # via the recurring _poll_result_queue. Never touch widgets here.
        def worker():
            try:
                cfg = SignalConfig.from_dict(raw)
                signal = IQGenerator(cfg).generate()
                self._result_queue.put((my_id, cfg, signal, None))
            except Exception as e:  # noqa: BLE001
                log.exception("Generation failed in worker")
                self._result_queue.put((my_id, None, None, e))

        threading.Thread(target=worker, name="iqgen-worker",
                          daemon=True).start()

    def _poll_result_queue(self):
        """Main-thread tick: drain any results pushed by worker threads."""
        try:
            while True:
                item = self._result_queue.get_nowait()
                self._on_generate_done(*item)
        except queue.Empty:
            pass
        # Keep polling. 50 ms is fast enough to feel instant, light on CPU.
        try:
            self.root.after(50, self._poll_result_queue)
        except tk.TclError:
            pass  # window destroyed

    def _on_generate_done(self, gen_id, cfg, signal, error):
        # Discard if user clicked Stop or kicked off a newer run
        if gen_id != self._gen_id:
            log.info("Discarding stale result from run %d (current %d)",
                     gen_id, self._gen_id)
            return
        self._set_busy(False)

        if error is not None:
            messagebox.showerror("Generation failed", str(error))
            return

        # Update derived display
        self.v["_symbol_rate"].set(f"{cfg.symbol_rate:g}")
        self.v["_sps"].set(str(cfg.samples_per_symbol))
        self.v["_effective_sample_rate"].set(f"{cfg.sample_rate:g}")
        self.v["_num_taps"].set(str(cfg.num_taps))

        # Write data file (fast — kB or low MB)
        try:
            if cfg.format == "sigmf":
                data_path, _ = SigMFWriter().write(signal, cfg)
            else:
                data_path = Cf32Writer().write(signal, cfg)
        except Exception as e:  # noqa: BLE001
            log.exception("Writing data file failed")
            messagebox.showerror("Write failed", str(e))
            return

        # Render plots once, then save the same figure as PNG
        title = (f"{cfg.name}  {cfg.modulation}  {cfg.bitrate:g} bps  "
                 f"sps={cfg.samples_per_symbol}  filter={cfg.filter_type}")
        render(self.fig, signal, cfg.sample_rate,
                cfg.center_frequency_hz,
                title=f"{title}  ({signal.size} samples)")
        self.canvas.draw()
        png_path = data_path.with_suffix(".png")
        try:
            self.fig.savefig(png_path, dpi=120)
            log.info("Wrote plot PNG: %s", png_path)
        except Exception:  # noqa: BLE001
            log.exception("PNG save failed (data file still written)")

    def stop(self):
        if not self._busy:
            return
        # Invalidate the in-flight run; result will be discarded on arrival.
        self._gen_id += 1
        self._set_busy(False)
        log.warning("Stop requested — discarding in-flight result. "
                     "(Numpy/scipy work cannot be interrupted; UI is now free.)")

    def _set_busy(self, busy: bool):
        self._busy = busy
        if busy:
            self.btn_generate.configure(state="disabled",
                                          text="Generating…")
            self.btn_stop.configure(state="normal")
            try:
                self.root.configure(cursor="watch")
            except tk.TclError:
                pass
        else:
            self.btn_generate.configure(state="normal", text="Generate")
            self.btn_stop.configure(state="disabled")
            try:
                self.root.configure(cursor="")
            except tk.TclError:
                pass


def main():
    root = tk.Tk()
    IQGenGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
