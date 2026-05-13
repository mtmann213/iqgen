# LLM Handover — iqgen

This document is intended for an LLM picking up the project. It captures the
build history, design decisions, gotchas, and current state in enough detail
that you can continue without re-deriving context.

## Current status

**Working state.** All 100 smoke tests pass
(`python3 tests/smoke_test.py`). CLI and GUI both function end-to-end.

Last completed feature: multi-frequency transmission (concurrent FDM and
frequency hopping). Verified that FFT peaks land at the configured offsets.

No known bugs. No pending features the user has asked for.

## Build history (phases)

The project was built incrementally. Each phase ended with the user verifying
it worked before moving on.

1. **Core (CLI)** — 9 modulations, 5 filters, 3 sources, cf32/SigMF output,
   YAML config, interconnected rate parameters with `bitrate` as source of
   truth.
2. **GUI** — Tkinter form, live 4-panel plot, PNG export per run, save/load
   presets.
3. **GUI polish** — threading (queue.Queue + `root.after` polling pattern),
   source-type field gating, empty-input validation, tooltips on every
   field.
4. **NRZ bug fix** — `filter_type: none` was producing an impulse-train
   spectrum instead of pulse-shaped. Fixed by splitting the upsampler into
   `_zero_upsample` (zero-stuff, used before filter convolution) and
   `_hold_upsample` (sample-and-hold NRZ, used when no filter). After the
   fix: ~90% of power within ±symbol_rate (was ~25% before); first -20 dB
   rolloff near the symbol rate as expected.
5. **Multi-frequency** — optional `channels:` YAML block with
   `concurrent` and `hopping` modes. GUI uses space-delimited values in the
   existing Center freq field (2+ values → every value is a baseband carrier
   offset, SigMF center = 0).
6. **π/2-BPSK** — added the 5G NR low-PAPR modulation (1 bit/symbol; even
   symbols on the real axis, odd on the imaginary).
7. **Verifier** (most recent) — noise-free matched receiver that recovers
   bits from a .cf32 or .sigmf-data file. CLI and Tkinter GUI. Supports
   every mod/filter combo + multi-frequency. Lives in `verifier.py`,
   `verify_cli.py`, `verifier_gui.py`. 50 round-trip cases (generate →
   verify) are part of the smoke suite.

## Architecture

```
bits ─► symbols ─► upsample ─► pulse-shape ─► normalize ─► multi-freq mix ─► write
```

Driver: `iqgen/generator.py::IQGenerator.generate()`. Each pipeline stage
lives in its own module so they can be swapped/extended:

- `sources.py` — `DataSource.from_config()` factory → `RandomSource`,
  `FileSource`, `BitstringSource`.
- `mappers.py` — `create_mapper()` factory → `BPSKMapper`, `QPSKMapper`,
  `PSK8Mapper`, `DifferentialMapper` wrapper, `Pi4QPSKMapper`,
  `Pi4PSK8Mapper`, `OQPSKMapper`. Uses MSB-first `bits_to_indices`
  packing and `binary_to_gray` (i ^ i>>1).
- `filters.py` — `PulseShaper` class with manual RRC implementation
  handling `t=0` and `t=±T/(4β)` singularities. RRC/RC are
  unit-energy normalized; Gaussian/rectangular are unit-DC-gain.
- `generator.py` — pipeline orchestrator. Contains both `_zero_upsample`
  and `_hold_upsample` (critical distinction — see "Gotchas" below) and
  `_apply_channels` for multi-freq mixing.
- `writers.py` — `Cf32Writer` and `SigMFWriter`. SigMFWriter emits one
  annotation per carrier (concurrent) or per hop (hopping).
- `plotting.py` — backend-agnostic `render(fig, ...)` used both by the GUI
  (embedded canvas) and by `save_png()` (headless Agg).
- `config.py` — `SignalConfig` dataclass; `from_yaml()` and `from_dict()`
  both call `_validate_and_derive()`. This is where bandwidth/Nyquist
  validation happens for multi-freq, and where `sample_rate` is
  auto-bumped to keep `sps` integer (and even for OQPSK).
- `gui.py` — additive only. Imports the pipeline; never modifies it.
- `verifier.py` — inverse pipeline. `demodulate(iq, ReceiveParams)`
  returns bits. Key insight: `PulseShaper.apply()` uses `mode="same"`, so
  post-convolution symbol *n* sits at sample `n*sps` (no group-delay
  offset to chase). Matched filter is applied only for RRC (turns it into
  RC — zero-ISI). For other filters direct sampling at `n*sps` works
  because TX already produces zero (or near-zero) ISI at symbol centers.
  OQPSK is handled specially: I and Q peaks are offset by `sps/2`.
- `verify_cli.py` — `python -m iqgen.verify_cli FILE [--bits ...] [--modulation ...]`.
  SigMF inputs auto-populate params from `iqgen:*` annotation keys; cf32
  inputs require the params via flags.
- `verifier_gui.py` — Tkinter form mirroring the generator GUI's fields.
  Auto-fills from a chosen .sigmf-meta. Constellation plot of the
  recovered (post-matched-filter) symbols after Verify.

## Important design decisions

- **`bitrate` is the source of truth.** Setting `symbol_rate` or
  `samples_per_symbol` in YAML logs a WARNING and is overridden by the
  derived value. The user explicitly chose this in the initial spec.
- **Sample rate may be auto-bumped.** Non-integer sps → bump upward and
  WARN. OQPSK with odd sps → bump by +1 and WARN. The effective rate is
  what gets written. The user said "make it apparent to the user," so all
  bumps log at WARNING level.
- **Bits are zero-padded** to a whole symbol if not divisible by
  `bits_per_symbol`. Logged at WARNING.
- **`filter_type: none` uses NRZ sample-and-hold**, not zero-stuffing.
  This is the right behavior — see Gotchas.
- **GUI is additive.** It must not modify the core pipeline. It builds the
  same config dict as the YAML loader.
- **GUI threading uses a `queue.Queue`.** Worker threads NEVER call
  `root.after` directly; they push results onto the queue. The main thread
  drains it via `_poll_result_queue` (recurring `root.after(50, ...)`).
  An earlier version that called `root.after` from a worker crashed with
  "main thread is not in main loop."
- **Multi-frequency input UI.** The user requested entering several freqs
  in the existing Center freq field with space delimiters. Semantics:
  *single* value → SigMF center frequency, one carrier at offset 0
  (unchanged single-carrier behavior); *two or more* values → every value
  is a baseband carrier offset, SigMF `core:frequency` defaults to 0.
  This was changed after the user reported the original "first =
  center, rest = offsets" rule was confusing (they expected every value
  to produce a carrier). YAML stays explicit via the `channels:` block.
- **Output is one file.** Even with multi-freq, only one data file is
  produced. SigMF annotations describe per-carrier metadata.
- **Bandwidth validation is conservative.** `bw_single = symbol_rate·(1+β)/2`
  for RRC/RC, `symbol_rate` (main-lobe) for everything else. Error message
  includes the minimum required `sample_rate`.

## Gotchas (read before changing things)

1. **`_zero_upsample` vs `_hold_upsample`.** These are NOT
   interchangeable.
   - Zero-stuff (`out[::sps] = sig`) is REQUIRED before a pulse-shaping
     filter, because the filter's impulse response is the pulse shape.
     Using sample-and-hold here would convolve a rect with the filter —
     wrong.
   - Sample-and-hold (`np.repeat(sig, sps)`) is REQUIRED when no filter is
     applied, otherwise you get a flat-spectrum impulse train instead of
     a sinc spectrum. This bug was reported by the user and is now fixed.

2. **Tkinter is not thread-safe.** Worker threads must NOT touch widgets
   or call `root.after`. Use `self._result_queue` and let
   `_poll_result_queue` (running on the main thread) handle results.

3. **OQPSK requires even sps.** Half-symbol Q-channel delay is implemented
   by inserting `sps // 2` zeros. Odd sps would lose precision and break
   the offset. `config.py` enforces this and bumps sps if needed.

4. **π/4-DQPSK Gray flag is informational.** The Gray-coded mapping is
   baked into the IS-54 delta table; toggling `gray_coding: false` has
   no effect for `pi4_qpsk`. Tooltip and example.yaml note this.

5. **`matplotlib.Figure.tight_layout` occasionally complains** about
   colorbar/text overlap. `plotting.py::render()` wraps it in a
   try/except — leave it that way.

6. **`Path` doesn't survive YAML round-trip.** When saving presets,
   `output_dir` is written as a string. The config loader handles both.

7. **Multi-freq renormalization.** Concurrent (FDM) sum can exceed unit
   amplitude. `_apply_channels` calls `_normalize` again after summing.
   Hopping doesn't need this — at any moment only one carrier is active.

8. **Hopping phase is discontinuous at boundaries on purpose.** Each
   hop is mixed using `t = arange(start, end) / fs`, which produces a
   different phase at the start of each hop. This matches how a real
   frequency hopper looks.

## Tests

`tests/smoke_test.py` (100 cases):

- 9 modulations × 5 filters × 2 formats = 90 cases
- Edge cases (10):
  - bitstring source
  - file source msb_first / lsb_first
  - duration_sec source
  - sample_rate auto-adjust (non-integer sps)
  - OQPSK odd-sps auto-bump
  - partial-symbol zero-pad
  - multi-freq concurrent (3 carriers, FDM)
  - multi-freq hopping (round-robin)
  - multi-freq Nyquist guard (must raise with helpful error)

Run: `python3 tests/smoke_test.py`. Cleans up `smoke_output/` automatically.

If you add a new modulation, filter, or source, extend the relevant
lists/loops here. If you add a new validation rule, add a failing-case
assertion that confirms it raises.

## File map (where to look)

| Want to change… | Edit… |
|---|---|
| Add a new modulation | `iqgen/mappers.py` + `BITS_PER_SYMBOL` in `config.py` + GUI `MODULATIONS` list |
| Add a new pulse-shaping filter | `iqgen/filters.py` + `VALID_FILTERS` in `config.py` + GUI `FILTERS` list |
| Add a new data source type | `iqgen/sources.py::DataSource.from_config` + GUI `SOURCE_TYPES` + new LabelFrame in `_tab_source` |
| Change SigMF annotation contents | `iqgen/writers.py::SigMFWriter.write` |
| Add a new plot panel | `iqgen/plotting.py` (new `_func`, add to `render` grid) |
| Add a new YAML field | `iqgen/config.py::_validate_and_derive` + dataclass field + `configs/example.yaml` doc + GUI `_make_vars` / `_build_config_dict` / `_populate_from_dict` + tooltip in `T` dict |
| Tweak GUI layout | `iqgen/gui.py::_tab_*` methods |

## Things the user has asked for repeatedly

- **Tooltips with constraints on every field.** Always update the `T` dict
  in `gui.py` when adding a new field, and attach it via the row helpers
  (`tip=T["..."]`).
- **Don't break working code.** Phases are additive — the GUI must not
  modify pipeline behavior, multi-freq must not break single-carrier
  behavior, etc.
- **Surface auto-adjustments to the user.** WARNING-level logs for any
  auto-bump or padding so the user can see what the tool changed.
- **Backwards compatibility for YAML presets.** New fields must be
  optional with sensible defaults; old presets must still load.

## Things to avoid

- Don't merge `_zero_upsample` and `_hold_upsample` — they serve
  different purposes (see Gotcha #1).
- Don't call `root.after` from a worker thread (see Gotcha #2).
- Don't import pyplot in `plotting.py` — it would break the embedded Tk
  canvas. Use the `Figure` class and the explicit backend.
- Don't add features the user didn't ask for. The user has corrected
  scope creep before; this codebase is intentionally lean.

## If the user asks for…

- **A new output format**: add a `*Writer` class in `writers.py`, register
  it in `VALID_FORMATS` and the `cli.py` writer-selection logic, expose
  it in GUI `FORMATS` and the Output tab.
- **CFO / sample-rate offset / IQ imbalance impairments**: add a new
  optional `impairments:` block in YAML, validate in `config.py`, apply
  in a new pipeline stage between normalize and multi-freq mix.
- **Multiple distinct signals at different offsets** (different
  modulations per carrier): this is bigger than the current multi-freq
  feature, which reuses the same baseband. Would need a list-of-signals
  config schema and multiple `IQGenerator` runs summed with offsets.
- **Real-time / streaming output**: current pipeline is batch-only.
  Would require restructuring `generate()` into a generator that yields
  blocks.

## Environment

- Repository root: clone of `iqgen` (no special path assumptions).
- Git repo (committed); no remote-specific configuration is required for
  local development.
- Python 3 (use `python3`, not `python` — `python` is not on PATH).
- Dependencies pinned in `requirements.txt`.
- Outputs go to `./output/` by default; smoke tests use `./smoke_output/`.
