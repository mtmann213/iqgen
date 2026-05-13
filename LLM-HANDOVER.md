# LLM Handover — iqgen

This document is intended for an LLM picking up the project. It captures the
build history, design decisions, gotchas, and current state in enough detail
that you can continue without re-deriving context.

## Current status

**Working state.** All 172 smoke tests pass
(`python3 tests/smoke_test.py`). CLI and GUI both function end-to-end.

Last completed feature: packet framing — `[preamble][syncword][FEC(header
‖ payload ‖ CRC)]` layer with a frame-aware verifier and full diagnostic
panel in the verifier GUI (sync offset/distance, header fields, FEC
codewords + corrections + corrected positions, CRC computed vs expected,
payload BER + first error positions).

No known bugs. Likely next user request: a channel/interference layer
(AWGN, tone, custom IQ file) at varying SNR/SIR with a BER sweep. See
"If the user asks for…" below.

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
7. **Verifier** — noise-free matched receiver that recovers bits from a
   .cf32 or .sigmf-data file. CLI and Tkinter GUI. Supports every
   mod/filter combo + multi-frequency. Lives in `verifier.py`,
   `verify_cli.py`, `verifier_gui.py`. 50 round-trip cases (generate →
   verify) are part of the smoke suite.
8. **Packet framing** (most recent) — `iqgen/framing.py` adds CRC
   (CCITT-16, CRC-32), FEC (Hamming(7,4), repetition-3, none), and a
   `FrameConfig` driving `build_frame()` / `parse_frame()`. New `framed`
   source in `sources.py` wraps any payload source. Verifier exposes
   `demodulate_frame()` returning recovered bits + a `FrameReport` with
   every diagnostic layer. `verifier_gui.py` got a scrollable left panel
   with a Framing LabelFrame (knobs for preamble/sync/header/CRC/FEC)
   and a multi-section diagnostic display. 12 framing smoke tests
   (all FEC×CRC combos at bit level + FEC-correctable error case +
   uncorrectable→CRC-detected case + modulated round-trip).

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
  recovered (post-matched-filter) symbols after Verify. Left panel is
  wrapped in a Canvas with a vertical scrollbar (the Framing section
  adds height beyond the visible window).
- `framing.py` — packet framing. On the TX side: `build_frame(payload,
  FrameConfig)` produces `[preamble][syncword][FEC(header‖payload‖CRC)]`.
  On the RX side: `parse_frame(bits, FrameConfig)` returns a
  `FrameReport` with sync offset/distance, header fields, FEC stats
  (codewords/corrections/positions), CRC pass/fail, payload bits, and
  optional payload BER vs expected. CRC is computed over (header‖payload)
  BEFORE FEC. FEC over the whole protected block (header+payload+CRC) as
  one contiguous bit stream. Preamble and syncword are transmitted
  uncoded for sync correlation. `find_sync()` does sliding Hamming-
  distance search against the combined `preamble||syncword` pattern.

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

9. **Frame CRC vs FEC ordering.** CRC is computed over (header ‖
   payload), THEN FEC encodes (header ‖ payload ‖ CRC) as one block.
   On decode: FEC first → CRC second. Don't swap; CRC-after-FEC is
   what gives the useful diagnostic ("FEC corrected N bits, CRC then
   passed/failed").

10. **Hamming(7,4) mis-corrects with 2 bit errors.** Beyond capacity it
    silently outputs wrong data and reports a correction. The CRC is
    your safety net. The smoke test relies on this: a 2-bit corruption
    in one codeword produces "FEC corrections: 1, CRC: FAIL" — that's
    a valid pass, not a bug.

11. **`IQGenerator.source` is set during `generate()`.** Useful for
    inspecting the actual generated payload (e.g. `gen.source._last_payload`
    on a FramedSource) for BER comparison. Before `generate()` it's None.

## Tests

`tests/smoke_test.py` (172 cases):

- 10 modulations × 5 filters × 2 formats = 100 cases
- Edge cases (~10): bitstring/file/duration sources, sample-rate
  auto-adjust, OQPSK odd-sps bump, partial-symbol zero-pad, multi-freq
  concurrent/hopping, Nyquist guard.
- Verifier round-trip: 10 modulations × 5 filters = 50 cases (generate
  → demod → BER must be 0).
- Framing (12 cases): every FEC×CRC combination at the bit level +
  Hamming 1-bit FEC-correctable + Hamming 2-bit-in-codeword
  uncorrectable (must trip CRC) + a modulated end-to-end framed
  round-trip (QPSK + RRC + Hamming + CRC-16).

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
| Add a new CRC or FEC | `iqgen/framing.py` (`CRC_SPECS` for CRCs; new branch in `fec_encode`/`fec_decode`/`fec_overhead_bits` for FEC) + add to `CRC_OPTIONS` / `FEC_OPTIONS` in `verifier_gui.py` |
| Change frame layout (header fields, sync word) | Frame is per-instance via `FrameConfig`. For new *defaults*, edit `_DEFAULT_*` at top of `framing.py` |
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
- **Channel / interference simulation** (likely next request): the user
  has already sketched it. Plan was: new `iqgen/channel.py` exposing
  `mix(signal, interferer, target_db, mode='snr'|'sir', ...)` plus
  interferer constructors (`awgn`, `tone`, `from_file`); new
  `iqgen.evaluate` CLI that sweeps SNR and reports BER per point by
  feeding mixed signals to the existing `demodulate`/`demodulate_frame`.
  Keep `channel.py` payload-agnostic — it just mixes IQ. Framing
  diagnostics naturally tell the story (sync still found? FEC
  corrections climbing? CRC failures?).
- **More FEC schemes** (convolutional, Reed-Solomon, LDPC): add to
  `framing.py` following the Hamming(7,4) pattern. Each needs
  `fec_encode`, `fec_decode` (returning `FecDecodeResult`), and
  `fec_overhead_bits`. Update `FEC_OPTIONS` in `verifier_gui.py`.
- **Frame on a noisy channel**: `parse_frame` already accepts
  `max_sync_distance`. When the interference layer lands, expose this
  knob in the GUI so the user can tune sync tolerance vs false-positive
  rate.

## Environment

- Repository root: clone of `iqgen` (no special path assumptions).
- Git repo (committed); no remote-specific configuration is required for
  local development.
- Python 3 (use `python3`, not `python` — `python` is not on PATH).
- Dependencies pinned in `requirements.txt`.
- Outputs go to `./output/` by default; smoke tests use `./smoke_output/`.
