# iqgen

Synthetic IQ signal generator. Produces `.cf32` (raw interleaved complex64) or
SigMF 1.0.0 (`.sigmf-data` + `.sigmf-meta`) files from a YAML configuration,
with an optional Tkinter GUI for interactive use.

## Features

- **10 modulations**: BPSK, DBPSK, π/2-BPSK, QPSK, DQPSK, π/4-DQPSK, OQPSK,
  8PSK, D8PSK, π/4-8PSK. Gray coding and differential encoding supported
  where applicable.
- **5 pulse-shaping filters**: none (NRZ sample-and-hold), root-raised-cosine,
  raised-cosine, Gaussian (BT product), rectangular.
- **3 data sources**: random (with optional seed), file (packed bits,
  msb/lsb-first), bitstring (literal `0`/`1` characters).
- **Interconnected rates**: `bitrate` is the source of truth.
  `symbol_rate = bitrate / bits_per_symbol` and
  `samples_per_symbol = sample_rate / symbol_rate` are derived. `sample_rate`
  is auto-bumped (with a `WARNING`) when needed to keep `sps` integer (and
  even for OQPSK).
- **Multi-frequency**: optional `channels:` block. `concurrent` mode (FDM)
  sums copies of the baseband at each offset; `hopping` mode cycles
  round-robin through offsets every `hop_duration_sec`.
- **Two output formats**: `cf32` (single file) and `sigmf` (data + JSON
  metadata, with one annotation per carrier or per hop).
- **GUI**: Tkinter form with tooltips on every field, live 4-panel plot
  (constellation, IQ vs time, PSD, spectrogram), save/load YAML presets,
  PNG plot exported alongside each generated data file.

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `numpy>=1.22`, `scipy>=1.9`, `PyYAML>=6.0`, `matplotlib>=3.5`.

## Usage

### Command line

```bash
python -m iqgen configs/example.yaml
python -m iqgen configs/example.yaml -v        # DEBUG logging
```

### GUI

```bash
python -m iqgen.gui
```

The GUI builds the same config dict as the YAML loader, so anything the CLI
accepts can be expressed in the form (and vice versa via Save preset).

### Verifier (demodulate a recording back to bits)

```bash
# SigMF input — parameters auto-read from the .sigmf-meta file
python -m iqgen.verify_cli recording.sigmf-meta --bits "10110010..."

# Raw .cf32 — supply parameters manually
python -m iqgen.verify_cli sig.cf32 -m qpsk -s 1e6 -b 100e3 \
    -f root_raised_cosine --roll-off 0.35 --bits-file expected.txt

# GUI version (auto-fill from SigMF, manual entry for cf32):
python -m iqgen.verifier_gui
```

The verifier is a noise-free matched receiver: it knows the parameters
(from SigMF metadata or supplied flags) and demodulates accordingly.
Supports every modulation/filter combo iqgen generates plus multi-frequency
(concurrent + hopping). Reports BER when expected bits are provided.

### Smoke tests

```bash
python3 tests/smoke_test.py
```

Covers all 9 modulations × 5 filters × 2 formats plus edge cases
(bitstring/file/duration sources, sps auto-adjust, OQPSK even-sps bump,
partial-symbol padding, multi-freq concurrent/hopping, Nyquist guard).

## Configuration

See `configs/example.yaml` for a fully-commented template. Minimal example:

```yaml
signal:
  name: test
  center_frequency_hz: 915e6
  sample_rate: 1e6
  normalization: peak           # peak | rms | none
  output_dir: ./output
  timestamp: true

source:
  type: random                  # random | file | bitstring
  bit_count: 10000
  seed: 42

modulation:
  type: qpsk                    # bpsk dbpsk qpsk dqpsk pi4_qpsk oqpsk 8psk d8psk pi4_8psk
  gray_coding: true

rate:
  bitrate: 100000               # bits/sec — SOURCE OF TRUTH

pulse_shaping:
  filter_type: root_raised_cosine
  span_symbols: 10
  roll_off: 0.35

output:
  format: sigmf                 # cf32 | sigmf
```

### Multi-frequency (optional)

```yaml
channels:
  mode: concurrent              # concurrent | hopping
  offsets_hz: [-100e3, 100e3]   # baseband offsets relative to center
  # hop_duration_sec: 0.01      # required when mode = hopping
```

`concurrent` (FDM) sums copies of the baseband signal at each offset and
renormalizes the result. `hopping` cycles through offsets every
`hop_duration_sec`; phase is discontinuous at each hop boundary (like a real
frequency hopper).

In the GUI: the Center freq field accepts space-delimited values. With a
single value it's the SigMF center frequency (single-carrier RF mode). With
two or more values, every value is a baseband carrier offset — e.g.
`1000 3000 5000` produces three carriers at 1, 3, 5 kHz, and SigMF
`core:frequency` is recorded as 0. Mode and hop duration live in the
"Multi-frequency" subsection of the Signal tab.

Each offset must satisfy `|offset| + bandwidth_single_sided ≤ sample_rate/2`;
violation produces an error message containing the minimum required
`sample_rate`. Bandwidth is `symbol_rate·(1+roll_off)/2` for RRC/RC and
`symbol_rate` for other filters (conservative NRZ main-lobe estimate).

## Pipeline

```
bits ─► symbols ─► upsample ─► pulse-shape ─► normalize ─► multi-freq mix ─► write
```

- **Upsample** has two modes:
  - `filter_type: none` → sample-and-hold (NRZ): repeats each symbol `sps`
    times. Spectrum is `sinc(f / symbol_rate)`.
  - Any actual filter → zero-stuff then convolve; the filter's impulse
    response *is* the pulse shape.
- **Multi-freq mix** is a no-op fast path when there's a single offset of 0.

## Output

- `cf32`: little-endian complex64 interleaved I/Q. Filename:
  `[timestamp_]name_modulation_<bitrate>Hz_<sample_rate>Hz.cf32`
- `sigmf`: paired `.sigmf-data` (cf32_le) and `.sigmf-meta` (JSON, SigMF
  1.0.0). Required core fields are always written; user-supplied
  `output.sigmf.*` keys map into `core:*` (author/description/license/
  hardware/recorder/version) or are passed through under `global_extra`.
  Annotations include per-carrier `iqgen:offset_hz`, `core:freq_lower_edge`,
  and `core:freq_upper_edge`.
- When using the GUI, a PNG of the 4-panel plot is saved next to each data
  file (rendered once and reused for both the embedded canvas and the PNG).

## Project layout

```
iqgen/
├── __init__.py
├── __main__.py        # python -m iqgen
├── cli.py             # CLI entry point
├── config.py          # SignalConfig dataclass + YAML parsing/validation
├── sources.py         # DataSource: random / file / bitstring
├── mappers.py         # 9 modulation mappers + create_mapper factory
├── filters.py         # PulseShaper: RRC / RC / Gaussian / rectangular
├── generator.py       # IQGenerator pipeline + multi-freq stage
├── writers.py         # Cf32Writer, SigMFWriter
├── plotting.py        # 4-panel render() + headless save_png()
├── gui.py             # Tkinter form (additive — does not modify core)
├── verifier.py        # Demodulator: IQ file → bits (inverse pipeline)
├── verify_cli.py      # CLI for the verifier
└── verifier_gui.py    # Tkinter GUI for the verifier

configs/example.yaml   # fully commented reference config
tests/smoke_test.py    # 100-case smoke test
```
