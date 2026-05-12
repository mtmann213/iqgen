from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import SignalConfig

log = logging.getLogger(__name__)


def make_filename_stem(cfg: SignalConfig, timestamp_str: Optional[str] = None) -> str:
    parts = []
    if cfg.timestamp:
        if timestamp_str is None:
            timestamp_str = datetime.now(timezone.utc).strftime(cfg.timestamp_format)
        parts.append(timestamp_str)
    parts.extend([
        cfg.name,
        cfg.modulation,
        f"{int(round(cfg.bitrate))}Hz",
        f"{int(round(cfg.sample_rate))}Hz",
    ])
    return "_".join(parts)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Cf32Writer:
    """Raw interleaved IQ as little-endian complex64 (.cf32)."""

    def write(self, signal: np.ndarray, cfg: SignalConfig) -> Path:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        stem = make_filename_stem(cfg)
        path = cfg.output_dir / f"{stem}.cf32"
        signal.astype(np.complex64).tofile(path)
        log.info("Wrote cf32: %s (%d samples)", path, signal.size)
        return path


class SigMFWriter:
    """SigMF 1.0.0: writes both .sigmf-data (cf32_le) and .sigmf-meta (JSON)."""

    SIGMF_VERSION = "1.0.0"

    # User-provided keys we map into the SigMF `global` object as `core:*`
    GLOBAL_PASSTHROUGH = (
        "hardware", "description", "author", "license",
        "recorder", "version",
    )

    def write(self, signal: np.ndarray, cfg: SignalConfig) -> Tuple[Path, Path]:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        stem = make_filename_stem(cfg)
        data_path = cfg.output_dir / f"{stem}.sigmf-data"
        meta_path = cfg.output_dir / f"{stem}.sigmf-meta"
        samples = signal.astype(np.complex64)
        samples.tofile(data_path)

        now_iso = _iso_utc_now()
        meta_user = cfg.sigmf_meta or {}

        global_meta = {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(cfg.sample_rate),
            "core:version": self.SIGMF_VERSION,
            "core:num_channels": 1,
            "core:datetime": now_iso,
        }
        # User overrides for global metadata (core:author, core:description, …)
        for k in self.GLOBAL_PASSTHROUGH:
            v = meta_user.get(k)
            if v is not None and v != "":
                global_meta[f"core:{k}"] = v
        # User-supplied free-form extras under "global_extra"
        for k, v in (meta_user.get("global_extra") or {}).items():
            global_meta[k] = v

        captures = [{
            "core:sample_start": 0,
            "core:frequency": float(cfg.center_frequency_hz),
            "core:datetime": now_iso,
        }]

        # Common per-annotation fields (modulation/filter/rate metadata is the
        # same for every carrier and every hop — they share the baseband).
        def _base_annotation() -> dict:
            ann = {
                "core:label": cfg.modulation.upper(),
                "core:description": meta_user.get(
                    "comment",
                    f"{cfg.modulation.upper()} @ {cfg.bitrate:g} bps, "
                    f"sps={cfg.samples_per_symbol}, filter={cfg.filter_type}",
                ),
                "iqgen:modulation": cfg.modulation,
                "iqgen:bitrate": float(cfg.bitrate),
                "iqgen:symbol_rate": float(cfg.symbol_rate),
                "iqgen:samples_per_symbol": int(cfg.samples_per_symbol),
                "iqgen:gray_coding": bool(cfg.gray_coding),
                "iqgen:pulse_shaping": cfg.filter_type,
            }
            if cfg.filter_type in ("root_raised_cosine", "raised_cosine"):
                ann["iqgen:roll_off"] = float(cfg.roll_off)
            if cfg.filter_type == "gaussian":
                ann["iqgen:bt_product"] = float(cfg.bt_product)
            return ann

        offsets = cfg.channel_offsets_hz or [0.0]
        annotations = []
        if cfg.channel_mode == "hopping" and len(offsets) > 1:
            # One annotation per hop window (round-robin through offsets).
            fs = float(cfg.sample_rate)
            spc = max(1, int(round((cfg.hop_duration_sec or 0.0) * fs)))
            n = int(samples.size)
            for hop_idx, start in enumerate(range(0, n, spc)):
                end = min(start + spc, n)
                f = offsets[hop_idx % len(offsets)]
                ann = _base_annotation()
                ann["core:sample_start"] = int(start)
                ann["core:sample_count"] = int(end - start)
                ann["iqgen:offset_hz"] = float(f)
                ann["iqgen:hop_index"] = int(hop_idx)
                ann["core:freq_lower_edge"] = (
                    float(cfg.center_frequency_hz) + f - cfg.symbol_rate / 2.0
                )
                ann["core:freq_upper_edge"] = (
                    float(cfg.center_frequency_hz) + f + cfg.symbol_rate / 2.0
                )
                annotations.append(ann)
        else:
            # Concurrent (or single-carrier): one annotation per carrier,
            # all spanning the full recording.
            for f in offsets:
                ann = _base_annotation()
                ann["core:sample_start"] = 0
                ann["core:sample_count"] = int(samples.size)
                ann["iqgen:offset_hz"] = float(f)
                ann["core:freq_lower_edge"] = (
                    float(cfg.center_frequency_hz) + f - cfg.symbol_rate / 2.0
                )
                ann["core:freq_upper_edge"] = (
                    float(cfg.center_frequency_hz) + f + cfg.symbol_rate / 2.0
                )
                annotations.append(ann)

        global_meta["iqgen:channel_mode"] = cfg.channel_mode
        if cfg.channel_mode == "hopping":
            global_meta["iqgen:hop_duration_sec"] = float(cfg.hop_duration_sec or 0.0)

        meta = {
            "global": global_meta,
            "captures": captures,
            "annotations": annotations,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Wrote SigMF: %s + %s (%d samples)",
                 data_path, meta_path, samples.size)
        return data_path, meta_path
