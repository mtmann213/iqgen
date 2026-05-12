from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import SignalConfig
from .generator import IQGenerator
from .writers import Cf32Writer, SigMFWriter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iqgen",
        description="Generate IQ files (cf32 or SigMF) from a YAML configuration.",
    )
    p.add_argument("config", type=Path, help="Path to YAML configuration file")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = SignalConfig.from_yaml(args.config)
    signal = IQGenerator(cfg).generate()

    writer = SigMFWriter() if cfg.format == "sigmf" else Cf32Writer()
    writer.write(signal, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
