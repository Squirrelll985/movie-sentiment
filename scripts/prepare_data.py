# -*- coding: utf-8 -*-
"""Prepare the IMDb dataset once and cache tokenised arrays."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment.config import Config  # noqa: E402
from sentiment.data import prepare_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = Config.load().resolve()
    bundle = prepare_data(config, force=args.force)
    print(
        f"train={bundle.train_x.shape} val={bundle.val_x.shape} "
        f"test={bundle.test_x.shape} vocab={bundle.vocab_size}"
    )


if __name__ == "__main__":
    main()
