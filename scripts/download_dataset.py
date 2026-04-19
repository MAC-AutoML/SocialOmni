#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths import PATHS
from models.utils.dataset_downloader import DATASET_SPECS, ensure_default_dataset_available


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SocialOmni benchmark data into the local data/ directory.")
    parser.add_argument(
        "--level",
        choices=["level1", "level2", "all"],
        default="all",
        help="Which benchmark split to download",
    )
    args = parser.parse_args()

    levels = ["level1", "level2"] if args.level == "all" else [args.level]

    for level in levels:
        spec = DATASET_SPECS[level]
        dataset_path = PATHS.root / spec.dataset_relpath
        video_dir = PATHS.root / spec.video_dir_relpath
        changed = ensure_default_dataset_available(level, dataset_path, video_dir)
        status = "downloaded" if changed else "already available"
        print(f"{level}: {status} -> dataset={dataset_path} videos={video_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
