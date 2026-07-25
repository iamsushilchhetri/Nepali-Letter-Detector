from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nepali_letter_detector.labels import sorted_label_names


SPLITS = ("train", "validation", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create empty class folders for the Nepali handwritten dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--write-gitkeep",
        action="store_true",
        help="Create a .gitkeep file in every empty class folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without writing to disk.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    created_paths: list[Path] = []
    for split_name in SPLITS:
        for label_name in sorted_label_names():
            class_dir = args.data_dir / split_name / label_name
            created_paths.append(class_dir)
            if args.dry_run:
                continue
            class_dir.mkdir(parents=True, exist_ok=True)
            if args.write_gitkeep:
                (class_dir / ".gitkeep").touch()

    action = "Would create" if args.dry_run else "Created"
    print(f"{action} {len(created_paths)} class directories.")
    for path in created_paths[:6]:
        print(path)
    if len(created_paths) > 6:
        print("...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
