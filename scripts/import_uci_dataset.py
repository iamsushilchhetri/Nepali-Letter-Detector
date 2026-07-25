from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import the UCI Devanagari dataset into train/validation/test folders."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw_tmp" / "DevanagariHandwrittenCharacterDataset",
        help="Root folder containing Train/ and Test/ from the UCI dataset.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Target project data directory.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of the source Train split to use for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    return parser


def list_png_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.png") if path.is_file())


def clear_placeholder(split_root: Path) -> None:
    gitkeep_path = split_root / ".gitkeep"
    if gitkeep_path.exists():
        gitkeep_path.unlink()


def transfer_file(source: Path, destination: Path, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), str(destination))


def import_split(
    source_train_root: Path,
    source_test_root: Path,
    target_root: Path,
    validation_ratio: float,
    seed: int,
    copy_files: bool,
) -> tuple[int, int, int]:
    random_generator = random.Random(seed)

    train_total = 0
    validation_total = 0
    test_total = 0

    class_directories = sorted(path for path in source_train_root.iterdir() if path.is_dir())
    for source_class_dir in class_directories:
        class_name = source_class_dir.name
        source_files = list_png_files(source_class_dir)
        random_generator.shuffle(source_files)

        validation_count = max(1, int(len(source_files) * validation_ratio))
        validation_files = source_files[:validation_count]
        train_files = source_files[validation_count:]

        for file_path in train_files:
            transfer_file(
                file_path,
                target_root / "train" / class_name / file_path.name,
                copy_files=copy_files,
            )
        for file_path in validation_files:
            transfer_file(
                file_path,
                target_root / "validation" / class_name / file_path.name,
                copy_files=copy_files,
            )

        source_test_class_dir = source_test_root / class_name
        for file_path in list_png_files(source_test_class_dir):
            transfer_file(
                file_path,
                target_root / "test" / class_name / file_path.name,
                copy_files=copy_files,
            )

        train_total += len(train_files)
        validation_total += len(validation_files)
        test_total += len(list_png_files(target_root / "test" / class_name))

    return train_total, validation_total, test_total


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not (0 < args.validation_ratio < 1):
        raise ValueError("--validation-ratio must be between 0 and 1.")

    source_train_root = args.source_root / "Train"
    source_test_root = args.source_root / "Test"
    if not source_train_root.exists() or not source_test_root.exists():
        raise FileNotFoundError(
            "The source dataset folders were not found. "
            "Expected Train/ and Test/ under the provided source root."
        )

    for split_name in ("train", "validation", "test"):
        clear_placeholder(args.target_root / split_name)

    train_total, validation_total, test_total = import_split(
        source_train_root=source_train_root,
        source_test_root=source_test_root,
        target_root=args.target_root,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        copy_files=args.copy,
    )

    print("dataset_import_complete: yes")
    print(f"train_images: {train_total}")
    print(f"validation_images: {validation_total}")
    print(f"test_images: {test_total}")
    print(f"copy_mode: {args.copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
