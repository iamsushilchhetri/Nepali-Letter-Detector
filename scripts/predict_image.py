from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nepali_letter_detector.inference import Predictor


def configure_stdout_for_unicode() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a handwritten Nepali character from an image file."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to show.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print predictions as JSON instead of plain text.",
    )
    return parser


def main() -> int:
    configure_stdout_for_unicode()
    parser = build_parser()
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")

    predictor = Predictor()
    with Image.open(args.image_path) as image:
        predictions = predictor.predict(image, top_k=args.top_k)

    if args.json:
        print(json.dumps(predictions, ensure_ascii=False, indent=2))
    else:
        print(f"image: {args.image_path}")
        print(f"checkpoint: {predictor.weights_path}")
        for rank, prediction in enumerate(predictions, start=1):
            print(
                f"{rank}. {prediction['label']} "
                f"({prediction['class']}) "
                f"- {prediction['confidence'] * 100:.2f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
