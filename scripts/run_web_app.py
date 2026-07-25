from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Nepali handwritten letter detector web app."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind. Use 0.0.0.0 for local network access.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to serve on.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app.run(debug=args.debug, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
