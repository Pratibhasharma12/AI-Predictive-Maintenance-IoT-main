from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered Predictive Maintenance System for IoT Devices"
    )
    parser.add_argument(
        "--mode",
        choices=["full"],
        default="full",
        help="Pipeline mode. 'full' runs the complete workflow.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Probability threshold used for alert generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where plots and reports are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "full":
        run_full_pipeline(alert_threshold=args.threshold, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

