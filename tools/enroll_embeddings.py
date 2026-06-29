#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_recognition_app.config import load_config
from src.face_recognition_app.enrollment import run_enrollment


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ArcFace embeddings from configured face data.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--data-dir", default=None, help="Override paths.face_data_dir.")
    parser.add_argument("--output", default=None, help="Override paths.database_path.")
    args = parser.parse_args()

    run_enrollment(load_config(args.config), data_dir=args.data_dir, output_path=args.output)


if __name__ == "__main__":
    main()
