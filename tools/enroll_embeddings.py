#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from face_recognition_app.config import load_config
from face_recognition_app.enrollment import run_enrollment


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ArcFace embeddings from configured face data.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--data-dir", default=None, help="Override paths.face_data_dir.")
    parser.add_argument("--output", default=None, help="Override paths.database_path.")
    args = parser.parse_args()

    run_enrollment(load_config(args.config), data_dir=args.data_dir, output_path=args.output)


if __name__ == "__main__":
    main()
