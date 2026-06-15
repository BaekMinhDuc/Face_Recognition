#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_recognition_app.config import load_config  # noqa: E402
from face_recognition_app.pipeline import run_recognition  # noqa: E402


LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


if __name__ == "__main__":
    config_path = LOCAL_CONFIG_PATH if LOCAL_CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    run_recognition(config)
