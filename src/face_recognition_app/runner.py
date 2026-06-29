from __future__ import annotations

from pathlib import Path

from .config import PROJECT_ROOT, load_config
from .enrollment import run_enrollment
from .pipeline import run_recognition


def recognize(config_file: str | Path) -> None:
    run_recognition(load_config(PROJECT_ROOT / config_file))


def enroll_embedding(config_file: str | Path) -> None:
    run_enrollment(load_config(PROJECT_ROOT / config_file))
