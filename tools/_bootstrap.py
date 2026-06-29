from __future__ import annotations

import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_ROOT / "_bootstrap.py"))
