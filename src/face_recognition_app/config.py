from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    default_cfg = _read_yaml(DEFAULT_CONFIG_PATH)
    if config_path is None:
        return default_cfg

    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT_ROOT / config_path
    user_cfg = _read_yaml(config_path)
    return _deep_merge(default_cfg, user_cfg)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def model_store_root(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["models"].get("root", "models/arcface"))


def insightface_pack_dir(cfg: dict[str, Any]) -> Path:
    return model_store_root(cfg) / cfg["models"]["insightface_name"]


def insightface_pack_zip(cfg: dict[str, Any]) -> Path:
    return model_store_root(cfg) / f"{cfg['models']['insightface_name']}.zip"


def tensorrt_detection_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["models"].get("tensorrt_detection_dir", "models/tensorrt"))


def detection_engine_path(cfg: dict[str, Any], suffix: str = "fp16") -> Path:
    det_w, det_h = tuple2(cfg["models"]["detection_size"], "models.detection_size")
    model_name = cfg["models"]["insightface_name"]
    return tensorrt_detection_dir(cfg) / model_name / f"detection_{det_w}x{det_h}_{suffix}.engine"


def tuple2(value: Any, name: str) -> tuple[int, int]:
    if value is None:
        raise ValueError(f"Missing size config: {name}")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly 2 numbers")
    return int(value[0]), int(value[1])


def color_tuple(value: Any, name: str) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers")
    return int(value[0]), int(value[1]), int(value[2])
