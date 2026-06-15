from __future__ import annotations

import os
from ctypes.util import find_library
from pathlib import Path

import onnxruntime as ort

from .config import resolve_path


def has_tensorrt_runtime() -> bool:
    if find_library("nvinfer") is not None:
        return True
    return any(_candidate_tensorrt_lib_dirs())


def tensorrt_library_dirs() -> list[Path]:
    return _candidate_tensorrt_lib_dirs()


def build_ort_provider_config(cfg: dict, use_tensorrt: bool | None = None) -> tuple[list[str], list[dict[str, str]]]:
    runtime_cfg = cfg["runtime"]
    available = ort.get_available_providers()
    providers: list[str] = []
    provider_options: list[dict[str, str]] = []

    gpu_id = str(runtime_cfg.get("gpu_id", 0))
    enable_tensorrt = bool(runtime_cfg.get("use_tensorrt", False)) if use_tensorrt is None else use_tensorrt
    if enable_tensorrt and has_tensorrt_runtime() and "TensorrtExecutionProvider" in available:
        cache_dir = resolve_path(runtime_cfg["tensorrt_engine_cache_dir"])
        os.makedirs(cache_dir, exist_ok=True)

        providers.append("TensorrtExecutionProvider")
        provider_options.append({
            "device_id": gpu_id,
            "trt_fp16_enable": "True" if runtime_cfg.get("tensorrt_fp16", True) else "False",
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": str(cache_dir),
        })

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        provider_options.append({"device_id": gpu_id})

    providers.append("CPUExecutionProvider")
    provider_options.append({})
    return providers, provider_options


def _candidate_tensorrt_lib_dirs() -> list[Path]:
    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / "python3.11" / "site-packages" / "tensorrt_libs")

    try:
        import tensorrt  # noqa: F401
        import site

        for site_dir in site.getsitepackages():
            candidates.append(Path(site_dir) / "tensorrt_libs")
    except Exception:
        pass

    return [path for path in candidates if (path / "libnvinfer.so.10").exists()]
