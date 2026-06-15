#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx

import _bootstrap  # noqa: F401

from face_recognition_app.config import load_config, resolve_path
from face_recognition_app.model_store import detection_onnx_path, ensure_model_pack_downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert face detection ONNX model FP32 to FP16.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--model", default=None, help="InsightFace model pack name, e.g. buffalo_l, buffalo_s, antelopev2.")
    parser.add_argument("--input", default=None, help="Input detection ONNX. Defaults to model pack detector.")
    parser.add_argument("--output", default=None, help="Output FP16 ONNX path.")
    parser.add_argument("--no-download", action="store_true", help="Do not auto-download missing InsightFace model pack.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["models"]["insightface_name"] = args.model
    if not args.no_download:
        ensure_model_pack_downloaded(cfg)

    input_path = _resolve_path_arg(args.input) if args.input else _default_detection_model(cfg)
    output_path = _resolve_path_arg(args.output) if args.output else input_path.with_name(f"{input_path.stem}_fp16.onnx")

    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise SystemExit("Missing onnxconverter-common. Install it with: pip install onnxconverter-common") from exc

    if not input_path.exists():
        raise SystemExit(f"Input model not found: {input_path}")

    print(f"Loading: {input_path}")
    model = onnx.load(str(input_path))
    print("Converting FP32 -> FP16...")
    fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(fp16_model, str(output_path))
    print(f"Saved: {output_path}")
    print(f"Size: {input_path.stat().st_size / 1024 / 1024:.2f} MB -> {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def _default_detection_model(cfg: dict) -> Path:
    try:
        return detection_onnx_path(cfg)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}. Run without --no-download once.") from exc


def _resolve_path_arg(path: str | None) -> Path:
    if path is None:
        raise ValueError("path is required")
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return resolve_path(candidate)


if __name__ == "__main__":
    main()
