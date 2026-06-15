#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import onnx

import _bootstrap  # noqa: F401

from face_recognition_app.config import (
    detection_engine_path,
    load_config,
    resolve_path,
    tuple2,
)
from face_recognition_app.model_store import detection_onnx_path, ensure_model_pack_downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Export face detection ONNX model to TensorRT engine using trtexec.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--model", default=None, help="InsightFace model pack name, e.g. buffalo_l, buffalo_s, antelopev2.")
    parser.add_argument("--input", default=None, help="Input detection ONNX. Defaults to model pack detector.")
    parser.add_argument("--output", default=None, help="Output .engine path.")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 builder mode.")
    parser.add_argument("--workspace", type=int, default=4096, help="Workspace size in MiB.")
    parser.add_argument("--no-download", action="store_true", help="Do not auto-download missing InsightFace model pack.")
    parser.add_argument("--dry-run", action="store_true", help="Print trtexec command without running it.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["models"]["insightface_name"] = args.model

    if not args.no_download:
        ensure_model_pack_downloaded(cfg)

    input_path = _resolve_input(args.input) if args.input else _default_detection_model(cfg)
    if not input_path.exists():
        raise SystemExit(f"Input model not found: {input_path}")

    suffix = "fp32" if args.fp32 else "fp16"
    output_path = _resolve_output(args.output) if args.output else detection_engine_path(cfg, suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_name = _onnx_input_name(input_path)
    det_w, det_h = tuple2(cfg["models"]["detection_size"], "models.detection_size")
    shape = f"{input_name}:1x3x{det_h}x{det_w}"
    cmd = [
        "trtexec",
        f"--onnx={input_path}",
        f"--saveEngine={output_path}",
        f"--minShapes={shape}",
        f"--optShapes={shape}",
        f"--maxShapes={shape}",
        f"--memPoolSize=workspace:{args.workspace}",
    ]
    if not args.fp32:
        cmd.append("--fp16")

    print("TensorRT export command:")
    print(" ".join(str(part) for part in cmd))
    if args.dry_run:
        return

    if shutil.which("trtexec") is None:
        raise SystemExit("trtexec not found. Install TensorRT and make sure trtexec is in PATH.")

    subprocess.run(cmd, check=True)
    print(f"Saved TensorRT engine: {output_path}")
    print("Note: TensorRT engines are GPU/driver/TensorRT-version specific. Rebuild on deployment machines.")


def _default_detection_model(cfg: dict) -> Path:
    try:
        return detection_onnx_path(cfg)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}. Run without --no-download once.") from exc


def _resolve_input(path: str | None) -> Path:
    if path is None:
        raise ValueError("path is required")
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return resolve_path(candidate)


def _resolve_output(path: str | None) -> Path:
    if path is None:
        raise ValueError("path is required")
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return resolve_path(candidate)


def _onnx_input_name(path: Path) -> str:
    model = onnx.load(str(path))
    return model.graph.input[0].name


if __name__ == "__main__":
    main()
