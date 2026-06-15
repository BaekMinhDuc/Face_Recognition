#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from face_recognition_app.config import load_config, resolve_path
from face_recognition_app.database import build_database
from face_recognition_app.models import FaceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ArcFace embeddings from face_data images.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--data-dir", default=None, help="Override face_data directory.")
    parser.add_argument("--output", default=None, help="Override output .npz database path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = resolve_path(args.data_dir or cfg["paths"]["face_data_dir"])
    output_path = resolve_path(args.output or cfg["paths"]["database_path"])
    extensions = {ext.lower() for ext in cfg["enroll"]["image_extensions"]}

    image_paths = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not image_paths:
        raise SystemExit(f"No images found in {data_dir}. Expected extensions: {sorted(extensions)}")

    engine = FaceEngine(cfg)
    samples = []
    failures = []

    for path in image_paths:
        person_name = _label_from_path(data_dir, path)
        embedding, info = engine.extract_image_embedding(path)
        if embedding is None:
            failures.append((person_name, path, info))
            print(f"[SKIP] {path}: {info['reason']}")
            continue
        samples.append((person_name, embedding, str(path.relative_to(data_dir))))
        print(f"[OK] {person_name}: {path.name} score={info['score']:.3f} faces={info['faces']}")

    if not samples:
        raise SystemExit("No valid face embeddings extracted.")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": cfg["models"]["insightface_name"],
        "source_dir": str(data_dir),
        "failed_images": len(failures),
    }
    database = build_database(samples, metadata)
    database.save_npz(output_path)

    print("\nSaved database:")
    print(f"  path: {output_path}")
    print(f"  people: {len(database.names)}")
    print(f"  samples: {len(database.sample_names)}")
    if failures:
        print(f"  skipped: {len(failures)}")


def _label_from_path(data_dir: Path, image_path: Path) -> str:
    rel = image_path.relative_to(data_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return image_path.stem


if __name__ == "__main__":
    main()
