from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .database import build_database
from .models import FaceEngine


@dataclass
class PersonEnrollStats:
    found: int = 0
    ok: int = 0
    failed: int = 0
    scores: list[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


def run_enrollment(
    cfg: dict[str, Any],
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    source_dir = resolve_path(data_dir or cfg["paths"]["face_data_dir"])
    database_path = resolve_path(output_path or cfg["paths"]["database_path"])
    if database_path.suffix.lower() not in {".npz", ".h5", ".hdf5"}:
        raise SystemExit(f"Unsupported database format: {database_path.suffix}. Use .npz, .h5, or .hdf5.")

    extensions = {ext.lower() for ext in cfg["enroll"]["image_extensions"]}
    image_paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not image_paths:
        raise SystemExit(f"No images found in {source_dir}. Expected extensions: {sorted(extensions)}")

    print(f"[INFO] Face data: {source_dir}")
    print(f"[INFO] Output database: {database_path}")
    print(f"[INFO] Model: {cfg['models']['insightface_name']}")
    print(f"[INFO] Images: {len(image_paths)}")

    engine = FaceEngine(cfg)
    samples: list[tuple[str, np.ndarray, str]] = []
    failures: list[tuple[str, Path, dict[str, Any]]] = []
    stats: defaultdict[str, PersonEnrollStats] = defaultdict(PersonEnrollStats)

    for path in image_paths:
        person_name = label_from_path(source_dir, path)
        stats[person_name].found += 1
        embedding, info = engine.extract_image_embedding(path)
        if embedding is None:
            failures.append((person_name, path, info))
            stats[person_name].failed += 1
            print(f"[SKIP] {person_name}: {path.name} reason={info['reason']}")
            continue

        score = float(info["score"])
        samples.append((person_name, embedding, str(path.relative_to(source_dir))))
        stats[person_name].ok += 1
        stats[person_name].scores.append(score)
        print(f"[OK] {person_name}: {path.name} score={score:.3f} faces={info['faces']}")

    if not samples:
        raise SystemExit("No valid face embeddings extracted.")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": cfg["models"]["insightface_name"],
        "source_dir": str(source_dir),
        "database_path": str(database_path),
        "failed_images": len(failures),
    }
    database = build_database(samples, metadata)
    database.save(database_path)
    save_manifest(database_path, database.names, stats, metadata, failures)
    print_summary(database_path, database.names, stats)


def label_from_path(data_dir: Path, image_path: Path) -> str:
    rel = image_path.relative_to(data_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return image_path.stem


def save_manifest(
    database_path: Path,
    names: list[str],
    stats: dict[str, PersonEnrollStats],
    metadata: dict[str, Any],
    failures: list[tuple[str, Path, dict[str, Any]]],
) -> None:
    manifest_path = database_path.with_name("manifest.json")
    manifest = {
        **metadata,
        "people": {
            name: {
                "images": stats[name].found,
                "embeddings": stats[name].ok,
                "failed": stats[name].failed,
                "mean_detection_score": round(stats[name].mean_score, 4),
            }
            for name in names
        },
        "failures": [
            {
                "person": person_name,
                "file": str(path),
                "reason": info.get("reason", "unknown"),
            }
            for person_name, path, info in failures
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(database_path: Path, names: list[str], stats: dict[str, PersonEnrollStats]) -> None:
    print("\nEnrollment summary")
    print(f"  database: {database_path}")
    print(f"  manifest: {database_path.with_name('manifest.json')}")
    print(f"  people: {len(names)}")
    print("\nPeople:")
    for index, name in enumerate(names, start=1):
        person = stats[name]
        print(
            f"  {index:02d}. {name}: "
            f"images={person.found} embeddings={person.ok} "
            f"failed={person.failed} mean_score={person.mean_score:.3f}"
        )
