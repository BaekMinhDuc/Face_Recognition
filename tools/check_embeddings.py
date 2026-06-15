#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

import _bootstrap  # noqa: F401

from face_recognition_app.config import load_config, resolve_path
from face_recognition_app.database import FaceDatabase, embedding_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a face embedding database.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--database", default=None, help="Override database path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = resolve_path(args.database or cfg["paths"]["database_path"])
    db = FaceDatabase.load(db_path)

    print("=" * 80)
    print("FACE EMBEDDING DATABASE")
    print("=" * 80)
    print(f"File: {db_path}")
    print(f"People: {len(db.names)}")
    print(f"Samples: {len(db.sample_names)}")
    print(f"Metadata: {db.metadata}")

    counts = db.per_person_counts()
    sample_map = defaultdict(list)
    for name, emb in zip(db.sample_names, db.sample_embeddings):
        sample_map[name].append(emb)

    print("\nPeople:")
    for idx, name in enumerate(db.names, start=1):
        avg = db.avg_embeddings[idx - 1]
        quality = embedding_quality(avg)
        samples = np.asarray(sample_map.get(name, []), dtype=np.float32)
        intra = _intra_similarity(samples, avg) if len(samples) else None
        intra_text = f" | sample->avg mean={intra:.3f}" if intra is not None else ""
        print(
            f"  {idx:02d}. {name}: samples={counts.get(name, 0)} "
            f"| dim={quality['dims']} | norm={quality['norm']:.3f} "
            f"| quality={quality['quality']}{intra_text}"
        )


def _intra_similarity(samples: np.ndarray, avg: np.ndarray) -> float:
    samples = samples / (np.linalg.norm(samples, axis=1, keepdims=True) + 1e-9)
    avg = avg / (np.linalg.norm(avg) + 1e-9)
    return float(np.mean(samples @ avg))


if __name__ == "__main__":
    main()
