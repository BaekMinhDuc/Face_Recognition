from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DB_VERSION = "1.0"


@dataclass
class FaceDatabase:
    names: list[str]
    avg_embeddings: np.ndarray
    sample_names: list[str]
    sample_embeddings: np.ndarray
    sample_files: list[str]
    metadata: dict[str, Any]

    @classmethod
    def empty(cls) -> "FaceDatabase":
        return cls([], np.empty((0, 512), dtype=np.float32), [], np.empty((0, 512), dtype=np.float32), [], {})

    @classmethod
    def load(cls, path: str | Path) -> "FaceDatabase":
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Embedding database not found: {path}")
        if path.suffix.lower() == ".npz":
            return cls._load_npz(path)
        if path.suffix.lower() in {".h5", ".hdf5"}:
            return cls._load_h5(path)
        raise ValueError(f"Unsupported database format: {path.suffix}")

    @classmethod
    def _load_npz(cls, path: Path) -> "FaceDatabase":
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
        names = [str(x) for x in data["names"].tolist()]
        avg_embeddings = _normalize_rows(data["avg_embeddings"].astype(np.float32))
        sample_names_raw = data["sample_names"] if "sample_names" in data else np.array([], dtype=str)
        sample_names = [str(x) for x in sample_names_raw.tolist()]
        sample_embeddings = data["sample_embeddings"] if "sample_embeddings" in data else np.empty((0, avg_embeddings.shape[1]), dtype=np.float32)
        sample_embeddings = _normalize_rows(sample_embeddings.astype(np.float32))
        sample_files_raw = data["sample_files"] if "sample_files" in data else np.array([], dtype=str)
        sample_files = [str(x) for x in sample_files_raw.tolist()]
        return cls(names, avg_embeddings, sample_names, sample_embeddings, sample_files, metadata)

    @classmethod
    def _load_h5(cls, path: Path) -> "FaceDatabase":
        names: list[str] = []
        avg_embeddings: list[np.ndarray] = []
        sample_names: list[str] = []
        sample_embeddings: list[np.ndarray] = []
        sample_files: list[str] = []
        metadata: dict[str, Any] = {"source_format": "h5"}

        with h5py.File(path, "r") as f:
            if "metadata" in f:
                metadata.update({k: _jsonable_attr(v) for k, v in f["metadata"].attrs.items()})

            for name in f["avg_embeddings"].keys():
                names.append(str(name))
                avg_embeddings.append(f["avg_embeddings"][name][()])

            if "embeddings" in f:
                for name in f["embeddings"].keys():
                    embeddings = np.asarray(f["embeddings"][name][()], dtype=np.float32)
                    embeddings = np.atleast_2d(embeddings)
                    sample_embeddings.extend(embeddings)
                    sample_names.extend([str(name)] * len(embeddings))
                    sample_files.extend([""] * len(embeddings))

        avg = _normalize_rows(np.asarray(avg_embeddings, dtype=np.float32))
        samples = _normalize_rows(np.asarray(sample_embeddings, dtype=np.float32)) if sample_embeddings else np.empty((0, avg.shape[1]), dtype=np.float32)
        return cls(names, avg, sample_names, samples, sample_files, metadata)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata.update({
            "version": DB_VERSION,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "embedding_dim": int(self.avg_embeddings.shape[1]) if self.avg_embeddings.size else 0,
            "num_people": len(self.names),
            "num_samples": len(self.sample_names),
        })
        np.savez_compressed(
            path,
            names=np.asarray(self.names, dtype=str),
            avg_embeddings=_normalize_rows(self.avg_embeddings.astype(np.float32)),
            sample_names=np.asarray(self.sample_names, dtype=str),
            sample_embeddings=_normalize_rows(self.sample_embeddings.astype(np.float32)),
            sample_files=np.asarray(self.sample_files, dtype=str),
            metadata=json.dumps(metadata, ensure_ascii=False),
        )

    def match(self, embeddings: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        if embeddings.shape[0] == 0:
            return []
        if len(self.names) == 0:
            return [("Unknown", -1.0)] * embeddings.shape[0]

        embeddings = _normalize_rows(embeddings.astype(np.float32))
        similarities = embeddings @ self.avg_embeddings.T
        best_indices = np.argmax(similarities, axis=1)
        best_scores = similarities[np.arange(similarities.shape[0]), best_indices]
        return [
            (self.names[int(idx)] if score >= threshold else "Unknown", float(score))
            for idx, score in zip(best_indices, best_scores)
        ]

    def per_person_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.names}
        for name in self.sample_names:
            counts[name] = counts.get(name, 0) + 1
        return counts


def build_database(samples: list[tuple[str, np.ndarray, str]], metadata: dict[str, Any]) -> FaceDatabase:
    grouped: dict[str, list[np.ndarray]] = {}
    sample_names: list[str] = []
    sample_embeddings: list[np.ndarray] = []
    sample_files: list[str] = []

    for name, embedding, source_file in samples:
        embedding = _normalize_rows(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        grouped.setdefault(name, []).append(embedding)
        sample_names.append(name)
        sample_embeddings.append(embedding)
        sample_files.append(source_file)

    names = sorted(grouped.keys())
    avg_embeddings = []
    for name in names:
        person_embeddings = np.vstack(grouped[name])
        avg = person_embeddings.mean(axis=0)
        avg_embeddings.append(avg)

    avg = _normalize_rows(np.asarray(avg_embeddings, dtype=np.float32)) if avg_embeddings else np.empty((0, 512), dtype=np.float32)
    samples_array = np.vstack(sample_embeddings).astype(np.float32) if sample_embeddings else np.empty((0, 512), dtype=np.float32)
    return FaceDatabase(names, avg, sample_names, _normalize_rows(samples_array), sample_files, metadata)


def embedding_quality(embedding: np.ndarray) -> dict[str, Any]:
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.size == 0:
        return {"quality": "Empty", "norm": 0.0, "dims": tuple(embedding.shape)}
    norm = float(np.linalg.norm(embedding))
    has_bad_values = bool(np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)))
    quality = "Good"
    if has_bad_values:
        quality = "Corrupted"
    elif norm < 0.9 or norm > 1.1:
        quality = "Suspicious"
    return {
        "quality": quality,
        "dims": tuple(embedding.shape),
        "norm": norm,
        "mean": float(np.mean(embedding)),
        "std": float(np.std(embedding)),
        "min": float(np.min(embedding)),
        "max": float(np.max(embedding)),
    }


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.reshape((0, 512)) if values.ndim < 2 else values
    values = np.atleast_2d(values)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / (norms + 1e-9)


def _jsonable_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if hasattr(value, "item"):
        return value.item()
    return value
