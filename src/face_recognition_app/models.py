from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from insightface.model_zoo import model_zoo
from insightface.utils import face_align

from .config import model_store_root, tuple2
from .model_store import detection_onnx_path, ensure_model_pack_downloaded, recognition_onnx_path
from .providers import build_ort_provider_config


@dataclass
class DetectionResult:
    bboxes: np.ndarray
    kpss: np.ndarray | None


class FaceEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_cfg = cfg["models"]
        self.runtime_cfg = cfg["runtime"]
        self.sahi_cfg = cfg["sahi"]
        self.enroll_cfg = cfg["enroll"]
        self.det_size = tuple2(self.model_cfg["detection_size"], "models.detection_size")
        self.embedding_batch_size = int(cfg["recognition"].get("embedding_batch_size", 32))
        self.model_root = model_store_root(cfg)

        ensure_model_pack_downloaded(cfg)
        ctx_id = int(self.runtime_cfg.get("gpu_id", 0))
        det_use_trt = bool(self.runtime_cfg.get("use_tensorrt", False)) and bool(
            self.runtime_cfg.get("use_tensorrt_detection", True)
        )
        rec_use_trt = bool(self.runtime_cfg.get("use_tensorrt", False)) and bool(
            self.runtime_cfg.get("use_tensorrt_recognition", False)
        )

        det_providers, det_provider_options = build_ort_provider_config(cfg, use_tensorrt=det_use_trt)
        rec_providers, rec_provider_options = build_ort_provider_config(cfg, use_tensorrt=rec_use_trt)

        det_path = detection_onnx_path(cfg)
        rec_path = recognition_onnx_path(cfg)
        try:
            self.det_model = self._load_model(det_path, det_providers, det_provider_options, "detection")
        except Exception as exc:
            print(f"[WARNING] Detection TensorRT/CUDA init failed, fallback to CUDA/CPU: {exc}")
            det_providers, det_provider_options = build_ort_provider_config(cfg, use_tensorrt=False)
            self.det_model = self._load_model(det_path, det_providers, det_provider_options, "detection")

        self.rec_model = self._load_model(rec_path, rec_providers, rec_provider_options, "recognition")
        self.det_model.prepare(
            ctx_id=ctx_id,
            det_thresh=float(self.model_cfg.get("detection_threshold", 0.5)),
            input_size=self.det_size,
        )
        self.det_model.nms_thresh = float(self.model_cfg.get("detection_nms_threshold", 0.4))
        self.rec_model.prepare(ctx_id=ctx_id)
        self.app = SimpleNamespace(models={"detection": self.det_model, "recognition": self.rec_model})

    def _load_model(self, path: Path, providers: list[str], provider_options: list[dict[str, str]], taskname: str):
        model = model_zoo.get_model(str(path), providers=providers, provider_options=provider_options)
        if model is None or model.taskname != taskname:
            actual = None if model is None else model.taskname
            raise RuntimeError(f"Expected {taskname} model from {path}, got {actual}")
        return model

    def detect(self, frame: np.ndarray) -> DetectionResult:
        all_bboxes: list[np.ndarray] = []
        all_kps: list[np.ndarray] = []

        for x1, y1, x2, y2 in self._sahi_windows(frame.shape):
            crop = frame[y1:y2, x1:x2]
            bboxes, kpss = self.det_model.detect(
                crop,
                input_size=self.det_size,
                max_num=0,
            )
            if bboxes.shape[0] == 0:
                continue
            bboxes[:, [0, 2]] += x1
            bboxes[:, [1, 3]] += y1
            all_bboxes.append(bboxes)

            if kpss is not None:
                kpss[:, :, 0] += x1
                kpss[:, :, 1] += y1
                all_kps.append(kpss)

        if not all_bboxes:
            return DetectionResult(np.empty((0, 5), dtype=np.float32), None)

        bboxes = np.vstack(all_bboxes).astype(np.float32, copy=False)
        kpss = np.vstack(all_kps).astype(np.float32, copy=False) if all_kps else None
        keep = _nms_boxes(
            bboxes[:, :4],
            bboxes[:, 4],
            float(self.sahi_cfg.get("merge_nms_threshold", 0.45)),
        )
        bboxes = bboxes[keep]
        if kpss is not None:
            kpss = kpss[keep]

        max_faces = int(self.model_cfg.get("max_faces", 0))
        if max_faces > 0 and bboxes.shape[0] > max_faces:
            order = np.argsort(bboxes[:, 4])[::-1][:max_faces]
            bboxes = bboxes[order]
            if kpss is not None:
                kpss = kpss[order]

        return DetectionResult(bboxes, kpss)

    def embed_faces(self, frame: np.ndarray, kpss: np.ndarray | None) -> np.ndarray:
        if kpss is None or len(kpss) == 0:
            return np.empty((0, 512), dtype=np.float32)

        aligned_faces = [
            face_align.norm_crop(
                frame,
                landmark=kps.astype(np.float32),
                image_size=self.rec_model.input_size[0],
            )
            for kps in kpss
        ]

        outputs = []
        for start in range(0, len(aligned_faces), self.embedding_batch_size):
            batch = aligned_faces[start:start + self.embedding_batch_size]
            outputs.append(self.rec_model.get_feat(batch).astype(np.float32))

        embeddings = np.vstack(outputs)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-9)

    def extract_image_embedding(self, image_path: str | Path) -> tuple[np.ndarray | None, dict]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            return None, {"status": "failed", "reason": "cannot_read_image", "file": str(image_path)}

        det = self.detect(image)
        if det.bboxes.shape[0] == 0 or det.kpss is None:
            return None, {"status": "failed", "reason": "no_face", "file": str(image_path)}

        if det.bboxes.shape[0] > 1 and self.enroll_cfg.get("fail_on_multiple_faces", False):
            return None, {"status": "failed", "reason": "multiple_faces", "faces": int(det.bboxes.shape[0]), "file": str(image_path)}

        scores = det.bboxes[:, 4]
        min_score = float(self.enroll_cfg.get("min_detection_score", 0.5))
        valid = np.where(scores >= min_score)[0]
        if valid.size == 0:
            return None, {"status": "failed", "reason": "low_detection_score", "max_score": float(scores.max()), "file": str(image_path)}

        if self.enroll_cfg.get("choose_largest_face", True):
            boxes = det.bboxes[valid, :4]
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            chosen = valid[int(np.argmax(areas))]
        else:
            chosen = valid[int(np.argmax(scores[valid]))]

        embedding = self.embed_faces(image, det.kpss[[chosen]])[0]
        return embedding, {
            "status": "ok",
            "file": str(image_path),
            "faces": int(det.bboxes.shape[0]),
            "score": float(det.bboxes[chosen, 4]),
            "bbox": det.bboxes[chosen, :4].astype(float).tolist(),
        }

    def _sahi_windows(self, frame_shape: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
        h, w = frame_shape[:2]
        if not self.sahi_cfg.get("enabled", False):
            return [(0, 0, w, h)]

        slice_w, slice_h = tuple2(self.sahi_cfg["slice_size"], "sahi.slice_size")
        slice_w = min(slice_w, w)
        slice_h = min(slice_h, h)
        windows: list[tuple[int, int, int, int]] = []
        if self.sahi_cfg.get("include_full_frame", True):
            windows.append((0, 0, w, h))

        if w <= slice_w and h <= slice_h:
            return windows or [(0, 0, w, h)]

        overlap = float(self.sahi_cfg.get("overlap_ratio", 0.2))
        step_x = max(1, int(slice_w * (1.0 - overlap)))
        step_y = max(1, int(slice_h * (1.0 - overlap)))
        xs = list(range(0, max(w - slice_w, 0) + 1, step_x))
        ys = list(range(0, max(h - slice_h, 0) + 1, step_y))
        if xs[-1] != w - slice_w:
            xs.append(w - slice_w)
        if ys[-1] != h - slice_h:
            ys.append(h - slice_h)

        windows.extend((x, y, x + slice_w, y + slice_h) for y in ys for x in xs)
        max_windows = int(self.sahi_cfg.get("max_windows", 0))
        if max_windows > 0 and len(windows) > max_windows:
            windows = windows[:max_windows]
        return windows


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep
