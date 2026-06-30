from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time

import cv2
import numpy as np

from .config import color_tuple, resolve_path, tuple2
from .database import FaceDatabase
from .models import FaceEngine
from .video import open_video_capture


@dataclass
class FaceAnnotation:
    name: str = ""
    score: float = 0.0
    gender: int | None = None
    age: int | None = None


def run_recognition(cfg: dict) -> None:
    features = cfg.get("features", {})
    detection_enabled = bool(features.get("detection", True))
    recognition_enabled = bool(features.get("recognition", True))
    gender_age_enabled = bool(features.get("gender_age", False))
    if not detection_enabled:
        raise ValueError("features.detection must be true for camera pipeline.")

    engine = FaceEngine(cfg)
    database = FaceDatabase.load(resolve_path(cfg["paths"]["database_path"])) if recognition_enabled else None
    threshold = float(cfg["recognition"].get("threshold", 0.4))

    rtsp_url = cfg["app"]["rtsp_url"]
    cap, backend = open_video_capture(rtsp_url, cfg)
    if cap is None:
        raise RuntimeError(f"Cannot open RTSP stream with GStreamer or FFMPEG: {rtsp_url}")
    print(f"[INFO] Video backend: {backend}")
    print(f"[INFO] Features: detection={detection_enabled} recognition={recognition_enabled} gender_age={gender_age_enabled}")
    if database is not None:
        print(f"[INFO] Loaded database: {len(database.names)} people, {len(database.sample_names)} samples")

    title = cfg["app"].get("title", "Face Recognition")
    display_size = cfg["app"].get("display_size")
    display_size = tuple2(display_size, "app.display_size") if display_size else None
    vis = cfg["visualization"]
    known_color = color_tuple(vis["known_color"], "visualization.known_color")
    unknown_color = color_tuple(vis["unknown_color"], "visualization.unknown_color")
    text_color = color_tuple(vis["text_color"], "visualization.text_color")
    processing_cfg = cfg.get("processing", {})
    detect_every_n_frames = max(1, int(processing_cfg.get("detect_every_n_frames", 1)))
    recognize_every_n_frames = max(1, int(processing_cfg.get("recognize_every_n_frames", 1)))
    log_interval = max(0.5, float(processing_cfg.get("log_interval_sec", 2.0)))

    frame_count = 0
    total_frames = 0
    fps = 0.0
    fps_update_time = time.time()
    last_log_time = fps_update_time
    last_bboxes = None
    last_annotations: list[FaceAnnotation] = []
    first_frame_logged = False
    analysis_ms = 0.0
    analysis_future: Future | None = None
    needs_face_analysis = recognition_enabled or gender_age_enabled
    analysis_executor = ThreadPoolExecutor(max_workers=1) if needs_face_analysis else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            frame_count += 1
            total_frames += 1
            now = time.time()
            if now - fps_update_time >= 1.0:
                fps = frame_count / (now - fps_update_time)
                frame_count = 0
                fps_update_time = now

            if not first_frame_logged:
                print(f"[INFO] First frame shape: {frame.shape[1]}x{frame.shape[0]}")
                first_frame_logged = True

            run_inference = total_frames == 1 or total_frames % detect_every_n_frames == 0
            detect_ms = 0.0
            if run_inference:
                t0 = time.perf_counter()
                det = engine.detect(frame)
                t1 = time.perf_counter()
                last_bboxes = det.bboxes
                detect_ms = (t1 - t0) * 1000.0

                can_submit = analysis_future is None or analysis_future.done()
                should_analyze = total_frames == 1 or total_frames % recognize_every_n_frames == 0
                has_faces = det.bboxes is not None and len(det.bboxes) > 0
                has_kps = det.kpss is not None and len(det.kpss) > 0
                can_recognize = not recognition_enabled or has_kps
                if needs_face_analysis and can_submit and should_analyze and has_faces and can_recognize:
                    kpss = None if det.kpss is None else det.kpss.copy()
                    analysis_future = analysis_executor.submit(
                        _analyze_faces,
                        engine,
                        database,
                        threshold,
                        recognition_enabled,
                        gender_age_enabled,
                        frame.copy(),
                        det.bboxes.copy(),
                        kpss,
                    )

            if analysis_future is not None and analysis_future.done():
                try:
                    last_annotations, analysis_ms = analysis_future.result()
                except Exception as exc:
                    print(f"[WARNING] Face analysis worker failed: {exc}")
                    last_annotations = []
                analysis_future = None

            if last_bboxes is not None:
                annotations = _annotations_for_draw(last_bboxes, last_annotations)
                _draw_results(frame, last_bboxes, annotations, known_color, unknown_color, text_color)

            display = cv2.resize(frame, display_size) if display_size else frame
            cv2.imshow(title, display)
            if now - last_log_time >= log_interval:
                faces = 0 if last_bboxes is None else len(last_bboxes)
                print(
                    f"[INFO] fps={fps:.1f} faces={faces} "
                    f"detect_ms={detect_ms:.1f} analysis_ms={analysis_ms:.1f}"
                )
                last_log_time = now
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if analysis_executor is not None:
            analysis_executor.shutdown(wait=False, cancel_futures=True)
        cap.release()
        cv2.destroyAllWindows()


def _analyze_faces(
    engine: FaceEngine,
    database: FaceDatabase | None,
    threshold: float,
    recognition_enabled: bool,
    gender_age_enabled: bool,
    frame: np.ndarray,
    bboxes: np.ndarray,
    kpss: np.ndarray | None,
) -> tuple[list[FaceAnnotation], float]:
    t0 = time.perf_counter()
    annotations = [FaceAnnotation() for _ in range(len(bboxes))]

    if recognition_enabled:
        if database is None:
            raise RuntimeError("Recognition is enabled but embedding database is not loaded.")
        embeddings = engine.embed_faces(frame, kpss)
        identities = database.match(embeddings, threshold)
        for annotation, (name, score) in zip(annotations, identities):
            annotation.name = name
            annotation.score = score

    if gender_age_enabled:
        gender_age = engine.gender_age_faces(frame, bboxes)
        for annotation, (gender, age) in zip(annotations, gender_age):
            annotation.gender = gender
            annotation.age = age

    analysis_ms = (time.perf_counter() - t0) * 1000.0
    return annotations, analysis_ms


def _annotations_for_draw(bboxes, annotations: list[FaceAnnotation]) -> list[FaceAnnotation]:
    if len(annotations) == len(bboxes):
        return annotations
    return [FaceAnnotation() for _ in range(len(bboxes))]


def _draw_results(frame, bboxes, annotations, known_color, unknown_color, text_color) -> None:
    h, w = frame.shape[:2]
    for bbox, annotation in zip(bboxes, annotations):
        x1, y1, x2, y2 = bbox[:4].astype(int)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        label = _annotation_label(annotation)
        color = unknown_color if annotation.name == "Unknown" else known_color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if not label:
            continue
        text_w, text_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        label_top = max(0, y1 - text_h - 10)
        cv2.rectangle(frame, (x1, label_top), (min(w - 1, x1 + text_w), y1), color, -1)
        cv2.putText(frame, label, (x1, max(text_h + 2, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)


def _annotation_label(annotation: FaceAnnotation) -> str:
    parts = []
    if annotation.name:
        parts.append(annotation.name)
    if annotation.gender is not None and annotation.age is not None:
        gender = "Male" if annotation.gender == 1 else "Female"
        parts.append(f"{gender} {annotation.age}")
    return " | ".join(parts)
