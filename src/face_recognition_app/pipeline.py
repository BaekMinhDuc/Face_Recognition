from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import time

import cv2
import numpy as np

from .config import color_tuple, resolve_path, tuple2
from .database import FaceDatabase
from .models import FaceEngine
from .video import open_video_capture


def run_recognition(cfg: dict) -> None:
    engine = FaceEngine(cfg)
    database = FaceDatabase.load(resolve_path(cfg["paths"]["database_path"]))
    threshold = float(cfg["recognition"].get("threshold", 0.4))

    rtsp_url = cfg["app"]["rtsp_url"]
    cap, backend = open_video_capture(rtsp_url, cfg)
    if cap is None:
        raise RuntimeError(f"Cannot open RTSP stream with GStreamer or FFMPEG: {rtsp_url}")
    print(f"[INFO] Video backend: {backend}")
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
    last_identities = []
    first_frame_logged = False
    embed_ms = 0.0
    recognition_future: Future | None = None
    recognition_executor = ThreadPoolExecutor(max_workers=1)

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

                can_submit = recognition_future is None or recognition_future.done()
                should_recognize = total_frames == 1 or total_frames % recognize_every_n_frames == 0
                if can_submit and should_recognize and det.kpss is not None and len(det.kpss) > 0:
                    recognition_future = recognition_executor.submit(
                        _embed_and_match,
                        engine,
                        database,
                        threshold,
                        frame.copy(),
                        det.kpss.copy(),
                    )

            if recognition_future is not None and recognition_future.done():
                try:
                    last_identities, embed_ms = recognition_future.result()
                except Exception as exc:
                    print(f"[WARNING] Recognition worker failed: {exc}")
                    last_identities = []
                recognition_future = None

            if last_bboxes is not None:
                identities = _identities_for_draw(last_bboxes, last_identities)
                _draw_results(frame, last_bboxes, identities, known_color, unknown_color, text_color)

            display = cv2.resize(frame, display_size) if display_size else frame
            cv2.imshow(title, display)
            if now - last_log_time >= log_interval:
                faces = 0 if last_bboxes is None else len(last_bboxes)
                print(
                    f"[INFO] fps={fps:.1f} faces={faces} "
                    f"detect_ms={detect_ms:.1f} embed_ms={embed_ms:.1f}"
                )
                last_log_time = now
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        recognition_executor.shutdown(wait=False, cancel_futures=True)
        cap.release()
        cv2.destroyAllWindows()


def _embed_and_match(
    engine: FaceEngine,
    database: FaceDatabase,
    threshold: float,
    frame: np.ndarray,
    kpss: np.ndarray,
) -> tuple[list[tuple[str, float]], float]:
    t0 = time.perf_counter()
    embeddings = engine.embed_faces(frame, kpss)
    identities = database.match(embeddings, threshold)
    embed_ms = (time.perf_counter() - t0) * 1000.0
    return identities, embed_ms


def _identities_for_draw(bboxes, identities: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if len(identities) == len(bboxes):
        return identities
    return [("Unknown", 0.0)] * len(bboxes)


def _draw_results(frame, bboxes, identities, known_color, unknown_color, text_color) -> None:
    h, w = frame.shape[:2]
    for bbox, (name, _score) in zip(bboxes, identities):
        x1, y1, x2, y2 = bbox[:4].astype(int)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        color = unknown_color if name == "Unknown" else known_color
        label = name
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_w, text_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        label_top = max(0, y1 - text_h - 10)
        cv2.rectangle(frame, (x1, label_top), (min(w - 1, x1 + text_w), y1), color, -1)
        cv2.putText(frame, label, (x1, max(text_h + 2, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
