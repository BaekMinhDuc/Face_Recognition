#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401

from face_recognition_app.config import load_config
from face_recognition_app.video import build_gstreamer_pipeline, open_video_capture

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RTSP decode FPS without face recognition.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--backend", default=None, help="Force backend: nvidia_cuda, nvidia, cpu, or ffmpeg.")
    parser.add_argument("--protocol", default="udp", help="GStreamer protocol when backend is forced.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    url = cfg["app"]["rtsp_url"]

    if args.backend and args.backend != "ffmpeg":
        pipeline = build_gstreamer_pipeline(url, cfg, args.protocol, args.backend)
        print(f"[INFO] Pipeline: {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        backend_name = f"gstreamer-{args.protocol}-{args.backend}"
    elif args.backend == "ffmpeg":
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        backend_name = "ffmpeg"
    else:
        cap, backend_name = open_video_capture(url, cfg)

    if cap is None or not cap.isOpened():
        raise SystemExit(f"Cannot open stream with backend={backend_name}")

    first_shape = None
    warmup_end = time.perf_counter() + args.warmup
    while time.perf_counter() < warmup_end:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        if first_shape is None:
            first_shape = frame.shape

    frames = 0
    start = time.perf_counter()
    end = start + args.seconds
    while time.perf_counter() < end:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        frames += 1
        if first_shape is None:
            first_shape = frame.shape
    elapsed = time.perf_counter() - start
    cap.release()

    print(f"[RESULT] backend={backend_name}")
    print(f"[RESULT] warmup={args.warmup:.2f}s")
    print(f"[RESULT] frames={frames}")
    print(f"[RESULT] elapsed={elapsed:.2f}s")
    print(f"[RESULT] fps={frames / elapsed:.2f}")
    print(f"[RESULT] frame_shape={first_shape}")


if __name__ == "__main__":
    main()
