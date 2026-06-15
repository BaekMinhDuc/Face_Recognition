from __future__ import annotations

import cv2


def build_gstreamer_pipeline(rtsp_url: str, cfg: dict, protocol: str, decoder_backend: str = "nvidia") -> str:
    video_cfg = cfg["video"]
    codec = str(video_cfg.get("codec", "h265")).lower()
    if codec == "h264":
        depay = "rtph264depay"
        parser = "h264parse config-interval=-1"
    else:
        depay = "rtph265depay"
        parser = "h265parse config-interval=-1"
    decoder = _decoder_element(video_cfg, codec, decoder_backend)

    if protocol == "udp":
        source_opts = "protocols=udp udp-buffer-size=1048576 retry=5 timeout=5000000"
    else:
        source_opts = "protocols=tcp tcp-timeout=5000000 retry=5"
    if video_cfg.get("drop_on_latency", True):
        source_opts = f"{source_opts} drop-on-latency=true"

    post_decode = _post_decode_elements(video_cfg, decoder_backend)

    return (
        f"rtspsrc location={rtsp_url} latency={int(video_cfg.get('latency', 200))} {source_opts} "
        f"! {depay} ! queue max-size-buffers=0 max-size-bytes=0 max-size-time=0 leaky=downstream "
        f"! {parser} ! {decoder} ! queue max-size-buffers=1 leaky=downstream "
        f"! {post_decode} "
        f"appsink sync=false max-buffers={int(video_cfg.get('max_buffers', 3))} "
        f"drop={str(bool(video_cfg.get('drop', True))).lower()}"
    )


def open_video_capture(rtsp_url: str, cfg: dict) -> tuple[cv2.VideoCapture | None, str]:
    video_cfg = cfg["video"]
    if video_cfg.get("use_gstreamer", True):
        for decoder_backend in video_cfg.get("decoder_priority", ["nvidia", "cpu"]):
            for protocol in video_cfg.get("protocols", ["udp", "tcp"]):
                pipeline = build_gstreamer_pipeline(rtsp_url, cfg, str(protocol), str(decoder_backend))
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap, f"gstreamer-{protocol}-{decoder_backend}"
                cap.release()

    if video_cfg.get("fallback_ffmpeg", True):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            return cap, "ffmpeg"
        cap.release()

    return None, "unavailable"


def _decoder_element(video_cfg: dict, codec: str, decoder_backend: str) -> str:
    if decoder_backend in {"nvidia", "nvidia_cuda"}:
        decoders = video_cfg.get("nvidia_decoders", {})
        default = "nvh265dec max-display-delay=0" if codec == "h265" else "nvh264dec max-display-delay=0"
        return str(decoders.get(codec, default))

    decoders = video_cfg.get("cpu_decoders", {})
    default = "avdec_h265" if codec == "h265" else "avdec_h264"
    return str(decoders.get(codec, default))


def _post_decode_elements(video_cfg: dict, decoder_backend: str) -> str:
    resize = _resize_elements(video_cfg)
    if decoder_backend == "nvidia_cuda":
        return (
            "cudaconvert ! video/x-raw(memory:CUDAMemory), format=BGR ! "
            f"cudadownload ! video/x-raw, format=BGR ! {resize}"
        )
    return f"videoconvert ! video/x-raw, format=BGR ! {resize}"


def _resize_elements(video_cfg: dict) -> str:
    if video_cfg.get("decode_original_size", True):
        return ""

    size = video_cfg.get("decode_size")
    if not size:
        return ""
    width, height = int(size[0]), int(size[1])
    return f"videoscale ! video/x-raw, width={width}, height={height}, format=BGR ! "
