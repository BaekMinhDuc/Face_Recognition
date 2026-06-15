# Face Recognition

InsightFace/ArcFace face recognition pipeline for RTSP cameras.

## Layout

- `recognize.py`: main camera runtime.
- `configs/default.yaml`: all runtime/model/video/enroll settings.
- `src/face_recognition_app/`: reusable application code.
- `tools/enroll_embeddings.py`: build an embedding database from `face_data/`.
- `tools/check_embeddings.py`: inspect database quality and enrolled people.
- `tools/quantize_detection_fp16.py`: convert detection ONNX to FP16.
- `tools/export_detection_trt.py`: export only the face detection ONNX to TensorRT engine with `trtexec`.
- `face_data/`: put enrollment images here.
- `db_embedding/`: generated embedding databases. Git-ignored except `.gitkeep`.
- `artifacts/models/arcface/`: repo-local InsightFace model packs, auto-downloaded here.
- `artifacts/models/tensorrt/`: manually exported detection TensorRT engines.
- `artifacts/runtime/ort_trt_cache/`: ONNX Runtime TensorRT cache, generated locally.

All runtime paths are relative to this `face_recognition/` folder. Models and embeddings are not written to `~/.insightface` or other machine cache folders.

## Install

```bash
conda create -n .face python=3.11 -y
conda activate .face
cd face_recognition
pip install -r requirements.txt
```

For GStreamer RTSP decode, use an OpenCV build with `GStreamer: YES`. The pip wheel is fine for FFMPEG fallback, but usually does not include GStreamer.

## Prepare Face Data

Simple layout:

```text
face_data/
  MinhDuc.jpg
  Alice.png
```

Folder layout is also supported:

```text
face_data/
  MinhDuc/
    img1.jpg
    img2.jpg
  Alice/
    001.png
```

For files directly under `face_data`, the filename stem is used as the person name. For subfolders, the first folder name is used.

Build the database:

```bash
conda activate .face
cd face_recognition
python3 tools/enroll_embeddings.py
```

Check the database:

```bash
cd face_recognition
python3 tools/check_embeddings.py
```

By default the database path is controlled by `paths.database_path` in `configs/default.yaml`.

## Run Recognition

For local runtime settings, copy the sample config and edit the local file:

```bash
cp configs/default.yaml configs/local.yaml
```

`configs/local.yaml` is ignored by Git. Put camera URLs, private IPs, and local database paths there.

Edit `configs/local.yaml`, especially:

- `app.rtsp_url`
- `paths.database_path`
- `models.insightface_name`
- `recognition.threshold`
- `sahi.enabled`
- `video.codec: h265` and `video.decoder_priority: [nvidia, cpu]` for NVIDIA H.265 GPU decode first.

Then run:

```bash
cd face_recognition
python3 recognize.py
```

On the first run for a model pack, InsightFace downloads it into:

```text
artifacts/models/arcface/models/<model_name>/
```

For example:

```text
artifacts/models/arcface/models/buffalo_l/
artifacts/models/arcface/models/buffalo_s/
artifacts/models/arcface/models/antelopev2/
```

## Detection Model Optimization

Quantize detection ONNX to FP16:

```bash
cd face_recognition
python3 tools/quantize_detection_fp16.py
```

Export detection ONNX to TensorRT engine:

```bash
cd face_recognition
python3 tools/export_detection_trt.py
```

Export a specific model pack:

```bash
python3 tools/export_detection_trt.py --model buffalo_l
python3 tools/export_detection_trt.py --model buffalo_s
python3 tools/export_detection_trt.py --model antelopev2
```

The export tool auto-downloads the model pack if it is missing, selects only `det*.onnx`, and writes the engine to:

```text
artifacts/models/tensorrt/<model_name>/detection_640x640_fp16.engine
```

This requires TensorRT and `trtexec` in `PATH`. TensorRT engines are specific to GPU, driver, CUDA, and TensorRT versions. Rebuild the engine on each deployment machine. RTX 3090 supports FP16 TensorRT.

The runtime uses ONNX Runtime providers. When TensorRT runtime libraries are installed, it enables `TensorrtExecutionProvider` with repo-local engine caching under `artifacts/runtime/ort_trt_cache`; otherwise it falls back to CUDA and then CPU.

Install TensorRT runtime for ONNX Runtime 1.22:

```bash
conda activate .face
cd face_recognition
pip install -r requirements-tensorrt.txt
```

The pip TensorRT libraries live under `site-packages/tensorrt_libs`; ensure that directory is in `LD_LIBRARY_PATH` before launching Python. The local `.face` env already has activation scripts configured for this.
