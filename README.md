# Face Recognition

InsightFace/ArcFace face recognition pipeline for RTSP cameras.

## Layout

- `recognize.py`: short camera runtime entrypoint, only selects `CONFIG_FILE`.
- `enroll_embedding.py`: short embedding enrollment entrypoint, only selects `CONFIG_FILE`.
- `runner.py`: shared entrypoint helpers for runtime scripts.
- `configs/default.yaml`: all runtime/model/video/enroll settings.
- `configs/congan.yaml`, `configs/truonghoc.yaml`: example dataset profiles.
- `src/face_recognition_app/`: reusable application code.
- `tools/enroll_embeddings.py`: CLI wrapper for building an embedding database.
- `tools/check_embeddings.py`: inspect database quality and enrolled people.
- `tools/quantize_detection_fp16.py`: convert detection ONNX to FP16.
- `tools/export_detection_trt.py`: export only the face detection ONNX to TensorRT engine with `trtexec`.
- `data/<profile>/face_data/`: enrollment images for each deployment profile.
- `data/<profile>/embeddings/<model_name>/`: generated embedding database and manifest.
- `models/arcface/`: repo-local InsightFace model packs, auto-downloaded here.
- `models/tensorrt/`: manually exported detection TensorRT engines.
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

## Config Profiles

Use one YAML file per deployment or dataset. Source code stays the same; only `CONFIG_FILE` changes.

Example:

```text
configs/
  congan.yaml
  truonghoc.yaml

data/
  congan/
    face_data/
    embeddings/
      buffalo_l/
        face_db.npz
        manifest.json

  truonghoc/
    face_data/
    embeddings/
      buffalo_l/
        face_db.npz
        manifest.json
```

In a profile config:

```yaml
paths:
  face_data_dir: data/congan/face_data
  database_path: data/congan/embeddings/buffalo_l/face_db.npz

models:
  insightface_name: buffalo_l
```

Then select that profile in `recognize.py` or `enroll_embedding.py`:

```python
CONFIG_FILE = "configs/congan.yaml"
```

## Prepare Face Data

Simple layout:

```text
data/congan/face_data/
  MinhDuc.jpg
  Alice.png
```

Folder layout is also supported:

```text
data/congan/face_data/
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
python3 enroll_embedding.py
```

Check the database:

```bash
cd face_recognition
python3 tools/check_embeddings.py --config configs/congan.yaml
```

The enrollment script saves the database to `paths.database_path` and writes a `manifest.json` next to it. `.npz`, `.h5`, and `.hdf5` database paths are supported.

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
- `video.device: rtx` for desktop NVIDIA GPUs using `nvh264dec`/`nvh265dec`.
- `video.device: jetson` for Jetson devices using `nvv4l2decoder`.
- `video.codec: h265` or `h264` to match the camera stream.

For RTX/desktop PC:

```yaml
video:
  device: rtx
  codec: h265
  decoder_priority: auto
```

For Jetson:

```yaml
video:
  device: jetson
  codec: h265
  decoder_priority: auto
```

`decoder_priority: auto` selects a GPU decoder for the selected device and falls back to CPU when configured. You can still force a backend manually with values such as `[nvidia_cuda, nvidia, cpu]` or `[jetson, cpu]`.

Then run:

```bash
cd face_recognition
python3 recognize.py
```

On the first run for a model pack, InsightFace downloads it into:

```text
models/arcface/<model_name>/
```

For example:

```text
models/arcface/buffalo_l/
models/arcface/buffalo_s/
models/arcface/antelopev2/
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
models/tensorrt/<model_name>/detection_640x640_fp16.engine
```

This requires TensorRT and `trtexec` in `PATH`. TensorRT engines are specific to GPU, driver, CUDA, and TensorRT versions. Rebuild the engine on each deployment machine. RTX 3090 supports FP16 TensorRT.

The runtime uses ONNX Runtime providers. When TensorRT runtime libraries are installed, it enables `TensorrtExecutionProvider` with repo-local engine caching under `artifacts/runtime/ort_trt_cache/<model_name>_detection_640x640_fp16/`; otherwise it falls back to CUDA and then CPU.

Install TensorRT runtime for ONNX Runtime 1.22:

```bash
conda activate .face
cd face_recognition
pip install -r requirements-tensorrt.txt
```

The pip TensorRT libraries live under `site-packages/tensorrt_libs`; ensure that directory is in `LD_LIBRARY_PATH` before launching Python. The local `.face` env already has activation scripts configured for this.
