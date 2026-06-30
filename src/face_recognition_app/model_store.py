from __future__ import annotations

from pathlib import Path

from .config import insightface_pack_dir, insightface_pack_zip, model_store_root


DETECTION_PATTERNS = (
    "det*.onnx",
    "scrfd*.onnx",
    "retinaface*.onnx",
    "blaze*.onnx",
    "yunet*.onnx",
)

RECOGNITION_PATTERNS = (
    "w600k*.onnx",
    "glintr*.onnx",
    "arcface*.onnx",
    "mobilefacenet*.onnx",
)

GENDER_AGE_PATTERNS = (
    "genderage*.onnx",
    "*gender*age*.onnx",
)


def normalize_insightface_pack(cfg: dict) -> None:
    model_dir = insightface_pack_dir(cfg)
    legacy_model_dir = model_store_root(cfg) / "models" / cfg["models"]["insightface_name"]
    if legacy_model_dir.exists() and not model_dir.exists():
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        legacy_model_dir.replace(model_dir)

    if not model_dir.exists():
        cleanup_model_archive(cfg)
        return

    if not list(model_dir.glob("*.onnx")):
        nested_dirs = [path for path in model_dir.iterdir() if path.is_dir() and list(path.glob("*.onnx"))]
        if len(nested_dirs) == 1:
            nested_dir = nested_dirs[0]
            for child in nested_dir.iterdir():
                target = model_dir / child.name
                if not target.exists():
                    child.replace(target)
            try:
                nested_dir.rmdir()
            except OSError:
                pass

    cleanup_model_archive(cfg)


def cleanup_model_archive(cfg: dict) -> None:
    archives = [
        insightface_pack_zip(cfg),
        model_store_root(cfg) / "models" / f"{cfg['models']['insightface_name']}.zip",
    ]
    for archive in archives:
        if archive.exists():
            archive.unlink()

    legacy_models_dir = model_store_root(cfg) / "models"
    try:
        legacy_models_dir.rmdir()
    except OSError:
        pass


def detection_onnx_path(cfg: dict) -> Path:
    normalize_insightface_pack(cfg)
    model_dir = insightface_pack_dir(cfg)
    candidates: list[Path] = []
    for pattern in DETECTION_PATTERNS:
        candidates.extend(model_dir.glob(pattern))

    candidates = sorted(
        {
            path
            for path in candidates
            if path.is_file() and "_fp16" not in path.stem and "_int8" not in path.stem
        }
    )
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find detection ONNX in {model_dir}. "
            f"Expected one of: {', '.join(DETECTION_PATTERNS)}"
        )
    return candidates[0]


def recognition_onnx_path(cfg: dict) -> Path:
    normalize_insightface_pack(cfg)
    model_dir = insightface_pack_dir(cfg)
    candidates: list[Path] = []
    for pattern in RECOGNITION_PATTERNS:
        candidates.extend(model_dir.glob(pattern))

    candidates = sorted(path for path in candidates if path.is_file())
    if not candidates:
        ignored_names = {"1k3d68.onnx", "2d106det.onnx", "genderage.onnx"}
        detection_path = None
        try:
            detection_path = detection_onnx_path(cfg).name
        except FileNotFoundError:
            pass
        candidates = sorted(
            path
            for path in model_dir.glob("*.onnx")
            if path.name not in ignored_names and path.name != detection_path
        )

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find recognition ONNX in {model_dir}. "
            f"Expected one of: {', '.join(RECOGNITION_PATTERNS)}"
        )
    return candidates[0]


def gender_age_onnx_path(cfg: dict) -> Path:
    normalize_insightface_pack(cfg)
    model_dir = insightface_pack_dir(cfg)
    candidates: list[Path] = []
    for pattern in GENDER_AGE_PATTERNS:
        candidates.extend(model_dir.glob(pattern))

    candidates = sorted(path for path in candidates if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find gender/age ONNX in {model_dir}. "
            f"Expected one of: {', '.join(GENDER_AGE_PATTERNS)}"
        )
    return candidates[0]


def ensure_model_pack_downloaded(
    cfg: dict,
    require_recognition: bool = True,
    require_gender_age: bool = False,
) -> None:
    try:
        detection_onnx_path(cfg)
        if require_recognition:
            recognition_onnx_path(cfg)
        if require_gender_age:
            gender_age_onnx_path(cfg)
        return
    except FileNotFoundError:
        pass

    from insightface.app import FaceAnalysis

    print(f"[INFO] Downloading InsightFace model pack to: {model_store_root(cfg)}")
    try:
        FaceAnalysis(
            name=cfg["models"]["insightface_name"],
            root=str(model_store_root(cfg)),
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
    except AssertionError:
        normalize_insightface_pack(cfg)
        detection_onnx_path(cfg)
        if require_recognition:
            recognition_onnx_path(cfg)
        if require_gender_age:
            gender_age_onnx_path(cfg)
    else:
        normalize_insightface_pack(cfg)
        detection_onnx_path(cfg)
        if require_recognition:
            recognition_onnx_path(cfg)
        if require_gender_age:
            gender_age_onnx_path(cfg)
