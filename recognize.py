import _bootstrap  # noqa: F401

from face_recognition_app.runner import recognize


CONFIG_FILE = "configs/local.yaml"

recognize(CONFIG_FILE)
