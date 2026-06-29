import _bootstrap  # noqa: F401

from face_recognition_app.runner import enroll_embedding


CONFIG_FILE = "configs/local.yaml"

enroll_embedding(CONFIG_FILE)
