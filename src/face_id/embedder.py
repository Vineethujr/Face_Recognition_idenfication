"""Face embeddings with OpenCV SFace (FaceRecognizerSF)."""

from __future__ import annotations

import cv2
import numpy as np

from face_id.config import SFACE_PATH
from face_id.detector import FaceDetection
from face_id.models import ensure_models


class FaceEmbedder:
    def __init__(self, model_path=None) -> None:
        ensure_models()
        path = str(model_path or SFACE_PATH)
        self._recognizer = cv2.FaceRecognizerSF.create(path, "")

    def align_and_embed(
        self, image_bgr: np.ndarray, detection: FaceDetection
    ) -> np.ndarray:
        aligned = self._recognizer.alignCrop(image_bgr, detection.raw)
        features = self._recognizer.feature(aligned)
        embedding = np.asarray(features, dtype=np.float32).reshape(-1)
        return l2_normalize(embedding)


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)
