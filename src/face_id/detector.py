"""Face detection with OpenCV YuNet (FaceDetectorYN)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from face_id.config import (
    DET_SCORE_THRESHOLD,
    NMS_THRESHOLD,
    TOP_K,
    YUNET_PATH,
)
from face_id.models import ensure_models


@dataclass
class FaceDetection:
    bbox: np.ndarray
    landmarks: np.ndarray
    score: float
    raw: np.ndarray


class FaceDetector:
    def __init__(
        self,
        model_path=None,
        score_threshold: float = DET_SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        top_k: int = TOP_K,
    ) -> None:
        ensure_models()
        self.model_path = str(model_path or YUNET_PATH)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector = None
        self._input_size = (0, 0)

    def _get(self, width: int, height: int):
        size = (width, height)
        if self._detector is None or self._input_size != size:
            self._detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                size,
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )
            self._input_size = size
        return self._detector

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        if image_bgr.ndim != 3:
            raise ValueError("Expected a color image (H, W, 3)")
        h, w = image_bgr.shape[:2]
        detector = self._get(w, h)
        _retval, faces = detector.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return []
        results: list[FaceDetection] = []
        for row in faces:
            row = np.asarray(row, dtype=np.float32)
            results.append(
                FaceDetection(
                    bbox=row[0:4],
                    landmarks=row[4:14].reshape(5, 2),
                    score=float(row[14]),
                    raw=row,
                )
            )
        results.sort(key=lambda d: d.score, reverse=True)
        return results
