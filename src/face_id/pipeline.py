"""End-to-end enroll / identify pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from face_id.config import COSINE_THRESHOLD, SCORE_MARGIN
from face_id.detector import FaceDetection, FaceDetector
from face_id.embedder import FaceEmbedder
from face_id.gallery import Gallery
from face_id.matcher import MatchDecision, identify


@dataclass
class IdentifyResult:
    decision: MatchDecision
    detections: list[FaceDetection]
    embedding: np.ndarray | None


class FaceIdentificationSystem:
    def __init__(
        self,
        gallery: Gallery | None = None,
        threshold: float = COSINE_THRESHOLD,
        margin: float = SCORE_MARGIN,
    ) -> None:
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.gallery = gallery or Gallery()
        self.threshold = threshold
        self.margin = margin

    def load_image(self, path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def detections_and_embedding(
        self, image_bgr: np.ndarray, face_index: int = 0
    ) -> tuple[list[FaceDetection], np.ndarray | None]:
        detections = self.detector.detect(image_bgr)
        if not detections:
            return [], None
        if face_index >= len(detections):
            raise IndexError(
                f"face_index {face_index} out of range ({len(detections)} faces)"
            )
        embedding = self.embedder.align_and_embed(image_bgr, detections[face_index])
        return detections, embedding

    def enroll_image(
        self,
        identity: str,
        image_path: str | Path,
        face_index: int = 0,
    ) -> int:
        image = self.load_image(image_path)
        return self.enroll_array(identity, image, source=str(image_path), face_index=face_index)

    def enroll_array(
        self,
        identity: str,
        image_bgr: np.ndarray,
        source: str = "upload",
        face_index: int = 0,
    ) -> int:
        detections, embedding = self.detections_and_embedding(image_bgr, face_index)
        if embedding is None:
            raise ValueError("No face found. Try a clearer, front-facing photo.")
        return self.gallery.enroll(identity, embedding, source=source)

    def identify_image(
        self, image_path: str | Path, face_index: int = 0
    ) -> IdentifyResult:
        image = self.load_image(image_path)
        return self.identify_array(image, face_index=face_index)

    def identify_array(
        self, image_bgr: np.ndarray, face_index: int = 0
    ) -> IdentifyResult:
        detections, embedding = self.detections_and_embedding(image_bgr, face_index)
        if embedding is None:
            empty = MatchDecision(
                identity=None,
                score=0.0,
                is_unknown=True,
                reason="no_face",
            )
            return IdentifyResult(decision=empty, detections=[], embedding=None)
        decision = identify(
            embedding,
            self.gallery.as_dict(),
            threshold=self.threshold,
            margin=self.margin,
        )
        return IdentifyResult(
            decision=decision, detections=detections, embedding=embedding
        )
