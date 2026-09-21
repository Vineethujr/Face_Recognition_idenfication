"""Face recognition identification: detect, embed, match, reject unknown."""

from face_id.config import COSINE_THRESHOLD, SCORE_MARGIN
from face_id.gallery import Gallery
from face_id.pipeline import FaceIdentificationSystem, IdentifyResult

__all__ = [
    "COSINE_THRESHOLD",
    "SCORE_MARGIN",
    "FaceIdentificationSystem",
    "IdentifyResult",
    "Gallery",
]
