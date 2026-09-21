"""Cosine similarity matching and unknown rejection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from face_id.config import COSINE_THRESHOLD, SCORE_MARGIN


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class MatchCandidate:
    identity: str
    score: float
    template_index: int


@dataclass
class MatchDecision:
    identity: str | None
    score: float
    is_unknown: bool
    reason: str
    candidates: list[MatchCandidate] = field(default_factory=list)


def identify(
    probe: np.ndarray,
    gallery: dict[str, list[np.ndarray]],
    threshold: float = COSINE_THRESHOLD,
    margin: float = SCORE_MARGIN,
) -> MatchDecision:
    if not gallery:
        return MatchDecision(
            identity=None,
            score=0.0,
            is_unknown=True,
            reason="empty_gallery",
        )

    per_identity: list[MatchCandidate] = []
    for name, templates in gallery.items():
        best_score = -1.0
        best_idx = -1
        for i, template in enumerate(templates):
            score = cosine_similarity(probe, template)
            if score > best_score:
                best_score = score
                best_idx = i
        per_identity.append(
            MatchCandidate(identity=name, score=best_score, template_index=best_idx)
        )

    per_identity.sort(key=lambda c: c.score, reverse=True)
    best = per_identity[0]
    second = per_identity[1].score if len(per_identity) > 1 else -1.0

    if best.score < threshold:
        return MatchDecision(
            identity=None,
            score=best.score,
            is_unknown=True,
            reason="below_threshold",
            candidates=per_identity,
        )

    if len(per_identity) > 1 and (best.score - second) < margin:
        return MatchDecision(
            identity=None,
            score=best.score,
            is_unknown=True,
            reason="ambiguous_margin",
            candidates=per_identity,
        )

    return MatchDecision(
        identity=best.identity,
        score=best.score,
        is_unknown=False,
        reason="matched",
        candidates=per_identity,
    )
