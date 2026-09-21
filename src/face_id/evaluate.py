"""Streamlined evaluation of core face identification & unknown rejection metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from face_id.config import COSINE_THRESHOLD, EMBEDDING_DIM, SCORE_MARGIN
from face_id.gallery import Gallery
from face_id.matcher import cosine_similarity, identify
from face_id.pipeline import FaceIdentificationSystem


def _unit_noise(rng: np.random.Generator, dim: int = EMBEDDING_DIM) -> np.ndarray:
    v = rng.normal(size=dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def run_synthetic_evaluation(
    n_identities: int = 10,
    templates_per_id: int = 2,
    probes_per_id: int = 2,
    n_unknown: int = 10,
    intra_noise: float = 0.10,
    threshold: float = COSINE_THRESHOLD,
    margin: float = SCORE_MARGIN,
    seed: int = 42,
) -> dict:
    """Streamlined benchmark evaluating essential identification and rejection metrics."""
    rng = np.random.Generator(np.random.PCG64(seed))
    centroids = [_unit_noise(rng) for _ in range(n_identities)]
    names = [f"person_{i+1:02d}" for i in range(n_identities)]

    # Build gallery with 2 templates per identity
    gallery: dict[str, list[np.ndarray]] = {}
    for name, center in zip(names, centroids):
        gallery[name] = [
            _add_noise(center, intra_noise, rng) for _ in range(templates_per_id)
        ]

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    correct_identifications = 0
    genuine_probes_total = 0
    genuine_rejected = 0
    strangers_rejected = 0

    # 1. Test Known Enrolled Faces
    for name, center in zip(names, centroids):
        for _ in range(probes_per_id):
            probe = _add_noise(center, intra_noise, rng)
            genuine_probes_total += 1

            # Compute score against own gallery templates
            best_genuine = max(cosine_similarity(probe, t) for t in gallery[name])
            genuine_scores.append(best_genuine)

            # Compute scores against others
            for other_name, other_templates in gallery.items():
                if other_name != name:
                    impostor_scores.append(
                        max(cosine_similarity(probe, t) for t in other_templates)
                    )

            decision = identify(probe, gallery, threshold=threshold, margin=margin)
            if decision.identity == name:
                correct_identifications += 1
            if decision.is_unknown:
                genuine_rejected += 1

    # 2. Test Unknown Strangers (Never Enrolled)
    for _ in range(n_unknown):
        stranger_probe = _unit_noise(rng)
        decision = identify(stranger_probe, gallery, threshold=threshold, margin=margin)
        if decision.is_unknown:
            strangers_rejected += 1

        for templates in gallery.values():
            impostor_scores.append(
                max(cosine_similarity(stranger_probe, t) for t in templates)
            )

    genuine_arr = np.asarray(genuine_scores)
    impostor_arr = np.asarray(impostor_scores)

    far = float(np.mean(impostor_arr >= threshold))
    frr = float(genuine_rejected / genuine_probes_total) if genuine_probes_total else 0.0

    return {
        "rank1_accuracy": round(correct_identifications / genuine_probes_total, 4),
        "unknown_rejection_rate": round(strangers_rejected / n_unknown, 4),
        "false_accept_rate": round(far, 4),
        "false_reject_rate": round(frr, 4),
        "mean_genuine_score": round(float(genuine_arr.mean()), 4),
        "mean_impostor_score": round(float(impostor_arr.mean()), 4),
        "threshold": threshold,
        "margin": margin,
        "enrolled_identities": n_identities,
        "probes_evaluated": genuine_probes_total + n_unknown,
    }


def _add_noise(center: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    noisy = center + noise_std * rng.normal(size=center.shape).astype(np.float32)
    return noisy / (np.linalg.norm(noisy) + 1e-12)


def run_folder_evaluation(
    dataset_root: Path,
    threshold: float = COSINE_THRESHOLD,
    margin: float = SCORE_MARGIN,
) -> dict:
    """Optional evaluation on real image folders: dataset/enrolled and dataset/probes."""
    enrolled_root = dataset_root / "enrolled"
    known_root = dataset_root / "probes" / "known"
    unknown_root = dataset_root / "probes" / "unknown"
    if not enrolled_root.is_dir():
        raise FileNotFoundError(f"Missing folder: {enrolled_root}")

    gallery = Gallery(root=dataset_root / "_eval_gallery")
    gallery.clear()

    system = FaceIdentificationSystem(
        gallery=gallery, threshold=threshold, margin=margin
    )

    for person_dir in sorted(p for p in enrolled_root.iterdir() if p.is_dir()):
        for img_path in _images(person_dir):
            try:
                system.enroll_image(person_dir.name, img_path)
            except Exception:
                pass

    correct = 0
    total_known = 0
    if known_root.is_dir():
        for person_dir in sorted(p for p in known_root.iterdir() if p.is_dir()):
            for img_path in _images(person_dir):
                total_known += 1
                res = system.identify_image(img_path)
                if res.decision.identity == person_dir.name:
                    correct += 1

    unknown_rejected = 0
    total_unknown = 0
    if unknown_root.is_dir():
        for img_path in _images(unknown_root):
            total_unknown += 1
            res = system.identify_image(img_path)
            if res.decision.is_unknown:
                unknown_rejected += 1

    return {
        "rank1_accuracy": (correct / total_known) if total_known else 0.0,
        "unknown_rejection_rate": (unknown_rejected / total_unknown) if total_unknown else 0.0,
        "known_probes": total_known,
        "unknown_probes": total_unknown,
    }


def _images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
