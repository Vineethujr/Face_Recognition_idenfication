"""Essential core unit tests for Face ID Lab."""

import numpy as np

from face_id.detector import FaceDetection
from face_id.embedder import l2_normalize
from face_id.gallery import Gallery
from face_id.hud import BiometricHUD
from face_id.matcher import cosine_similarity, identify


def test_enroll_and_load(tmp_path):
    gallery = Gallery(root=tmp_path)
    emb = l2_normalize(np.ones(128, dtype=np.float32))
    assert gallery.enroll("alice", emb, source="cam.jpg") == 1
    assert gallery.identities() == ["alice"]

    reloaded = Gallery(root=tmp_path)
    assert reloaded.identities() == ["alice"]
    assert reloaded.template_count("alice") == 1


def test_known_face_match():
    probe = l2_normalize(np.arange(128, dtype=np.float32) + 1)
    other = l2_normalize(np.arange(128, dtype=np.float32)[::-1] + 1)
    gallery = {"alice": [probe], "bob": [other]}

    decision = identify(probe, gallery, threshold=0.5, margin=0.05)
    assert not decision.is_unknown
    assert decision.identity == "alice"
    assert decision.score > 0.99


def test_unknown_rejection():
    probe = l2_normalize(np.ones(128, dtype=np.float32))
    enrolled = l2_normalize(np.concatenate([np.ones(64), -np.ones(64)]))
    gallery = {"alice": [enrolled]}

    decision = identify(probe, gallery, threshold=0.5, margin=0.05)
    assert decision.is_unknown
    assert decision.reason == "below_threshold"


def test_ambiguous_margin():
    base = l2_normalize(np.arange(128, dtype=np.float32) + 5)
    close = l2_normalize(base + 0.01)
    gallery = {"alice": [base], "bob": [close]}

    decision = identify(base, gallery, threshold=0.2, margin=0.15)
    assert decision.is_unknown
    assert decision.reason == "ambiguous_margin"


def test_biometric_hud_overlay():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = FaceDetection(
        bbox=np.array([100, 100, 150, 180], dtype=np.float32),
        landmarks=np.array([[140, 160], [210, 160], [175, 200], [150, 240], [200, 240]], dtype=np.float32),
        score=0.98,
        raw=np.zeros(15, dtype=np.float32),
    )
    out = BiometricHUD.draw_hud(frame, [det], label="Alice", score=0.88, is_unknown=False)
    assert out.shape == frame.shape
    assert np.any(out > 0)
