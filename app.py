"""Beginner-friendly Sci-Fi Biometric Face ID Web Application.

Features:
- Web-only browser interface with Sci-Fi HUD aesthetics (built on uploaded templates).
- Real-time webcam frame snapshot and photo file upload.
- Face detection (YuNet), 128-D embeddings (SFace), cosine similarity matching.
- 4-stage Unknown rejection engine (below threshold, ambiguous margin, empty database, no face).
- Streamlined evaluation dashboard with essential metrics.
- Quick Quit button to stop the server instantly.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from face_id.config import COSINE_THRESHOLD, SCORE_MARGIN
from face_id.evaluate import run_synthetic_evaluation
from face_id.hud import BiometricHUD
from face_id.models import ensure_models, models_ready
from face_id.pipeline import FaceIdentificationSystem

st.set_page_config(
    page_title="Face ID Lab — Biometric Web HUD",
    page_icon="🤖",
    layout="wide",
)

# Sci-Fi HUD Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #030a16;
        color: #c9e4ff;
    }
    header {
        background-color: #030a16 !important;
    }
    h1, h2, h3, h4 {
        color: #00f0ff !important;
        font-family: 'Segoe UI', 'Consolas', sans-serif;
        letter-spacing: 1px;
    }
    .hud-box {
        background-color: rgba(6, 18, 38, 0.9);
        border: 1px solid #00b4d8;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 0 12px rgba(0, 180, 216, 0.18);
    }
    .hud-tag-green {
        background-color: #064e3b;
        border: 1px solid #10b981;
        color: #34d399;
        padding: 6px 12px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: bold;
        font-size: 14px;
    }
    .hud-tag-red {
        background-color: #4c111a;
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 6px 12px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: bold;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FRIENDLY_REASONS = {
    "matched": "Verified match: High-confidence similarity above security threshold.",
    "below_threshold": "Unknown rejected: Score is below the 0.363 cutoff. Face is not in database.",
    "ambiguous_margin": "Unknown rejected: Two enrolled people look almost equally similar. Rejection triggered for safety.",
    "empty_gallery": "Unknown rejected: Database is empty. Please enroll someone first.",
    "no_face": "Unknown rejected: No face detected. Please face the camera with good lighting.",
}


def to_bgr(image_file) -> np.ndarray:
    image = Image.open(image_file).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def annotate(
    image_bgr: np.ndarray,
    detections,
    label: str | None = None,
    score: float | None = None,
    is_unknown: bool = False,
    reason: str | None = None,
) -> np.ndarray:
    annotated = BiometricHUD.draw_hud(
        image_bgr,
        detections,
        label=label,
        score=score,
        is_unknown=is_unknown,
        reason=reason,
        show_mesh=True,
    )
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner="Loading YuNet & SFace AI models...")
def load_system() -> FaceIdentificationSystem:
    ensure_models()
    return FaceIdentificationSystem()


def photo_input(key: str):
    tab_cam, tab_file = st.tabs(["📸 Webcam (Current Frame)", "📁 Upload Image File"])
    with tab_cam:
        cam_photo = st.camera_input("Capture current frame from webcam", key=f"{key}_cam")
    with tab_file:
        file_photo = st.file_uploader(
            "Upload a JPG or PNG picture",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key=f"{key}_file",
        )
    return cam_photo or file_photo


def page_home():
    # Show HUD template banner if available
    template_path = Path("assets/media_1789982050295.jpg")
    if template_path.exists():
        st.image(str(template_path), use_container_width=True, caption="Biometric HUD System Template")

    st.title("🤖 Biometric Face ID Lab")
    st.subheader("Futuristic Face Recognition, Enrollment, and Unknown Rejection System")

    st.markdown(
        """
        <div class="hud-box">
            <h4>AI Biometric Architecture</h4>
            <ul>
                <li><b>Face Detection (YuNet)</b>: Fast edge CNN extracting face bounding box and 5 biometric landmarks (eyes, nose, mouth).</li>
                <li><b>Biometric Embedding (SFace)</b>: Aligns facial features and projects them into a <b>128-dimensional unit vector</b>.</li>
                <li><b>Similarity Matching</b>: Computes cosine similarity in 128-D hyperspace.</li>
                <li><b>Unknown Rejection Engine</b>: Enforces 4 security guardrails (<code>below_threshold</code>, <code>ambiguous_margin</code>, <code>empty_gallery</code>, <code>no_face</code>).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="hud-box">
                <b>Quick Start Steps</b><br>
                1. Open <b>Enroll</b> in the sidebar.<br>
                2. Enter a name and snap a photo or upload an image.<br>
                3. Open <b>Identify</b> to test recognition with Unknown rejection!
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="hud-box">
                <b>Default Security Thresholds</b><br>
                • <b>Cosine Cutoff (τ)</b>: <code>0.363</code><br>
                • <b>Ambiguity Margin (Δ)</b>: <code>0.050</code><br>
                • <b>Vector Space</b>: <code>128-D L2 Unit Sphere</code>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_enroll(system: FaceIdentificationSystem):
    st.title("📸 Enroll a Person")
    st.write("Register a new face into the database. You can capture a current frame via webcam or upload a picture.")

    col1, col2 = st.columns([1, 1])
    with col1:
        name = st.text_input("Subject Name", placeholder="e.g. Alice")
        photo = photo_input("enroll")

    with col2:
        if photo is not None:
            image_bgr = to_bgr(photo)
            detections = system.detector.detect(image_bgr)
            preview = annotate(image_bgr, detections)
            st.image(preview, caption=f"Biometric Mesh Preview ({len(detections)} face detected)", width=420)
            if not detections:
                st.warning("⚠️ No face detected. Please ensure good lighting and face the camera.")

    if st.button("💾 Save Face to Database", type="primary", disabled=photo is None):
        if not name.strip():
            st.error("Please enter a subject name first.")
            return
        try:
            count = system.enroll_array(name.strip(), to_bgr(photo), source=getattr(photo, "name", "webcam"))
            st.success(f"✅ Successfully enrolled **{name.strip()}**! Total stored templates: **{count}**.")
        except Exception as exc:
            st.error(f"Enrollment error: {exc}")


def page_identify(system: FaceIdentificationSystem):
    st.title("🔍 Identify a Face")
    st.write("Capture your current frame via webcam or upload a photo to identify against the enrolled database.")

    col1, col2 = st.columns([1, 1])
    with col1:
        photo = photo_input("identify")

    if photo is None:
        return

    image_bgr = to_bgr(photo)

    with col2:
        system.gallery.reload()
        result = system.identify_array(image_bgr)
        dec = result.decision
        matched = not dec.is_unknown and dec.identity is not None
        label = dec.identity if matched else "Unknown"

        preview = annotate(
            image_bgr,
            result.detections,
            label=label,
            score=dec.score,
            is_unknown=dec.is_unknown,
            reason=dec.reason,
        )
        st.image(preview, caption="Biometric HUD Scan Result", width=440)

        if matched:
            st.markdown(
                f"""
                <div class="hud-tag-green">
                    VERIFIED MATCH: {dec.identity.upper()} (Similarity: {dec.score:.4f})
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="hud-tag-red">
                    ALERT: UNKNOWN (Best Score: {dec.score:.4f} | Reason: {dec.reason})
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(FRIENDLY_REASONS.get(dec.reason, dec.reason))

        if dec.candidates:
            st.write("#### Top Candidates Leaderboard")
            st.dataframe(
                {
                    "Rank": [f"#{i+1}" for i in range(len(dec.candidates[:5]))],
                    "Identity": [c.identity for c in dec.candidates[:5]],
                    "Cosine Score": [f"{c.score:.4f}" for c in dec.candidates[:5]],
                },
                hide_index=True,
                use_container_width=True,
            )


def page_people(system: FaceIdentificationSystem):
    st.title("👥 Enrolled Database")
    system.gallery.reload()
    names = system.gallery.identities()

    if not names:
        st.warning("Database is empty. Go to **Enroll** to register faces.")
        return

    st.write(f"Total Enrolled Identities: **{len(names)}**")

    for name in names:
        col_name, col_btn = st.columns([4, 1])
        with col_name:
            st.markdown(f"**👤 {name}** — `{system.gallery.template_count(name)}` photo sample(s)")
        with col_btn:
            if st.button("Delete", key=f"del_{name}"):
                system.gallery.remove(name)
                st.rerun()

    st.markdown("---")
    if st.button("⚠️ Clear Entire Database", type="secondary"):
        system.gallery.clear()
        st.rerun()


def page_eval(threshold: float, margin: float):
    st.title("📊 Essential Evaluation Benchmark")
    st.write(
        "Runs an evaluation on the cosine similarity matcher and 4-stage unknown rejection engine "
        "evaluating the critical security metrics."
    )

    if st.button("▶ Run Essential Evaluation", type="primary"):
        report = run_synthetic_evaluation(threshold=threshold, margin=margin)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rank-1 Accuracy", f"{report['rank1_accuracy'] * 100:.1f}%")
        m2.metric("Unknown Rejection", f"{report['unknown_rejection_rate'] * 100:.1f}%")
        m3.metric("False Accept (FAR)", f"{report['false_accept_rate'] * 100:.2f}%")
        m4.metric("False Reject (FRR)", f"{report['false_reject_rate'] * 100:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="hud-box">
                    <b>Score Distributions</b><br>
                    • Genuine Match Average: <code>{report['mean_genuine_score']:.4f}</code><br>
                    • Impostor Stranger Average: <code>{report['mean_impostor_score']:.4f}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="hud-box">
                    <b>Benchmark Scope</b><br>
                    • Enrolled Identities: <code>{report['enrolled_identities']}</code><br>
                    • Probes Evaluated: <code>{report['probes_evaluated']}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.sidebar.title("🤖 Face ID Lab")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Enroll", "Identify", "People", "Evaluation"],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Security Thresholds")
    threshold = st.sidebar.slider(
        "Cosine Threshold (τ)",
        min_value=0.20,
        max_value=0.60,
        value=float(COSINE_THRESHOLD),
        step=0.001,
        help="Higher = stricter security. Faces with cosine score below this are rejected as Unknown.",
    )
    margin = st.sidebar.slider(
        "Ambiguity Margin (Δ)",
        min_value=0.00,
        max_value=0.20,
        value=float(SCORE_MARGIN),
        step=0.01,
        help="Rejects as Unknown if the top 2 candidates are closer than this margin.",
    )

    st.sidebar.markdown("---")

# Quick Quit is a local-only convenience. It kills the whole server
# process, which is dangerous on a shared cloud instance (it would
# take the app offline for every visitor, not just the person who
# clicked it) - so it only appears when FACE_ID_LOCAL=1 is set.
    IS_LOCAL = os.environ.get("FACE_ID_LOCAL") == "1"

    if IS_LOCAL:
        if st.sidebar.button(
        "⏹️ Quick Quit",
            type="secondary",
            help="Stop the application server immediately (local mode only)",
        ):
            st.sidebar.warning("Shutting down Face ID Lab...")
            time.sleep(0.4)
        os._exit(0)
    if page == "Home":
        page_home()
        return
    if page == "Evaluation":
        page_eval(threshold, margin)
        return

    if not models_ready():
        with st.spinner("Downloading AI models (YuNet & SFace)..."):
            ensure_models()

    system = load_system()
    system.threshold = threshold
    system.margin = margin

    if page == "Enroll":
        page_enroll(system)
    elif page == "Identify":
        page_identify(system)
    elif page == "People":
        page_people(system)


if __name__ == "__main__":
    main()
