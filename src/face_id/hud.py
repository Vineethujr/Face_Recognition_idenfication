"""Biometric HUD overlay rendering inspired by futuristic sci-fi templates."""

from __future__ import annotations

import cv2
import numpy as np

from face_id.detector import FaceDetection


class BiometricHUD:
    """Draws sci-fi biometric HUD elements: corner brackets, facial mesh,

    telemetry readouts, and holographic scanning lines.
    """

    # Colors in BGR format
    CYAN = (255, 220, 0)        # Bright neon cyan (B, G, R) -> (255, 220, 0)
    YELLOW = (0, 235, 255)      # HUD target yellow
    GREEN = (80, 220, 20)       # Verified match green
    RED = (50, 50, 240)         # Alert/Unknown red
    WHITE = (245, 245, 245)
    DIM_CYAN = (180, 140, 20)   # Wireframe connector lines

    @staticmethod
    def draw_hud(
        image_bgr: np.ndarray,
        detections: list[FaceDetection],
        label: str | None = None,
        score: float | None = None,
        is_unknown: bool = False,
        reason: str | None = None,
        scan_progress: float | None = None,
        show_mesh: bool = True,
    ) -> np.ndarray:
        """Annotate an image with high-tech biometric HUD elements."""
        vis = image_bgr.copy()
        h_img, w_img = vis.shape[:2]

        if not detections:
            # Draw idle search reticle
            cv2.putText(
                vis,
                "[ SCANNING FOR TARGET... NO FACE DETECTED ]",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                BiometricHUD.CYAN,
                2,
                cv2.LINE_AA,
            )
            return vis

        for idx, det in enumerate(detections):
            x, y, w, h = [int(v) for v in det.bbox]
            # Ensure within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w_img - x, w)
            h = min(h_img - y, h)

            # Determine primary status color
            if label is not None:
                status_color = BiometricHUD.RED if is_unknown else BiometricHUD.GREEN
            else:
                status_color = BiometricHUD.CYAN

            # 1. Draw Sci-Fi Corner Brackets (Image 1 style)
            BiometricHUD._draw_corner_brackets(vis, x, y, w, h, BiometricHUD.YELLOW)

            # 2. Draw Facial Mesh Triangulation (Image 1 style)
            if show_mesh and det.landmarks is not None and len(det.landmarks) >= 5:
                BiometricHUD._draw_face_mesh(vis, det.landmarks, x, y, w, h)

            # 3. Draw Scanning Sweep Line if active
            if scan_progress is not None:
                sweep_y = int(y + (scan_progress % 1.0) * h)
                if y <= sweep_y <= y + h:
                    overlay = vis.copy()
                    cv2.line(overlay, (x, sweep_y), (x + w, sweep_y), BiometricHUD.CYAN, 2, cv2.LINE_AA)
                    cv2.rectangle(overlay, (x, max(y, sweep_y - 8)), (x + w, min(y + h, sweep_y + 8)), (255, 180, 0), -1)
                    cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

            # 4. Telemetry overlay box
            BiometricHUD._draw_telemetry(
                vis,
                x,
                y,
                w,
                h,
                det_score=det.score,
                label=label if idx == 0 else None,
                match_score=score if idx == 0 else None,
                is_unknown=is_unknown if idx == 0 else False,
                reason=reason if idx == 0 else None,
                status_color=status_color,
            )

        return vis

    @staticmethod
    def _draw_corner_brackets(img: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int]):
        """Draw sci-fi target bracket corners around the bounding box."""
        # Pad box slightly
        pad = int(min(w, h) * 0.08)
        bx = max(0, x - pad)
        by = max(0, y - pad)
        bw = min(img.shape[1] - bx, w + 2 * pad)
        bh = min(img.shape[0] - by, h + 2 * pad)

        corner_len = max(12, int(min(bw, bh) * 0.22))
        thickness = 2

        # Top-Left
        cv2.line(img, (bx, by), (bx + corner_len, by), color, thickness, cv2.LINE_AA)
        cv2.line(img, (bx, by), (bx, by + corner_len), color, thickness, cv2.LINE_AA)

        # Top-Right
        cv2.line(img, (bx + bw, by), (bx + bw - corner_len, by), color, thickness, cv2.LINE_AA)
        cv2.line(img, (bx + bw, by), (bx + bw, by + corner_len), color, thickness, cv2.LINE_AA)

        # Bottom-Left
        cv2.line(img, (bx, by + bh), (bx + corner_len, by + bh), color, thickness, cv2.LINE_AA)
        cv2.line(img, (bx, by + bh), (bx, by + bh - corner_len), color, thickness, cv2.LINE_AA)

        # Bottom-Right
        cv2.line(img, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), color, thickness, cv2.LINE_AA)
        cv2.line(img, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_face_mesh(img: np.ndarray, landmarks: np.ndarray, x: int, y: int, w: int, h: int):
        """Synthesizes an anatomical triangulation mesh around the facial keypoints

        resembling Image 1 (forehead, temples, eyes, nose bridge, mouth, chin).
        """
        re = landmarks[0]  # Right eye (subject's right)
        le = landmarks[1]  # Left eye
        nt = landmarks[2]  # Nose tip
        rm = landmarks[3]  # Right mouth corner
        lm = landmarks[4]  # Left mouth corner

        eye_mid = (re + le) / 2.0
        eye_dist = np.linalg.norm(re - le)
        mouth_mid = (rm + lm) / 2.0

        # Synthesize anatomical mesh points
        forehead_mid = eye_mid - np.array([0, eye_dist * 0.75])
        temple_r = re + np.array([-eye_dist * 0.55, -eye_dist * 0.35])
        temple_l = le + np.array([eye_dist * 0.55, -eye_dist * 0.35])
        forehead_r = (forehead_mid + temple_r) / 2.0 + np.array([0, -eye_dist * 0.1])
        forehead_l = (forehead_mid + temple_l) / 2.0 + np.array([0, -eye_dist * 0.1])

        nose_bridge = eye_mid + (nt - eye_mid) * 0.4
        cheek_r = re + (rm - re) * 0.55 + np.array([-eye_dist * 0.45, 0])
        cheek_l = le + (lm - le) * 0.55 + np.array([eye_dist * 0.45, 0])

        chin = mouth_mid + np.array([0, eye_dist * 0.85])
        jaw_r = rm + (chin - rm) * 0.5 + np.array([-eye_dist * 0.4, 0])
        jaw_l = lm + (chin - lm) * 0.5 + np.array([eye_dist * 0.4, 0])

        all_points = [
            forehead_mid, forehead_r, forehead_l, temple_r, temple_l,
            re, le, nose_bridge, nt, cheek_r, cheek_l,
            rm, lm, mouth_mid, jaw_r, jaw_l, chin
        ]

        # Mesh edges connecting vertices (Delaunay-style facial wireframe)
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (0, 7), (1, 5), (2, 6), (3, 5), (4, 6),
            (3, 9), (4, 10), (5, 7), (6, 7),
            (5, 8), (6, 8), (7, 8), (5, 9), (6, 10),
            (9, 11), (10, 12), (8, 11), (8, 12), (8, 13),
            (11, 13), (12, 13), (11, 14), (12, 15),
            (9, 14), (10, 15), (14, 16), (15, 16), (13, 16),
            (5, 6), (11, 12)
        ]

        # Draw wireframe lines
        for i1, i2 in edges:
            p1 = (int(all_points[i1][0]), int(all_points[i1][1]))
            p2 = (int(all_points[i2][0]), int(all_points[i2][1]))
            cv2.line(img, p1, p2, BiometricHUD.DIM_CYAN, 1, cv2.LINE_AA)

        # Draw glowing landmark nodes
        for pt in all_points:
            px, py = int(pt[0]), int(pt[1])
            # Outer white ring
            cv2.circle(img, (px, py), 3, BiometricHUD.WHITE, -1, cv2.LINE_AA)
            # Inner cyan core
            cv2.circle(img, (px, py), 1, BiometricHUD.CYAN, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_telemetry(
        img: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        det_score: float,
        label: str | None,
        match_score: float | None,
        is_unknown: bool,
        reason: str | None,
        status_color: tuple[int, int, int],
    ):
        """Draw tech readout labels and status banner beside the face."""
        # Top label
        det_text = f"DET: {det_score * 100:.1f}% | 128-D EMBED"
        cv2.putText(
            img,
            det_text,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            BiometricHUD.YELLOW,
            1,
            cv2.LINE_AA,
        )

        # Bottom identification status tag
        if label is not None:
            if is_unknown:
                banner = f"UNKNOWN ({reason or 'no match'})"
                tag_col = BiometricHUD.RED
            else:
                score_str = f"{match_score:.3f}" if match_score is not None else ""
                banner = f"ID: {label.upper()} [{score_str}]"
                tag_col = BiometricHUD.GREEN

            by = min(img.shape[0] - 8, y + h + 24)
            # Draw semi-transparent tag pill
            (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x, by - th - 6), (x + tw + 12, by + 4), (10, 15, 25), -1)
            cv2.rectangle(img, (x, by - th - 6), (x + tw + 12, by + 4), tag_col, 1)
            cv2.putText(
                img,
                banner,
                (x + 6, by - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                tag_col,
                2,
                cv2.LINE_AA,
            )
