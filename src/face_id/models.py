"""Download YuNet (detector) and SFace (embedder) ONNX weights."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from face_id.config import MODELS_DIR, SFACE_PATH, SFACE_URL, YUNET_PATH, YUNET_URL


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "face-id-lab/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
    if tmp.stat().st_size < 10_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download too small, likely failed: {url}")
    tmp.replace(dest)
    return dest


def models_ready() -> bool:
    return (
        YUNET_PATH.exists()
        and YUNET_PATH.stat().st_size > 0
        and SFACE_PATH.exists()
        and SFACE_PATH.stat().st_size > 0
    )


def ensure_models() -> tuple[Path, Path]:
    yunet = _download(YUNET_URL, YUNET_PATH)
    sface = _download(SFACE_URL, SFACE_PATH)
    return yunet, sface
