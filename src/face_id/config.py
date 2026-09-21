from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
GALLERY_DIR = DATA_DIR / "gallery"

YUNET_URL = (
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
    "face_detection_yunet_2023mar.onnx?download=true"
)
SFACE_URL = (
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/"
    "face_recognition_sface_2021dec.onnx?download=true"
)
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# OpenCV SFace recommended cosine cutoff. Higher = more similar.
COSINE_THRESHOLD = 0.363
SCORE_MARGIN = 0.05
DET_SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000
EMBEDDING_DIM = 128
