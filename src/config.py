"""Paths and constants shared by the training and inference scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

IMAGE_FEATURES_CSV = DATA_DIR / "image_features.csv"
AUDIO_FEATURES_CSV = DATA_DIR / "audio_features.csv"
MERGED_DATA_CSV = DATA_DIR / "merged_customer_data.csv"
MEMBER_CUSTOMER_MAP_CSV = DATA_DIR / "member_customer_map.csv"

# Final models produced by src/train_all.py
FACE_MODEL_PATH = MODELS_DIR / "face_recognition.joblib"
VOICE_MODEL_PATH = MODELS_DIR / "voiceprint_verification.joblib"
PRODUCT_MODEL_PATH = MODELS_DIR / "product_recommendation.joblib"

METRICS_CSV = OUTPUTS_DIR / "model_metrics.csv"

RANDOM_STATE = 42
