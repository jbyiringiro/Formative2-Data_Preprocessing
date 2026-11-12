"""Feature extraction for new face images, voice samples and customer records.

The image and audio functions reproduce the feature engineering of notebooks 02 and 03
exactly, so a sample captured at prediction time is described by the same features
the models were trained on (data/image_features.csv and data/audio_features.csv).
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# --------------------------------------------------------------------------- images
IMAGE_SIZE = (128, 128)
HISTOGRAM_BINS = 32
IMAGE_FEATURE_COLUMNS = [f"feat_{i}" for i in range(3 * HISTOGRAM_BINS)]


def load_image(path):
    """Read an image as a 128x128 BGR array (same as cv2.imread + cv2.resize in notebook 02)."""
    data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII Windows paths
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.resize(image, IMAGE_SIZE)


def histogram_features(image):
    """Normalised 32-bin histogram for each of the blue, green and red channels (96 values)."""
    features = []
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], None, [HISTOGRAM_BINS], [0, 256]).flatten()
        features.extend(hist / hist.sum())
    return features


def extract_image_features(path):
    """Return a one-row DataFrame of image features for the image at `path`."""
    return pd.DataFrame([histogram_features(load_image(path))], columns=IMAGE_FEATURE_COLUMNS)


# --------------------------------------------------------------------------- audio
N_MFCC = 13
AUDIO_FEATURE_COLUMNS = (
    [f"mfcc_{i}_{stat}" for i in range(1, N_MFCC + 1) for stat in ("mean", "std")]
    + [
        f"{name}_{stat}"
        for name in ("rolloff", "rms_energy", "spectral_centroid", "spectral_bandwidth", "zero_crossing_rate")
        for stat in ("mean", "std")
    ]
)


def _ffmpeg_executable():
    """FFmpeg from the PATH, or the copy bundled with the imageio-ffmpeg package."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("Decoding this audio format needs FFmpeg: install it or `pip install imageio-ffmpeg`.") from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio(path):
    """Load an audio file as a mono float32 signal at its native sample rate.

    WAV/FLAC/OGG files are read directly. Compressed formats such as .m4a are decoded
    to 16-bit PCM with FFmpeg first, which is what librosa did in Colab when the
    training features were extracted. Channels are averaged to mono, as librosa does.
    """
    path = Path(path)
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        signal, sr = sf.read(path, dtype="float32", always_2d=True)
        return signal.mean(axis=1), sr

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "decoded.wav"
        result = subprocess.run(
            [_ffmpeg_executable(), "-v", "error", "-y", "-i", str(path), "-acodec", "pcm_s16le", str(wav_path)],
            capture_output=True,
            text=True,
        )
        # Some iPhone voice memos end with an empty ALAC frame, which FFmpeg reports
        # but decodes the full recording anyway; only fail if no audio was produced.
        if result.returncode != 0 or not wav_path.exists():
            raise RuntimeError(f"FFmpeg could not decode {path}:\n{result.stderr}")
        signal, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    return signal.mean(axis=1), sr


def audio_features(signal, sr):
    """36 acoustic features: mean/std of 13 MFCCs, roll-off, RMS energy, centroid, bandwidth, ZCR."""
    features = {}
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i + 1}_std"] = np.std(mfcc[i])

    frames = {
        "rolloff": librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=0.85),
        "rms_energy": librosa.feature.rms(y=signal),
        "spectral_centroid": librosa.feature.spectral_centroid(y=signal, sr=sr),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=signal, sr=sr),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(y=signal),
    }
    for name, values in frames.items():
        features[f"{name}_mean"] = np.mean(values)
        features[f"{name}_std"] = np.std(values)
    return features


def extract_audio_features(path):
    """Return a one-row DataFrame of audio features for the recording at `path`."""
    signal, sr = load_audio(path)
    return pd.DataFrame([audio_features(signal, sr)], columns=AUDIO_FEATURE_COLUMNS)


# --------------------------------------------------------------------------- customers
PRODUCT_FEATURE_COLUMNS = [
    "avg_engagement_score",
    "avg_purchase_interest_score",
    "social_record_count",
    "platform_count",
    "sentiment_Negative",
    "sentiment_Neutral",
    "sentiment_Positive",
    "has_social_profile",
]
_COUNT_COLUMNS = ["social_record_count", "platform_count", "sentiment_Negative", "sentiment_Neutral", "sentiment_Positive"]


def product_features(merged):
    """Customer features used by the product model (same selection as notebook 01).

    Customers without a social profile have zero social records, platforms and sentiments;
    their average scores stay missing and are median-imputed inside the model pipeline.
    """
    X = merged[PRODUCT_FEATURE_COLUMNS].copy()
    X.loc[X["has_social_profile"] == 0, _COUNT_COLUMNS] = 0
    return X
