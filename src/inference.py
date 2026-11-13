"""Load the trained models and run the multimodal authentication flow.

Decision logic (assignment flow):
    1. Face image  -> face model. Access continues only if the face is recognised as a
                      team member with confidence >= the face threshold.
    2. Product     -> the recognised member's customer profile is sent to the product
                      model; the prediction is kept back until the voice check passes.
    3. Voice       -> voiceprint model. The speaker must be the *same* member as the face,
                      with confidence >= the voice threshold.
    4. Only then is the predicted product category displayed; any failed check ends in
       ACCESS DENIED.

Used by the command-line demo; also usable on its own:
    from src.inference import load_artifacts, run_transaction
    result = run_transaction("data/images/Josue/Josue_neutral.jpg", "data/audio/Josue/Josue_recording1.m4a")
"""
import joblib
import numpy as np
import pandas as pd

from src.config import FACE_MODEL_PATH, MEMBER_CUSTOMER_MAP_CSV, MERGED_DATA_CSV, PRODUCT_MODEL_PATH, VOICE_MODEL_PATH
from src.features import extract_audio_features, extract_image_features, product_features


def load_artifacts():
    """Load the three saved models (each stored with its classes, feature list and threshold)."""
    return {
        "face": joblib.load(FACE_MODEL_PATH),
        "voice": joblib.load(VOICE_MODEL_PATH),
        "product": joblib.load(PRODUCT_MODEL_PATH),
    }


def _predict(artifact, features):
    proba = artifact["model"].predict_proba(features[artifact["feature_columns"]])[0]
    best = int(np.argmax(proba))
    probabilities = {name: float(p) for name, p in zip(artifact["classes"], proba)}
    return artifact["classes"][best], float(proba[best]), probabilities


def identify_face(image_path, artifacts):
    """Who is in the image, and is the match confident enough to grant access?"""
    face = artifacts["face"]
    member, confidence, probabilities = _predict(face, extract_image_features(image_path))
    return {
        "member": member,
        "confidence": confidence,
        "threshold": face["threshold"],
        "accepted": confidence >= face["threshold"],
        "probabilities": probabilities,
    }


def verify_voice(audio_path, claimed_member, artifacts):
    """Does the voice belong to `claimed_member` (the person recognised by the face model)?"""
    voice = artifacts["voice"]
    speaker, confidence, probabilities = _predict(voice, extract_audio_features(audio_path))
    return {
        "speaker": speaker,
        "claimed_member": claimed_member,
        "confidence": confidence,
        "threshold": voice["threshold"],
        "accepted": speaker == claimed_member and confidence >= voice["threshold"],
        "probabilities": probabilities,
    }


def customer_for_member(member):
    """Customer record linked to a team member (data/member_customer_map.csv)."""
    mapping = pd.read_csv(MEMBER_CUSTOMER_MAP_CSV).set_index("member")["customer_id_key"]
    if member not in mapping.index:
        raise KeyError(f"No customer profile is linked to {member}")
    return int(mapping[member])


def recommend_product(member, artifacts):
    """Predict the product category for the member's customer profile."""
    customer_id = customer_for_member(member)
    merged = pd.read_csv(MERGED_DATA_CSV)
    customer_rows = merged[merged["customer_id_key"] == customer_id]
    if customer_rows.empty:
        raise KeyError(f"Customer {customer_id} is not in {MERGED_DATA_CSV.name}")
    features = product_features(customer_rows).iloc[[0]]  # social features are the same on every row of a customer
    category, confidence, probabilities = _predict(artifacts["product"], features)
    return {
        "member": member,
        "customer_id": customer_id,
        "category": category,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def run_transaction(image_path, audio_path, artifacts=None):
    """Full flow: face -> product model -> voice confirmation -> display or deny."""
    artifacts = artifacts or load_artifacts()

    face = identify_face(image_path, artifacts)
    if not face["accepted"]:
        return {
            "status": "ACCESS DENIED",
            "reason": f"Face not recognised (confidence {face['confidence']:.2f} < {face['threshold']:.2f})",
            "face": face,
        }

    product = recommend_product(face["member"], artifacts)  # computed, but withheld until the voice check

    voice = verify_voice(audio_path, face["member"], artifacts)
    if not voice["accepted"]:
        if voice["speaker"] != face["member"]:
            reason = f"Voice does not match {face['member']} (closest voice: {voice['speaker']})"
        else:
            reason = f"Voice match not confident enough ({voice['confidence']:.2f} < {voice['threshold']:.2f})"
        return {"status": "ACCESS DENIED", "reason": reason, "face": face, "voice": voice}

    return {"status": "APPROVED", "face": face, "voice": voice, "product": product}
