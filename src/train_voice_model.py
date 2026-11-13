"""Train and evaluate the voiceprint verification model.

Data    data/audio_features.csv: 8 recordings (4 members x 2 phrases), each in 4 versions
        (original, pitch shift, time stretch, background noise) -> 32 rows, 36 acoustic features.
Design  Leave-one-recording-out cross-validation: a recording and all its augmented versions
        are held out together, so the model is always tested on a recording it has never
        heard (8 folds). Metrics are reported on the original clips and on all versions.
        The hold-out of notebook 03 (train on first recordings, test on the original second
        recordings) is reported for comparison. Open-set test: each member is removed from
        training in turn and presented as an unknown voice to set the rejection threshold.
Output  models/voiceprint_verification.joblib

Run     python -m src.train_voice_model
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

from src.config import AUDIO_FEATURES_CSV, VOICE_MODEL_PATH
from src.evaluation import candidate_models, classification_metrics, cross_val_proba, open_set_summary, select_best
from src.features import AUDIO_FEATURE_COLUMNS

MODEL_TYPE = "Voiceprint verification"
CV_ORIGINAL = "Leave-one-recording-out CV: original clips (8)"
CV_ALL = "Leave-one-recording-out CV: all versions (32)"
HOLDOUT_NAME = "Hold-out: second recordings (4 original clips)"


def load_data():
    df = pd.read_csv(AUDIO_FEATURES_CSV)
    df["speaker"] = df["speaker"].str.strip()
    df["recording"] = df["speaker"] + " | " + df["original_filename"]
    encoder = LabelEncoder().fit(df["speaker"])
    return df, df[AUDIO_FEATURE_COLUMNS], encoder.transform(df["speaker"]), list(encoder.classes_)


def run(save=True, verbose=True):
    df, X, y, classes = load_data()
    n_classes = len(classes)
    original = (df["augmentation"] == "original").to_numpy()

    recordings = sorted(df["recording"].unique())
    splits = [(np.where(df["recording"] != r)[0], np.where(df["recording"] == r)[0]) for r in recordings]
    second = df["original_filename"].str.startswith("Second")
    holdout = [(np.where(~second)[0], np.where(second)[0])]

    models = candidate_models()
    rows, cv_probas = [], {}
    for name, make in models.items():
        cv = cross_val_proba(make, X, y, splits, n_classes)
        ho = cross_val_proba(make, X, y, holdout, n_classes)
        ho_tested = ~np.isnan(ho[:, 0]) & original
        cv_probas[name] = cv
        rows.append({"model": MODEL_TYPE, "algorithm": name, "evaluation": CV_ORIGINAL,
                     **classification_metrics(y[original], cv[original], n_classes)})
        rows.append({"model": MODEL_TYPE, "algorithm": name, "evaluation": CV_ALL, **classification_metrics(y, cv, n_classes)})
        rows.append({"model": MODEL_TYPE, "algorithm": name, "evaluation": HOLDOUT_NAME,
                     **classification_metrics(y[ho_tested], ho[ho_tested], n_classes)})
    metrics = pd.DataFrame(rows)

    open_set, open_details = open_set_summary(models, X, y, n_classes, cv_probas)
    # Rank by the verification error (FAR + FRR), which also counts misidentified speakers,
    # then by closed-set macro F1 and log loss.
    ranking = metrics[metrics["evaluation"] == CV_ALL].merge(open_set, on="algorithm")
    best = select_best(ranking, by=("open_set_error", "f1_macro", "log_loss"), ascending=(True, False, True))
    chosen = open_set.set_index("algorithm").loc[best]

    final_model = models[best]().fit(X, y)
    artifact = {
        "model": final_model,
        "model_name": best,
        "classes": classes,
        "feature_columns": AUDIO_FEATURE_COLUMNS,
        "threshold": float(chosen["threshold"]),
        "metrics": metrics[metrics["algorithm"] == best].to_dict("records"),
        "open_set": chosen.to_dict(),
        "sklearn_version": sklearn.__version__,
        "trained_on": "data/audio_features.csv (32 rows: 8 recordings x 4 versions)",
    }
    if save:
        VOICE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, VOICE_MODEL_PATH)

    if verbose:
        print(f"\n=== {MODEL_TYPE} ===")
        print(metrics.round(4).to_string(index=False))
        print("\nOpen-set test (unknown voices):")
        print(open_set.round(4).to_string(index=False))
        print(f"\nSelected: {best} | threshold {chosen['threshold']:.3f} | saved: {save}")

    return {
        "metrics": metrics,
        "open_set": open_set,
        "best": best,
        "classes": classes,
        "y": y,
        "original": original,
        "cv_proba": cv_probas[best],
        "impostor_scores": open_details[best]["impostor_scores"],
        "threshold_table": open_details[best]["threshold_table"],
        "threshold": float(chosen["threshold"]),
        "artifact": artifact,
    }


if __name__ == "__main__":
    run()
