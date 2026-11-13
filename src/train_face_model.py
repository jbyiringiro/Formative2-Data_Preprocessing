"""Train and evaluate the facial recognition model.

Data    data/image_features.csv: 12 photos (4 members x neutral / smiling / surprised),
        each in 4 versions (original, rotated, flipped, grayscale) -> 48 rows,
        96 colour-histogram features.
Design  Leave-one-expression-out cross-validation: the test expression is never seen in
        training (3 folds x 16 rows). The 'surprised' fold alone is the hold-out used in
        notebook 02. Open-set test: each member is removed from training in turn and
        presented as an unknown face; this sets the confidence threshold that rejects
        unauthorized users.
Output  models/face_recognition.joblib

Run     python -m src.train_face_model
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

from src.config import FACE_MODEL_PATH, IMAGE_FEATURES_CSV
from src.evaluation import candidate_models, classification_metrics, cross_val_proba, open_set_summary, select_best
from src.features import IMAGE_FEATURE_COLUMNS

MODEL_TYPE = "Face recognition"
CV_NAME = "Leave-one-expression-out CV (48 images)"
HOLDOUT_NAME = "Hold-out: surprised expression (16 images)"


def load_data():
    df = pd.read_csv(IMAGE_FEATURES_CSV)
    df["member"] = df["member"].str.strip()
    encoder = LabelEncoder().fit(df["member"])
    return df, df[IMAGE_FEATURE_COLUMNS], encoder.transform(df["member"]), list(encoder.classes_)


def run(save=True, verbose=True):
    df, X, y, classes = load_data()
    n_classes = len(classes)
    expressions = sorted(df["expression"].unique())
    splits = [(np.where(df["expression"] != e)[0], np.where(df["expression"] == e)[0]) for e in expressions]
    holdout = [splits[expressions.index("surprised")]]

    models = candidate_models()
    rows, cv_probas = [], {}
    for name, make in models.items():
        cv = cross_val_proba(make, X, y, splits, n_classes)
        ho = cross_val_proba(make, X, y, holdout, n_classes)
        tested = ~np.isnan(ho[:, 0])
        cv_probas[name] = cv
        rows.append({"model": MODEL_TYPE, "algorithm": name, "evaluation": CV_NAME, **classification_metrics(y, cv, n_classes)})
        rows.append({"model": MODEL_TYPE, "algorithm": name, "evaluation": HOLDOUT_NAME,
                     **classification_metrics(y[tested], ho[tested], n_classes)})
    metrics = pd.DataFrame(rows)

    open_set, open_details = open_set_summary(models, X, y, n_classes, cv_probas)
    # Rank by the verification error (FAR + FRR), which also counts misidentified members,
    # then by closed-set macro F1 and log loss.
    ranking = metrics[metrics["evaluation"] == CV_NAME].merge(open_set, on="algorithm")
    best = select_best(ranking, by=("open_set_error", "f1_macro", "log_loss"), ascending=(True, False, True))
    chosen = open_set.set_index("algorithm").loc[best]

    final_model = models[best]().fit(X, y)
    artifact = {
        "model": final_model,
        "model_name": best,
        "classes": classes,
        "feature_columns": IMAGE_FEATURE_COLUMNS,
        "threshold": float(chosen["threshold"]),
        "metrics": metrics[metrics["algorithm"] == best].to_dict("records"),
        "open_set": chosen.to_dict(),
        "sklearn_version": sklearn.__version__,
        "trained_on": "data/image_features.csv (48 rows: 12 photos x 4 versions)",
    }
    if save:
        FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, FACE_MODEL_PATH)

    if verbose:
        print(f"\n=== {MODEL_TYPE} ===")
        print(metrics.round(4).to_string(index=False))
        print("\nOpen-set test (unknown faces):")
        print(open_set.round(4).to_string(index=False))
        print(f"\nSelected: {best} | threshold {chosen['threshold']:.3f} | saved: {save}")

    return {
        "metrics": metrics,
        "open_set": open_set,
        "best": best,
        "classes": classes,
        "y": y,
        "cv_proba": cv_probas[best],
        "impostor_scores": open_details[best]["impostor_scores"],
        "threshold_table": open_details[best]["threshold_table"],
        "threshold": float(chosen["threshold"]),
        "artifact": artifact,
    }


if __name__ == "__main__":
    run()
