"""Train and evaluate the product recommendation model.

Data    data/merged_customer_data.csv: 150 transactions with the customer's aggregated
        social-media features; target = product_category (5 classes).
Design  Customer-grouped, stratified 5-fold cross-validation: all transactions of a customer
        are in the same fold, so the model is always tested on customers it has never seen.
        The first fold is exactly the test split of notebook 01 (trained on the other 119
        transactions) and is reported for comparison.
Output  models/product_recommendation.joblib

Run     python -m src.train_product_model
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.config import MERGED_DATA_CSV, PRODUCT_MODEL_PATH, RANDOM_STATE
from src.evaluation import candidate_models, classification_metrics, cross_val_proba, select_best
from src.features import PRODUCT_FEATURE_COLUMNS, product_features

MODEL_TYPE = "Product recommendation"
CV_NAME = "Customer-grouped 5-fold CV (150 transactions)"
HOLDOUT_NAME = "Hold-out: notebook 01 test split (31 transactions)"


def load_data():
    merged = pd.read_csv(MERGED_DATA_CSV)
    encoder = LabelEncoder().fit(merged["product_category"])
    return merged, product_features(merged), encoder.transform(merged["product_category"]), list(encoder.classes_)


def run(save=True, verbose=True):
    merged, X, y, classes = load_data()
    n_classes = len(classes)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(X, y, groups=merged["customer_id_key"]))
    holdout = [splits[0]]

    models = candidate_models(impute=True)
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

    best = select_best(metrics[metrics["evaluation"] == CV_NAME])
    final_model = models[best]().fit(X, y)
    artifact = {
        "model": final_model,
        "model_name": best,
        "classes": classes,
        "feature_columns": PRODUCT_FEATURE_COLUMNS,
        "target_column": "product_category",
        "metrics": metrics[metrics["algorithm"] == best].to_dict("records"),
        "sklearn_version": sklearn.__version__,
        "trained_on": "data/merged_customer_data.csv (150 transactions)",
    }
    if save:
        PRODUCT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, PRODUCT_MODEL_PATH)

    if verbose:
        print(f"\n=== {MODEL_TYPE} ===")
        print(metrics.round(4).to_string(index=False))
        print(f"\nSelected: {best} | saved: {save}")

    return {
        "metrics": metrics,
        "best": best,
        "classes": classes,
        "y": y,
        "cv_proba": cv_probas[best],
        "holdout_index": holdout[0][1],
        "artifact": artifact,
    }


if __name__ == "__main__":
    run()
