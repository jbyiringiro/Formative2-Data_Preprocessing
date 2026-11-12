"""Candidate algorithms and evaluation helpers shared by the three training scripts.

All models are trained on integer-encoded labels (0..k-1) so that Logistic Regression,
Random Forest and XGBoost can be compared on exactly the same footing; the class names
are stored next to each saved model.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

BASELINE = "Baseline (class prior)"


def candidate_models(impute=False):
    """Factories for the algorithms compared for every model, plus a no-skill baseline.

    impute=True prepends median imputation (used by the product model, whose social
    scores are missing for customers without a social profile).
    """

    def pipeline(estimator, scale=False):
        steps = [("imputer", SimpleImputer(strategy="median"))] if impute else []
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", estimator))
        return Pipeline(steps)

    return {
        BASELINE: lambda: pipeline(DummyClassifier(strategy="prior")),
        "Logistic Regression": lambda: pipeline(
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), scale=True
        ),
        "Random Forest": lambda: pipeline(
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        ),
        "XGBoost": lambda: pipeline(
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.1,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
    }


def full_proba(model, X, n_classes):
    """predict_proba with one column per class, even if a class was absent from training."""
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    out = np.zeros((len(X), n_classes))
    out[:, np.asarray(model.classes_, dtype=int)] = proba
    return out / out.sum(axis=1, keepdims=True)  # XGBoost returns float32 rows that sum to 0.9999999


def cross_val_proba(make_model, X, y, splits, n_classes):
    """Out-of-fold class probabilities; rows never used as test data stay NaN."""
    proba = np.full((len(y), n_classes), np.nan)
    for train_idx, test_idx in splits:
        model = make_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        proba[test_idx] = full_proba(model, X.iloc[test_idx], n_classes)
    return proba


def classification_metrics(y_true, proba, n_classes):
    """Accuracy, macro F1-score and log loss (cross-entropy) from class probabilities."""
    y_pred = proba.argmax(axis=1)
    labels = list(range(n_classes))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=labels),
    }


def impostor_scores(make_model, X, y, n_classes):
    """Open-set test: retrain without each person and score that person as an unknown user.

    Returns the highest class probability the model gives each unseen person's samples,
    i.e. how confident the system would be when a stranger presents themselves.
    """
    scores = np.zeros(len(y))
    for person in range(n_classes):
        seen = y != person
        model = make_model()
        model.fit(X[seen], LabelEncoder().fit_transform(y[seen]))
        scores[~seen] = model.predict_proba(X[~seen]).max(axis=1)
    return scores


def choose_threshold(genuine_scores, genuine_correct, impostor, grid=None):
    """Pick the confidence threshold that minimises FAR + FRR (ties: lowest FAR, then middle).

    A genuine attempt is accepted only if it is identified correctly *and* its confidence
    reaches the threshold; an impostor is (wrongly) accepted if its confidence reaches it.
    """
    grid = np.round(np.linspace(0, 1, 201), 3) if grid is None else grid
    rows = []
    for t in grid:
        frr = np.mean(~(genuine_correct & (genuine_scores >= t)))
        far = np.mean(impostor >= t)
        rows.append({"threshold": t, "FAR": far, "FRR": frr, "total_error": far + frr})
    table = pd.DataFrame(rows)
    best = table[np.isclose(table["total_error"], table["total_error"].min())]
    best = best[np.isclose(best["FAR"], best["FAR"].min())]
    threshold = float(np.round(best["threshold"].median(), 3))
    far = float(np.mean(impostor >= threshold))
    frr = float(np.mean(~(genuine_correct & (genuine_scores >= threshold))))
    return threshold, far, frr, table


def select_best(table, by=("f1_macro", "log_loss"), ascending=(False, True)):
    """Best non-baseline algorithm, ranked by the given columns (default: macro F1, then log loss)."""
    candidates = table[table["algorithm"] != BASELINE]
    ranked = candidates.sort_values(list(by), ascending=list(ascending))
    return ranked.iloc[0]["algorithm"]


def open_set_summary(make_models, X, y, n_classes, cv_probas):
    """Threshold, FAR and FRR of every (non-baseline) algorithm in the open-set test."""
    rows, details = [], {}
    for name, make in make_models.items():
        if name == BASELINE:
            continue
        genuine = cv_probas[name]
        scored = ~np.isnan(genuine[:, 0])
        impostor = impostor_scores(make, X, y, n_classes)
        threshold, far, frr, table = choose_threshold(
            genuine[scored].max(axis=1), genuine[scored].argmax(axis=1) == y[scored], impostor
        )
        rows.append({"algorithm": name, "threshold": threshold, "FAR": far, "FRR": frr, "open_set_error": far + frr})
        details[name] = {"impostor_scores": impostor, "threshold_table": table}
    return pd.DataFrame(rows), details
