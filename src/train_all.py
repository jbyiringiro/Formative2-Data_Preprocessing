"""Train and evaluate all three models and save the combined metrics table.

Run     python -m src.train_all
Output  models/face_recognition.joblib
        models/voiceprint_verification.joblib
        models/product_recommendation.joblib
        outputs/model_metrics.csv          accuracy, macro F1 and log loss per model / algorithm / evaluation
        outputs/open_set_thresholds.csv    rejection thresholds with FAR / FRR for face and voice
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import train_face_model, train_product_model, train_voice_model
from src.config import METRICS_CSV, OUTPUTS_DIR


def run_all(save=True, verbose=True):
    results = {
        "face": train_face_model.run(save=save, verbose=verbose),
        "voice": train_voice_model.run(save=save, verbose=verbose),
        "product": train_product_model.run(save=save, verbose=verbose),
    }
    metrics = pd.concat([r["metrics"] for r in results.values()], ignore_index=True)
    open_set = pd.concat(
        [results[key]["open_set"].assign(model=results[key]["metrics"]["model"].iloc[0]) for key in ("face", "voice")],
        ignore_index=True,
    )[["model", "algorithm", "threshold", "FAR", "FRR", "open_set_error"]]
    selected = pd.DataFrame(
        [{"model": r["metrics"]["model"].iloc[0], "selected_algorithm": r["best"]} for r in results.values()]
    )

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics.round(4).to_csv(METRICS_CSV, index=False)
        open_set.round(4).to_csv(OUTPUTS_DIR / "open_set_thresholds.csv", index=False)

    if verbose:
        print("\n=== Selected models ===")
        print(selected.to_string(index=False))
    return {**results, "metrics": metrics, "open_set": open_set, "selected": selected}


if __name__ == "__main__":
    run_all()
