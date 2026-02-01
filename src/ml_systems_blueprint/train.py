from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .data import load_dataset
from .modeling import build_pipeline


def _plot_confusion_matrix(cm, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for (i, j), v in __import__("numpy").ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def train_and_save(out_dir: Path, seed: int = 42) -> dict:
    ds = load_dataset(seed=seed)
    pipe = build_pipeline(seed=seed)

    pipe.fit(ds.X_train, ds.y_train)
    y_pred = pipe.predict(ds.X_test)

    metrics = {
        "accuracy": float(accuracy_score(ds.y_test, y_pred)),
        "f1": float(f1_score(ds.y_test, y_pred)),
        "precision": float(precision_score(ds.y_test, y_pred)),
        "recall": float(recall_score(ds.y_test, y_pred)),
        "classification_report": classification_report(ds.y_test, y_pred, output_dict=True),
    }

    cm = confusion_matrix(ds.y_test, y_pred)

    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    cm_path = out_dir / "confusion_matrix.png"

    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _plot_confusion_matrix(cm, cm_path)

    return {
        "artifacts": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "confusion_matrix": str(cm_path),
        },
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
    }


def evaluate_saved_model(model_path: Path, seed: int = 42) -> dict:
    ds = load_dataset(seed=seed)
    pipe = joblib.load(model_path)
    y_pred = pipe.predict(ds.X_test)

    return {
        "accuracy": float(accuracy_score(ds.y_test, y_pred)),
        "f1": float(f1_score(ds.y_test, y_pred)),
        "precision": float(precision_score(ds.y_test, y_pred)),
        "recall": float(recall_score(ds.y_test, y_pred)),
    }
