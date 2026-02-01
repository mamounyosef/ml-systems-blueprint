from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class BinaryMetrics:
    roc_auc: float
    pr_auc: float
    brier: float
    f1: float
    precision: float
    recall: float
    threshold: float


def pick_threshold(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    """Pick a decision threshold by maximizing F1 (simple + strong default)."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (prob_pos >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def compute_binary_metrics(
    y_true: np.ndarray, prob_pos: np.ndarray, threshold: float
) -> BinaryMetrics:
    y_pred = (prob_pos >= threshold).astype(int)
    return BinaryMetrics(
        roc_auc=float(roc_auc_score(y_true, prob_pos)),
        pr_auc=float(average_precision_score(y_true, prob_pos)),
        brier=float(brier_score_loss(y_true, prob_pos)),
        f1=float(f1_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred)),
        threshold=float(threshold),
    )


def save_metrics(metrics: BinaryMetrics, out_path: Path) -> None:
    out_path.write_text(json.dumps(asdict(metrics), indent=2))


def plot_roc(y_true: np.ndarray, prob_pos: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    auc = roc_auc_score(y_true, prob_pos)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pr(y_true: np.ndarray, prob_pos: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, prob_pos)
    ap = average_precision_score(y_true, prob_pos)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR AUC={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, prob_pos: np.ndarray, out_path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, prob_pos, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray, prob_pos: np.ndarray, threshold: float, out_path: Path
) -> None:
    y_pred = (prob_pos >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix (t={threshold:.2f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
