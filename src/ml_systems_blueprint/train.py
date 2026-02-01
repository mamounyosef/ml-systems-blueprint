from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import ExperimentConfig
from .data import load_dataset
from .metrics import (
    compute_binary_metrics,
    pick_threshold,
    plot_calibration,
    plot_confusion,
    plot_pr,
    plot_roc,
    save_metrics,
)
from .modeling import build_pipeline


def _cv_oof_probabilities(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    c_grid: list[float],
    cv_folds: int,
) -> tuple[float, np.ndarray]:
    """Select C via CV and return OOF probabilities for the chosen setting."""

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    best_c = c_grid[0]
    best_auc = -1.0
    best_oof = None

    for c in c_grid:
        oof = np.zeros_like(y_train, dtype=float)
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]

            pipe = build_pipeline(seed=seed)
            pipe.set_params(clf__C=float(c))
            pipe.fit(X_tr, y_tr)
            oof[val_idx] = pipe.predict_proba(X_val)[:, 1]

        auc = float(__import__("sklearn.metrics").metrics.roc_auc_score(y_train, oof))
        if auc > best_auc:
            best_auc = auc
            best_c = float(c)
            best_oof = oof

    assert best_oof is not None
    return best_c, best_oof


def train_and_save(out_dir: Path, seed: int = 42) -> dict:
    """Train with simple CV selection + threshold tuning, then write artifacts."""
    ds = load_dataset(seed=seed)

    best_c, oof_prob = _cv_oof_probabilities(
        ds.X_train,
        ds.y_train,
        seed=seed,
        c_grid=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        cv_folds=5,
    )

    threshold = pick_threshold(ds.y_train, oof_prob)

    pipe = build_pipeline(seed=seed)
    pipe.set_params(clf__C=best_c)
    pipe.fit(ds.X_train, ds.y_train)

    test_prob = pipe.predict_proba(ds.X_test)[:, 1]

    metrics = compute_binary_metrics(ds.y_test, test_prob, threshold=threshold)

    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    params_path = out_dir / "params.json"
    roc_path = out_dir / "roc.png"
    pr_path = out_dir / "pr.png"
    cal_path = out_dir / "calibration.png"
    cm_path = out_dir / "confusion_matrix.png"

    joblib.dump({"pipeline": pipe, "threshold": threshold}, model_path)
    save_metrics(metrics, metrics_path)
    params_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "best_C": best_c,
                "threshold": threshold,
            },
            indent=2,
        )
    )

    plot_roc(ds.y_test, test_prob, roc_path)
    plot_pr(ds.y_test, test_prob, pr_path)
    plot_calibration(ds.y_test, test_prob, cal_path)
    plot_confusion(ds.y_test, test_prob, threshold, cm_path)

    return {
        "artifacts": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "params": str(params_path),
            "roc": str(roc_path),
            "pr": str(pr_path),
            "calibration": str(cal_path),
            "confusion_matrix": str(cm_path),
        },
        "metrics": json.loads(metrics_path.read_text()),
    }


def train_from_config(cfg: ExperimentConfig) -> dict:
    out_dir = Path(cfg.artifacts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(seed=cfg.seed, test_size=cfg.data.test_size)

    best_c, oof_prob = _cv_oof_probabilities(
        ds.X_train,
        ds.y_train,
        seed=cfg.seed,
        c_grid=cfg.training.c_grid,
        cv_folds=cfg.training.cv_folds,
    )

    threshold = pick_threshold(ds.y_train, oof_prob)

    pipe = build_pipeline(seed=cfg.seed)
    pipe.set_params(clf__C=best_c)
    pipe.fit(ds.X_train, ds.y_train)

    test_prob = pipe.predict_proba(ds.X_test)[:, 1]
    metrics = compute_binary_metrics(ds.y_test, test_prob, threshold=threshold)

    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    params_path = out_dir / "params.json"

    joblib.dump({"pipeline": pipe, "threshold": threshold}, model_path)
    save_metrics(metrics, metrics_path)
    params_path.write_text(
        json.dumps(
            {
                "seed": cfg.seed,
                "test_size": cfg.data.test_size,
                "best_C": best_c,
                "threshold": threshold,
                "cv_folds": cfg.training.cv_folds,
                "c_grid": cfg.training.c_grid,
            },
            indent=2,
        )
    )

    plot_roc(ds.y_test, test_prob, out_dir / "roc.png")
    plot_pr(ds.y_test, test_prob, out_dir / "pr.png")
    plot_calibration(ds.y_test, test_prob, out_dir / "calibration.png")
    plot_confusion(ds.y_test, test_prob, threshold, out_dir / "confusion_matrix.png")

    return {
        "artifacts": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "params": str(params_path),
        },
        "metrics": json.loads(metrics_path.read_text()),
    }


def evaluate_saved_model(model_path: Path, seed: int = 42) -> dict:
    ds = load_dataset(seed=seed)
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    threshold = float(bundle["threshold"])

    prob_pos = pipe.predict_proba(ds.X_test)[:, 1]
    metrics = compute_binary_metrics(ds.y_test, prob_pos, threshold=threshold)
    return json.loads(json.dumps(metrics.__dict__))
