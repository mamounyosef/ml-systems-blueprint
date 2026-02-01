# Model Card — ml-systems-blueprint baseline

## Summary

This repository trains a **tabular binary classifier** on the scikit-learn Breast Cancer dataset using a **standardized Logistic Regression** pipeline.

It intentionally focuses on **ML-systems quality** (reproducibility, packaging, tests, CI) rather than squeezing leaderboard performance.

## Intended use

- Demonstration / template for building maintainable ML repos
- Baseline training + evaluation flow with:
  - cross-validation hyperparameter selection
  - threshold tuning
  - ROC/PR/Calibration plots

## Data

- Source: `sklearn.datasets.load_breast_cancer()`
- Notes:
  - built-in, no network dependency
  - deterministic split with stratification

## Model

- Pipeline:
  - `StandardScaler`
  - `LogisticRegression(solver=lbfgs)`
- Selection:
  - cross-validated selection over `C` grid
- Decision threshold:
  - chosen by maximizing F1 on out-of-fold probabilities

## Metrics

Metrics are written to `artifacts/metrics.json` and include:
- ROC AUC
- PR AUC
- Brier score
- Precision / Recall / F1 at the tuned threshold

## Ethical considerations

This dataset is a classic benchmark. Do not use this model for medical decisions.

## Reproducibility

- Fixed random seed
- Deterministic train/test split
- CI runs lint + tests
