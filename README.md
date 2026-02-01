# ml-systems-blueprint

Note: This whole repo was created End to End by OpenClaw using GPT-5.2-Codex, without any human intervention.
The goal here was to demonstrate the capability and extendibility of Agentic systems like this.


A **production-minded ML starter repo**: reproducible experiments, clean packaging, a tiny CLI, tests, linting, and CI.

It ships with a complete, end-to-end example:
- dataset: scikit-learn built-in Breast Cancer dataset
- model: Logistic Regression + StandardScaler
- outputs: metrics, confusion matrix, and a serialized pipeline

## Quickstart

### 1) Create a virtualenv & install

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -e ".[dev]"
```

### 2) Run tests

```bash
pytest -q
```

### 3) Train a model

Recommended (config-driven):

```bash
python -m ml_systems_blueprint run --config configs/default.yaml
```

Quick path (defaults):

```bash
python -m ml_systems_blueprint train --out-dir artifacts
```

Artifacts written to `artifacts/`:
- `model.joblib` (pipeline + tuned threshold)
- `metrics.json`
- `params.json`
- `roc.png`, `pr.png`, `calibration.png`
- `confusion_matrix.png`

### 4) Evaluate the saved model

```bash
python -m ml_systems_blueprint eval --model artifacts/model.joblib
```

## Repo layout

- `src/ml_systems_blueprint/` — package code
- `tests/` — unit + smoke tests
- `.github/workflows/ci.yml` — GitHub Actions (lint + tests)
- `ruff.toml` — formatting + lint rules

## Why this exists

Most "ML repos" are notebooks + chaos. This one is a blueprint:
- **reproducible** (fixed seeds, deterministic split)
- **packaged** (importable modules, CLI entrypoint)
- **tested** (smoke tests that run in CI)
- **maintainable** (ruff + pre-commit)

## License

MIT
