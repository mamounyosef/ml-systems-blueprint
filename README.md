# ML Systems Blueprint

A **production-minded ML starter repository** that provides a solid foundation for machine learning projects with reproducible experiments, clean packaging, a command-line interface, comprehensive tests, linting, and CI/CD pipeline.

## 🎯 Features

- **End-to-End Example**: Complete ML pipeline from data loading to model evaluation
- **Reproducible Experiments**: Fixed seeds, deterministic splits, and configurable parameters
- **Production Ready**: Packaged as importable Python modules with CLI entrypoint
- **Comprehensive Testing**: Unit tests and smoke tests integrated in CI
- **Code Quality**: Automated linting and formatting with Ruff
- **Flexible Configuration**: YAML-based configuration for experiments
- **Rich Outputs**: Metrics, visualizations (ROC, PR curves, calibration plots), confusion matrix

## 📊 Example Project

This blueprint includes a complete implementation using:
- **Dataset**: Breast Cancer dataset (scikit-learn built-in)
- **Model**: Logistic Regression with StandardScaler preprocessing
- **Training**: Cross-validation for hyperparameter tuning
- **Evaluation**: Comprehensive metrics and visualizations

## 🚀 Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/MacOS
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install -U pip
pip install -e ".[dev]"
```

### 2. Run Tests

Ensure everything is working correctly:

```bash
pytest -q
```

### 3. Train Your First Model

Two approaches are available:

**Recommended (Config-driven)**:
```bash
python -m ml_systems_blueprint run --config configs/default.yaml
```

**Quick Start (with defaults)**:
```bash
python -m ml_systems_blueprint train --out-dir artifacts
```

### 4. Evaluate the Model

```bash
python -m ml_systems_blueprint eval --model artifacts/model.joblib
```

### 5. Explore the Results

Training generates the following artifacts in `artifacts/`:
- `model.joblib` - Serialized pipeline with optimized threshold
- `metrics.json` - Performance metrics (accuracy, precision, recall, F1, AUC)
- `params.json` - Training parameters and configuration
- `roc.png` - Receiver Operating Characteristic curve
- `pr.png` - Precision-Recall curve
- `calibration.png` - Calibration plot
- `confusion_matrix.png` - Confusion matrix visualization

## 📁 Repository Structure

```
ml-systems-blueprint/
├── src/ml_systems_blueprint/    # Package source code
│   ├── __init__.py
│   ├── cli.py                   # Command-line interface
│   ├── train.py                 # Training pipeline
│   ├── eval.py                  # Evaluation utilities
│   └── utils/                   # Helper modules
├── tests/                       # Unit and integration tests
├── configs/                     # Experiment configurations
│   └── default.yaml            # Default training parameters
├── docs/                        # Documentation
│   └── MODEL_CARD.md           # Model documentation template
├── .github/workflows/           # CI/CD pipelines
│   └── ci.yml                  # GitHub Actions workflow
├── pyproject.toml              # Project configuration
├── ruff.toml                   # Linting and formatting rules
└── README.md                   # This file
```

## ⚙️ Configuration

The `configs/default.yaml` file demonstrates how to configure experiments:

```yaml
# Default experiment configuration
seed: 42

data:
  test_size: 0.2

training:
  # Cross-validation grid for regularization strength
  c_grid: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
  cv_folds: 5

  # Metric for threshold selection (f1, precision, recall)
  threshold_metric: f1

artifacts:
  out_dir: artifacts
```

## 🛠️ Development

### Code Quality Tools

This project uses:
- **Ruff**: Fast Python linter and formatter
- **Pre-commit hooks**: Ensure code quality before commits
- **Type checking**: Optional mypy integration
- **Testing**: Pytest with comprehensive coverage

### Adding New Features

1. Add source code to `src/ml_systems_blueprint/`
2. Write tests in `tests/`
3. Update CLI commands in `cli.py` if needed
4. Document new configurations in `docs/`

## 📋 CLI Commands

The package provides a `mlsb` command for quick access:

```bash
# Train with default config
mlsb train --out-dir artifacts

# Run with custom config
mlsb run --config configs/experiment.yaml

# Evaluate model
mlsb eval --model artifacts/model.joblib
```

## 🧪 Why This Blueprint?

Many ML repositories start as notebooks and evolve into chaos. This blueprint provides:

- ✅ **Reproducibility**: Fixed random seeds, deterministic data splits
- ✅ **Modularity**: Clean, importable modules with clear separation of concerns
- ✅ **Testing**: Comprehensive test suite that runs in CI/CD
- ✅ **Maintainability**: Consistent code style and automated quality checks
- ✅ **Scalability**: Structure that grows from prototype to production

## 📚 Model Cards

Each trained model should include a model card. See `docs/MODEL_CARD.md` for a template.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with best practices from the ML community, inspired by the need for production-ready ML templates that bridge the gap between research notebooks and deployed systems.
