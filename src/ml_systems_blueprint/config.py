from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    test_size: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    c_grid: list[float]
    cv_folds: int = 5
    threshold_metric: str = "f1"


@dataclass(frozen=True)
class ArtifactsConfig:
    out_dir: str = "artifacts"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    training: TrainingConfig
    artifacts: ArtifactsConfig


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    return d[key]


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())

    data_raw = _require(raw, "data")
    training_raw = _require(raw, "training")
    artifacts_raw = _require(raw, "artifacts")

    return ExperimentConfig(
        seed=int(_require(raw, "seed")),
        data=DataConfig(test_size=float(data_raw.get("test_size", 0.2))),
        training=TrainingConfig(
            c_grid=[float(x) for x in _require(training_raw, "c_grid")],
            cv_folds=int(training_raw.get("cv_folds", 5)),
            threshold_metric=str(training_raw.get("threshold_metric", "f1")),
        ),
        artifacts=ArtifactsConfig(out_dir=str(artifacts_raw.get("out_dir", "artifacts"))),
    )
