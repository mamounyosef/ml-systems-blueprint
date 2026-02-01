from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(seed: int = 42) -> Pipeline:
    """A strong baseline for tabular binary classification."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    random_state=seed,
                    n_jobs=None,
                ),
            ),
        ]
    )
