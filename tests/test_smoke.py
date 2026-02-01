from pathlib import Path

from ml_systems_blueprint.train import evaluate_saved_model, train_and_save


def test_train_and_eval_smoke(tmp_path: Path) -> None:
    res = train_and_save(out_dir=tmp_path, seed=123)
    model_path = Path(res["artifacts"]["model"])
    metrics = evaluate_saved_model(model_path, seed=123)
    # sanity checks on probability-based metrics
    assert 0.9 <= metrics["roc_auc"] <= 1.0
    assert 0.9 <= metrics["pr_auc"] <= 1.0
