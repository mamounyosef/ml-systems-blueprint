from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from .config import load_config
from .train import evaluate_saved_model, train_and_save, train_from_config

app = typer.Typer(no_args_is_help=True, help="ml-systems-blueprint CLI")

# ruff (B008) prefers these Option/Argument objects to be module-level singletons
OUT_DIR_OPT = typer.Option(Path("artifacts"), help="Where to write artifacts")
SEED_OPT = typer.Option(42, help="Random seed")
MODEL_OPT = typer.Option(..., help="Path to a saved joblib pipeline")
CONFIG_OPT = typer.Option(Path("configs/default.yaml"), help="Path to a YAML config")


@app.command()
def train(
    out_dir: Path = OUT_DIR_OPT,
    seed: int = SEED_OPT,
) -> None:
    """Train the example model and write artifacts (quick path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_and_save(out_dir=out_dir, seed=seed)
    print("[bold green]Training complete[/bold green]")
    print(json.dumps(result, indent=2))


@app.command()
def run(
    config: Path = CONFIG_OPT,
) -> None:
    """Run training from a YAML config (recommended)."""
    cfg = load_config(config)
    result = train_from_config(cfg)
    print("[bold green]Run complete[/bold green]")
    print(json.dumps(result, indent=2))


@app.command(name="eval")
def eval_cmd(
    model: Path = MODEL_OPT,
) -> None:
    """Evaluate a saved model."""
    metrics = evaluate_saved_model(model)
    print("[bold cyan]Evaluation[/bold cyan]")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
