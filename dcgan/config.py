"""Run configuration: CLI parsing, W&B overrides, serialization."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from .paths import CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR, WEIGHTS_DIR


@dataclass(frozen=True)
class Config:
    """Everything a training run needs. Built by parse_args(), then optionally
    overridden by a W&B sweep via merge_wandb_config()."""

    data_path: Path
    epochs: int
    batch_size: int
    lr: float
    beta1: float
    beta2: float
    seed: int
    num_workers: int
    device: str
    amp: bool
    log_every: int
    sample_every: int
    max_batches: int
    weights_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    checkpoint_every: int
    resume: str | None
    wandb_project: str | None
    wandb_mode: str
    wandb_run_name: str | None
    sweep_config: Path | None
    sweep_count: int


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Train the DCGAN on MNIST (same defaults as the notebook).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run train.py --epochs 20 --device cuda\n"
            "  uv run train.py --resume\n"
            "  uv run train.py --wandb-project gan-dcgan\n"
            "  uv run train.py --sweep-config sweep.yaml --sweep-count 8\n"
        ),
    )
    parser.add_argument("--data-path", type=Path, default=DATA_DIR, help="where MNIST is downloaded")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="images per batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--seed", type=int, default=1, help="random seed for weights, noise, and shuffling")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes (Kaggle: 4)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="compute device")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mixed precision (fp16) on CUDA; ignored on CPU",
    )
    parser.add_argument("--log-every", type=int, default=50, help="print stats every N batches")
    parser.add_argument("--sample-every", type=int, default=500, help="save a sample grid every N batches")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="stop each epoch after N batches (0 = all; use a small value for a smoke test)",
    )
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR, help="where generator.pth / discriminator.pth go")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="samples, loss curve, history.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR, help="resumable .pt checkpoints")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="write latest.pt every N batches (0 = only at epoch end)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="PATH",
        help="resume from PATH, or from <checkpoint-dir>/latest.pt when no path is given",
    )
    parser.add_argument("--wandb-project", default=None, help="enable Weights & Biases logging for this project")
    parser.add_argument(
        "--wandb-mode",
        choices=("auto", "online", "offline", "disabled"),
        default="auto",
        help="auto = online when a WANDB_API_KEY is available, otherwise offline",
    )
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    parser.add_argument("--sweep-config", type=Path, default=None, help="W&B sweep YAML (runs an agent instead of one run)")
    parser.add_argument("--sweep-count", type=int, default=10, help="number of trials to run for --sweep-config")
    return Config(**vars(parser.parse_args(argv)))


def cfg_as_dict(cfg: Config) -> dict[str, Any]:
    """Config -> primitive dict, for W&B config and checkpoints."""
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(cfg).items()}


def merge_wandb_config(cfg: Config, run_config: dict[str, Any]) -> Config:
    """Let a W&B sweep override dataclass fields, keeping their types."""
    updates: dict[str, Any] = {}
    for f in fields(cfg):
        if f.name not in run_config:
            continue
        current, value = getattr(cfg, f.name), run_config[f.name]
        if current is None:
            continue
        if isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, Path):
            value = Path(value)
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        updates[f.name] = value
    return replace(cfg, **updates) if updates else cfg
