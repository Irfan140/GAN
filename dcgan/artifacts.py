"""Artifacts written at the end of a run: weights, loss curve, history."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

# Headless: runs happen on Kaggle / remote boxes with no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from model import Discriminator, Generator  # noqa: E402

from .paths import OUTPUT_DIR, WEIGHTS_DIR  # noqa: E402
from .state import History  # noqa: E402


def save_checkpoints(
    netG: Generator,
    netD: Discriminator,
    weights_dir: Path = WEIGHTS_DIR,
) -> tuple[Path, Path]:
    """Write plain state dicts, the layout inference.py and load_weights expect."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    g_path = weights_dir / "generator.pth"
    d_path = weights_dir / "discriminator.pth"
    torch.save(netG.state_dict(), g_path)
    torch.save(netD.state_dict(), d_path)
    return g_path, d_path


def plot_losses(history: History, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ax.plot(history.d_loss, label="D loss", alpha=0.8)
    ax.plot(history.g_loss, label="G loss", alpha=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("DCGAN training losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_history(history: History, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "history.json"
    path.write_text(json.dumps(asdict(history), indent=2), encoding="utf-8")
    return path
