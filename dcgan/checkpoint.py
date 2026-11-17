"""Checkpoint save / load / resume resolution.

Checkpoints survive Kaggle's session limit: everything needed to continue at the
exact batch is stored (networks, optimizers, AMP scaler, RNG state, history).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config
from .state import History


def save_checkpoint(
    path: Path,
    netG: nn.Module,
    netD: nn.Module,
    optimizerG: optim.Optimizer,
    optimizerD: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    batch: int,
    step: int,
    history: History,
) -> None:
    """Store the full training state. `epoch`/`batch` are the resume point,
    i.e. the next batch to process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "batch": batch,
            "step": step,
            "netG": netG.state_dict(),
            "netD": netD.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "optimizerD": optimizerD.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "history": asdict(history),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    netG: nn.Module,
    netD: nn.Module,
    optimizerG: optim.Optimizer,
    optimizerD: optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int, int, History]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    netG.load_state_dict(checkpoint["netG"])
    netD.load_state_dict(checkpoint["netD"])
    optimizerG.load_state_dict(checkpoint["optimizerG"])
    optimizerD.load_state_dict(checkpoint["optimizerD"])
    if checkpoint["scaler"] is not None and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler"])
    torch.set_rng_state(checkpoint["rng"])
    if checkpoint["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    history = History(**checkpoint["history"])
    return checkpoint["epoch"], checkpoint["batch"], checkpoint["step"], history


def resolve_resume(cfg: Config) -> Path | None:
    """Turn --resume into a concrete path; None means start from scratch."""
    if not cfg.resume:
        return None
    path = cfg.checkpoint_dir / "latest.pt" if cfg.resume == "latest" else Path(cfg.resume)
    if not path.exists():
        if cfg.resume == "latest":
            print(f"no checkpoint at {path}, starting fresh")
            return None
        raise SystemExit(f"checkpoint not found: {path}")
    return path
