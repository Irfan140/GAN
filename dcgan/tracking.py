"""Weights & Biases integration: credentials, run setup, logging."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import Config, cfg_as_dict


def wandb_api_key() -> str | None:
    """W&B key from the environment, or from Kaggle secrets when present."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient  # only exists on Kaggle
    except ImportError:
        return None
    try:
        return UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception:  # secret not configured for this notebook
        return None


def wandb_credentials(cfg: Config) -> tuple[str | None, str]:
    """Resolve the API key (env or Kaggle secret) and the effective W&B mode."""
    key = wandb_api_key()
    if key:
        os.environ.setdefault("WANDB_API_KEY", key)
    mode = cfg.wandb_mode
    if mode == "auto":
        mode = "online" if key else "offline"
    return key, mode


def init_wandb(cfg: Config) -> Any | None:
    """Start a W&B run, or return None when logging is not requested/unavailable."""
    if not cfg.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        print(f"wandb unavailable ({exc}); continuing without experiment tracking")
        return None

    _, mode = wandb_credentials(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"W&B: project={cfg.wandb_project} mode={mode}")
    try:
        return wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            mode=mode,
            dir=str(cfg.output_dir),
            config=cfg_as_dict(cfg),
        )
    except Exception as exc:  # e.g. blocked service process, bad key, no network
        print(f"W&B init failed ({exc}); continuing without experiment tracking")
        return None


def log_samples(run: Any | None, path: Path, step: int) -> None:
    if run is None:
        return
    import wandb

    run.log({"samples": wandb.Image(str(path))}, step=step)
