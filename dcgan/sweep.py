"""Weights & Biases hyperparameter search: run an agent over sweep.yaml."""

from __future__ import annotations

from dataclasses import replace

import torch

from .artifacts import save_checkpoints
from .config import Config, cfg_as_dict, merge_wandb_config
from .tracking import wandb_credentials
from .training import run_single


def run_sweep(cfg: Config, device: torch.device) -> None:
    """Run a W&B hyperparameter search; each trial calls run_single()."""
    import wandb
    import yaml

    assert cfg.sweep_config is not None
    sweep = yaml.safe_load(cfg.sweep_config.read_text(encoding="utf-8"))
    base = replace(cfg, sweep_config=None)
    _, mode = wandb_credentials(cfg)
    if mode != "online":
        print("warning: W&B sweeps normally run in online mode; set WANDB_API_KEY to log results")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    sweep_id = wandb.sweep(sweep, project=cfg.wandb_project)
    print(f"W&B sweep: {sweep_id} ({cfg.sweep_count} trials)")

    def trial() -> None:
        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            mode=mode,
            dir=str(cfg.output_dir),
            config=cfg_as_dict(base),
        )
        trial_cfg = merge_wandb_config(base, dict(run.config))
        trial_dir = cfg.output_dir / f"trial_{run.id}"
        trial_cfg = replace(
            trial_cfg,
            output_dir=trial_dir,
            weights_dir=trial_dir / "weights",
            checkpoint_dir=trial_dir / "checkpoints",
        )
        print(f"trial {run.id}: lr={trial_cfg.lr} beta1={trial_cfg.beta1} batch={trial_cfg.batch_size}")
        netG, netD, _ = run_single(trial_cfg, device, run=run)
        save_checkpoints(netG, netD, trial_cfg.weights_dir)
        run.finish()

    wandb.agent(sweep_id, function=trial, count=cfg.sweep_count)
