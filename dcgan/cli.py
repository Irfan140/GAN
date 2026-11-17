"""Command-line entry point: wire config, tracking, training, and artifacts."""

from __future__ import annotations

import torch

from .artifacts import plot_losses, save_checkpoints, save_history
from .config import cfg_as_dict, merge_wandb_config, parse_args
from .runtime import configure_backends, resolve_device, set_seed
from .sweep import run_sweep
from .tracking import init_wandb
from .training import run_single


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    configure_backends(device)

    print(f"PyTorch {torch.__version__} | device: {device}")
    print(f"epochs {cfg.epochs} | batch {cfg.batch_size} | lr {cfg.lr} | seed {cfg.seed}")

    if cfg.sweep_config is not None:
        if not cfg.wandb_project:
            raise SystemExit("--sweep-config needs --wandb-project")
        run_sweep(cfg, device)
        return

    run = init_wandb(cfg)
    if run is not None:
        cfg = merge_wandb_config(cfg, dict(run.config))
        print("config after W&B overrides:", cfg_as_dict(cfg))

    try:
        netG, netD, history = run_single(cfg, device, run=run)
    except KeyboardInterrupt:
        print(f"\ninterrupted - resume with: uv run train.py --resume --checkpoint-dir {cfg.checkpoint_dir}")
        raise SystemExit(130)

    g_path, d_path = save_checkpoints(netG, netD, cfg.weights_dir)
    print("saved:", g_path)
    print("saved:", d_path)
    print("loss curve:", plot_losses(history, cfg.output_dir))
    print("history:", save_history(history, cfg.output_dir))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
