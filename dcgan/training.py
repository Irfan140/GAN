"""The adversarial training loop and its per-batch helpers.

The update order mirrors GAN_implementation.ipynb exactly: D on real, D on fake,
then G on the (now updated) discriminator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from model import Z_DIM, Discriminator, Generator

from .checkpoint import load_checkpoint, resolve_resume, save_checkpoint
from .config import Config
from .data import build_dataloader
from .state import History
from .tracking import log_samples

REAL_LABEL = 1.0
FAKE_LABEL = 0.0
NUM_SAMPLES = 64  # 8x8 grid from a fixed noise vector
TAIL = 100  # iterations averaged for the stats reported to W&B at the end


def weights_init(module: nn.Module) -> None:
    """DCGAN init: Conv ~ N(0, 0.02), BatchNorm ~ N(1, 0.02) with zero bias."""
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def save_samples(netG: Generator, noise: torch.Tensor, path: Path) -> Path:
    """Render the fixed noise batch through G and write it as one grid image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        fake = netG(noise)
    vutils.save_image(fake, path, nrow=8, padding=2, normalize=True)
    return path


def tail_mean(values: list[float], n: int = TAIL) -> float:
    tail = values[-n:] or [0.0]
    return sum(tail) / len(tail)


def train(
    cfg: Config,
    dataloader: DataLoader,
    device: torch.device,
    run: Any | None = None,
) -> tuple[Generator, Discriminator, History]:
    """Run the adversarial loop, checkpointing as it goes."""
    samples_dir = cfg.output_dir / "samples"
    use_amp = cfg.amp and device.type == "cuda"

    netG = Generator().to(device)
    netG.apply(weights_init)
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(NUM_SAMPLES, Z_DIM, 1, 1, device=device)
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = History()
    step = 0
    start_epoch, start_batch = 0, 0

    resume_path = resolve_resume(cfg)
    if resume_path is not None:
        start_epoch, start_batch, step, history = load_checkpoint(
            resume_path, netG, netD, optimizerG, optimizerD, scaler
        )
        print(f"resumed from {resume_path} (epoch {start_epoch}, batch {start_batch}, step {step})")

    print(f"mixed precision: {'on' if use_amp else 'off'} | workers: {cfg.num_workers}")

    for epoch in range(start_epoch, cfg.epochs):
        for i, (real, _) in enumerate(dataloader):
            if epoch == start_epoch and i < start_batch:
                continue

            # (1) Update D on a real batch: maximise log D(x)
            netD.zero_grad(set_to_none=True)
            real = real.to(device, non_blocking=True)
            b_size = real.size(0)
            label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = netD(real)
                errD_real = criterion(output, label)
            scaler.scale(errD_real).backward()
            d_x = output.mean().item()

            # (2) Update D on a fake batch: minimise log D(G(z))
            noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fake = netG(noise)
                output = netD(fake.detach())
                label.fill_(FAKE_LABEL)
                errD_fake = criterion(output, label)
            scaler.scale(errD_fake).backward()
            d_g_z1 = output.mean().item()
            errD = errD_real + errD_fake
            scaler.step(optimizerD)

            # (3) Update G: fake labels are "real" for the generator's cost
            netG.zero_grad(set_to_none=True)
            label.fill_(REAL_LABEL)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = netD(fake)
                errG = criterion(output, label)
            scaler.scale(errG).backward()
            d_g_z2 = output.mean().item()
            scaler.step(optimizerG)
            scaler.update()

            history.g_loss.append(errG.item())
            history.d_loss.append(errD.item())
            history.d_x.append(d_x)
            history.d_g_z1.append(d_g_z1)
            history.d_g_z2.append(d_g_z2)

            if i % cfg.log_every == 0:
                print(
                    f"[{epoch + 1}/{cfg.epochs}][{i}/{len(dataloader)}]\t"
                    f"Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t"
                    f"D(x): {d_x:.4f}\tD(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}"
                )
                if run is not None:
                    run.log(
                        {
                            "loss_D": errD.item(),
                            "loss_G": errG.item(),
                            "D(x)": d_x,
                            "D(G(z))": d_g_z1,
                        },
                        step=step,
                    )

            if step % cfg.sample_every == 0:
                path = save_samples(netG, fixed_noise, samples_dir / f"step_{step:06d}.png")
                log_samples(run, path, step)

            if cfg.checkpoint_every and step > 0 and step % cfg.checkpoint_every == 0:
                save_checkpoint(
                    cfg.checkpoint_dir / "latest.pt",
                    netG, netD, optimizerG, optimizerD, scaler,
                    epoch=epoch, batch=i + 1, step=step, history=history,
                )
                print(f"checkpoint: {cfg.checkpoint_dir / 'latest.pt'}")

            step += 1
            if cfg.max_batches and i + 1 >= cfg.max_batches:
                break

        save_checkpoint(
            cfg.checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
            netG, netD, optimizerG, optimizerD, scaler,
            epoch=epoch + 1, batch=0, step=step, history=history,
        )
        save_checkpoint(
            cfg.checkpoint_dir / "latest.pt",
            netG, netD, optimizerG, optimizerD, scaler,
            epoch=epoch + 1, batch=0, step=step, history=history,
        )
        print(f"epoch {epoch + 1}/{cfg.epochs} done ({step} batches seen), checkpoint saved")

    final_path = save_samples(netG, fixed_noise, samples_dir / "final.png")
    log_samples(run, final_path, step)

    if run is not None:
        run.summary.update(
            {
                "final/loss_G": tail_mean(history.g_loss),
                "final/loss_D": tail_mean(history.d_loss),
                "final/D_x": tail_mean(history.d_x),
                "final/D_G_z": tail_mean(history.d_g_z2),
            }
        )

    return netG, netD, history


def run_single(cfg: Config, device: torch.device, run: Any | None = None) -> tuple[Generator, Discriminator, History]:
    """Build the DataLoader and train once with the given config."""
    dataloader = build_dataloader(cfg, device)
    return train(cfg, dataloader, device, run=run)
