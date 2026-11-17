"""Dataset and device setup."""

from __future__ import annotations

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .config import Config

# The generator ends in five 2x upsampling stages from 4x4, so its output is 64x64.
IMAGE_SIZE = 64


def build_dataloader(cfg: Config, device: torch.device) -> DataLoader:
    """MNIST with the notebook's preprocessing, loaded by worker processes."""
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = dset.MNIST(root=cfg.data_path, download=True, transform=transform)
    print(f"MNIST: {len(dataset)} training images")

    parallel = cfg.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=parallel,
        prefetch_factor=2 if parallel else None,
    )
