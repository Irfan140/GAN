"""Device selection and reproducibility."""

from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    """Map --device {auto,cpu,cuda} to a concrete torch device."""
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda was requested but CUDA is not available")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_backends(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
