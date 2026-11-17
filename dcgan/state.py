"""The run state container: per-iteration losses.

Kept separate from Config because it is mutable and serialized into
checkpoints, while Config is frozen and describes how to train.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class History:
    """Per-iteration losses, kept for plotting, resuming, and W&B summaries."""

    g_loss: list[float] = field(default_factory=list)
    d_loss: list[float] = field(default_factory=list)
    d_x: list[float] = field(default_factory=list)
    d_g_z1: list[float] = field(default_factory=list)
    d_g_z2: list[float] = field(default_factory=list)
