"""Filesystem defaults, resolved against the repository root (not the CWD).

Same convention as inference.py, so scripts write to the same place no matter
which directory they are launched from.
"""

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent

DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weights"
OUTPUT_DIR = BASE_DIR / "output" / "training"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
