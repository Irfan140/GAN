# Kaggle entry point for train.py.
#
#   kaggle kernels push -p kaggle
#   kaggle kernels status irfan140/gan-dcgan-training
#   kaggle kernels output irfan140/gan-dcgan-training -p kaggle-output
#
# This clones the repo (so push your changes to GitHub first) and trains on the
# T4 with the arguments below. Everything worth keeping - weights, checkpoints,
# samples - is written under /kaggle/working, which Kaggle saves with the
# notebook version.
#
# To continue a run in a later session: download /kaggle/working/checkpoints,
# upload it as a Kaggle dataset, then add
#   "--resume", "/kaggle/input/<dataset>/latest.pt"
# to ARGS.

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/Irfan140/GAN.git"
SRC = Path("/kaggle/working/GAN")
WORK = Path("/kaggle/working")

ARGS = [
    "--epochs", "20",
    "--batch-size", "128",
    "--num-workers", "4",
    "--device", "cuda",
    "--amp",
    "--data-path", str(WORK / "data"),
    "--weights-dir", str(WORK / "weights"),
    "--output-dir", str(WORK / "output"),
    "--checkpoint-dir", str(WORK / "checkpoints"),
    # Drop these two lines if you have not added WANDB_API_KEY to Kaggle secrets.
    "--wandb-project", "gan-dcgan",
    "--wandb-run-name", "kaggle-t4",
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if not SRC.exists():
    run(["git", "clone", "--depth", "1", REPO, str(SRC)])

# Kaggle images usually ship wandb, but installing is cheap and keeps this
# working if the base image changes.
run([sys.executable, "-m", "pip", "install", "-q", "--no-input", "wandb"])

os.chdir(SRC)
run([sys.executable, "train.py", *ARGS])
