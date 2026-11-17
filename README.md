# DCGAN — Handwritten Digit Generation

A from-scratch PyTorch implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on **MNIST** to generate 64×64 grayscale handwritten digits from random noise.

![DCGAN training progression](dcgan_training.gif)

*Generator samples on a fixed noise vector as training progresses.*

## Overview

This repo contains a complete DCGAN workflow:

- **Training notebook** (`04_GAN_implementation.ipynb`) — dataset loading, model definitions, weight initialization, the adversarial training loop, and loss/sample visualization.
- **Reusable model code** (`model.py`) — the exact `Generator` and `Discriminator` architectures used for training, plus a checkpoint loader.
- **Inference script** (`inferance.py`) — loads the trained weights, samples 16 digits in a 4×4 grid, and optionally scores them with the discriminator. It creates `output/` on demand and resolves its paths relative to the script, so it runs from any directory.
- **Pre-trained checkpoints** — `generator.pth` and `discriminator.pth`, ready to use without retraining.
- **Training animation** — `dcgan_training.gif`, generated from snapshots taken during training.

The generator maps a 100-dimensional latent vector to a 1×64×64 image. The discriminator is a convolutional classifier trained to tell real MNIST digits from generated ones. Both networks are trained together using binary cross-entropy and the two-player minimax objective.

## Architecture

### Generator

| Layer | Operation | Output shape |
| ----- | --------- | ------------ |
| Input | Latent vector `z` | 100×1×1 |
| 1 | `ConvTranspose2d(100 → 512, 4×4, stride 1)` + BatchNorm + ReLU | 512×4×4 |
| 2 | `ConvTranspose2d(512 → 256, 4×4, stride 2, pad 1)` + BatchNorm + ReLU | 256×8×8 |
| 3 | `ConvTranspose2d(256 → 128, 4×4, stride 2, pad 1)` + BatchNorm + ReLU | 128×16×16 |
| 4 | `ConvTranspose2d(128 → 64, 4×4, stride 2, pad 1)` + BatchNorm + ReLU | 64×32×32 |
| Output | `ConvTranspose2d(64 → 1, 4×4, stride 2, pad 1)` + Tanh | 1×64×64 |

### Discriminator

| Layer | Operation | Output shape |
| ----- | --------- | ------------ |
| Input | Image | 1×64×64 |
| 1 | `Conv2d(1 → 64, 4×4, stride 2, pad 1)` + LeakyReLU(0.2) | 64×32×32 |
| 2 | `Conv2d(64 → 128, 4×4, stride 2, pad 1)` + BatchNorm + LeakyReLU(0.2) | 128×16×16 |
| 3 | `Conv2d(128 → 256, 4×4, stride 2, pad 1)` + BatchNorm + LeakyReLU(0.2) | 256×8×8 |
| 4 | `Conv2d(256 → 512, 4×4, stride 2, pad 1)` + BatchNorm + LeakyReLU(0.2) | 512×4×4 |
| Output | `Conv2d(512 → 1, 4×4, stride 1)` + Sigmoid | scalar probability |

> A `.pth` file stores weights only, not the model shape. `model.py` mirrors the notebook architecture exactly, so keep them in sync if you change the networks.

## Project structure

```text
GAN/
├── 04_GAN_implementation.ipynb   # Training notebook (MNIST + DCGAN)
├── model.py                      # Generator, Discriminator, load_weights
├── inferance.py                  # Inference / sampling script
├── generator.pth                 # Trained generator weights (~14 MB)
├── discriminator.pth             # Trained discriminator weights (~11 MB)
├── weights/                      # Checkpoint layout expected by inferance.py
│   ├── generator.pth
│   └── discriminator.pth
├── output/                       # Generated grids (created at runtime)
├── dcgan_training.gif            # Training progress animation
├── pyproject.toml                # Project metadata and dependencies (uv)
├── uv.lock                       # Locked dependency versions
└── README.md
```

## Getting started

### Prerequisites

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) (recommended) or a plain Python environment with `pip`

### Installation

```bash
git clone https://github.com/Irfan140/GAN.git
cd GAN
uv sync
```

`uv run` will also install dependencies automatically on first use.

## Generating images

`inferance.py` loads the generator, samples 16 random latent vectors (seed 42), saves them as a 4×4 grid, and prints the discriminator's mean score for the fakes.

The script reads the checkpoints from `weights/` and creates `output/` itself, so only the weights need to be put in place:

**PowerShell (Windows)**

```powershell
New-Item -ItemType Directory -Force weights | Out-Null
Copy-Item generator.pth, discriminator.pth weights\
uv run inferance.py
```

**bash (Linux / macOS / Git Bash)**

```bash
mkdir -p weights
cp generator.pth discriminator.pth weights/
uv run inferance.py
```

> `inferance.py` resolves `WEIGHTS_DIR` and `OUTPUT_DIR` relative to its own location, so it works regardless of the directory you launch it from. Point `WEIGHTS_DIR` elsewhere if your checkpoints live outside `weights/`.

Output:

```text
Generator loaded
Discriminator loaded
Discriminator score for fake images (0 = fake, 1 = real):
Mean score: 0.xxxx
Saved to <repo>/output/generated.png
```

The generated grid is written to `output/generated.png` — the folder is created automatically on first run — and also displayed with Matplotlib. Inference runs on CPU, so no GPU is required.

## Training

Open `04_GAN_implementation.ipynb` in Jupyter, JupyterLab, or Google Colab and run all cells. MNIST is downloaded automatically to `./data`.

The notebook was trained on a Colab T4 GPU. `CUDA` is set to `False` in the configuration cell by default; set it to `True` (together with a CUDA-enabled PyTorch install) to train on GPU.

### Training hyperparameters

| Parameter | Value |
| --------- | ----- |
| Dataset | MNIST |
| Image size | 64×64, grayscale |
| Batch size | 128 |
| Latent dimension (`Z_DIM`) | 100 |
| Generator / discriminator features | 64 |
| Epochs | 5 |
| Optimizer | Adam (`lr=2e-4`, `betas=(0.5, 0.999)`) |
| Loss | Binary cross-entropy |
| Real / fake labels | 1 / 0 |
| Weight init | Conv: `N(0, 0.02)`, BatchNorm: `N(1, 0.02)`, bias `0` |
| Seed | 1 |

During training the notebook logs `Loss_D`, `Loss_G`, `D(x)`, and `D(G(z))` every 50 iterations, stores a fixed-noise sample grid every 500 iterations, and plots real vs. generated images at the end.

### Recreating the GIF

The **Save the progress as GIF** cell assembles the stored sample grids into `dcgan_training.gif` using Matplotlib's `ArtistAnimation`. It requires **ImageMagick** to be installed and on `PATH` because it uses the `imagemagick` writer. Change the writer to `pillow` if you prefer not to install ImageMagick.

## Notes

- The trained checkpoints are tied to the notebook's architecture and 64×64 input size.
- `load_weights()` in `model.py` tolerates both plain state dicts and checkpoints saved with a `DataParallel` `module.` prefix.
- Generated images use `Tanh` output in the range `[-1, 1]`; they are normalized for display and saving.
- `inferance.py` builds its paths from `BASE_DIR = Path(__file__).resolve().parent` and calls `OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` before saving, so it never fails on a missing `output/` folder.

## Author

**Irfan Mehmud** — [@Irfan140](https://github.com/Irfan140)
