# DCGAN training entry point. The implementation lives in the dcgan/ package.
# Local:  uv run train.py --epochs 5
# Kaggle: kaggle kernels push -p kaggle   (see README, "Training on Kaggle")
# Output: weights/*.pth, checkpoints/*.pt, output/training/*

from dcgan.cli import main

if __name__ == "__main__":
    main()
