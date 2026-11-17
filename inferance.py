# GAN inference: generate images from the trained generator
# Run with: uv run infer.py
# Output: output/generated.png (16 images, 4x4 grid)

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from model import Generator, Discriminator, load_weights, Z_DIM

# CPU is enough for inference, generating 16 images takes seconds
device = torch.device("cpu")

# 1. Create the generator with the same shape as training
netG = Generator().to(device)
netG = load_weights(netG, "weights/generator.pth", device)
netG.eval()
print("Generator loaded")

# 2. Optional: load the discriminator only to score the fake images
# You do not need it to generate images, this is just a check.
try:
    netD = Discriminator().to(device)
    netD = load_weights(netD, "weights/discriminator.pth", device)
    netD.eval()
    print("Discriminator loaded")
except Exception as e:
    print("Discriminator not loaded, skipping score check:", e)
    netD = None

# 3. Make random noise, same shape as training: (16, 100, 1, 1)
torch.manual_seed(42)
noise = torch.randn(16, Z_DIM, 1, 1, device=device)

# 4. Generate, no gradients needed for inference
with torch.no_grad():
    fake_images = netG(noise)
    if netD is not None:
        scores = netD(fake_images)
        print("Discriminator score for fake images (0 = fake, 1 = real):")
        print("Mean score:", float(scores.mean()))

# 5. Save as one grid image, normalize because output is Tanh (-1 to 1)
vutils.save_image(fake_images, "output/generated.png", nrow=4, normalize=True)
print("Saved to output/generated.png")

# 6. Show it
grid = vutils.make_grid(fake_images, nrow=4, normalize=True).cpu()
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated digits")
plt.imshow(grid[0], cmap="gray")
plt.show()
