import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os
import random

class Discriminator(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # input size: 256 x 256
            *discriminator_block(3, channels, normalize=False),  # 128 x 128
            *discriminator_block(channels, channels * 2),        # 64 x 64
            *discriminator_block(channels * 2, channels * 4),    # 32 x 32
            *discriminator_block(channels * 4, channels * 8),    # 16 x 16
            *discriminator_block(channels * 8, channels * 16),   # 8 x 8
            nn.Conv2d(channels * 16, 1, 4, padding=0),          # 5 x 5
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(validity.size(0), -1).mean(1)

class Generator(nn.Module):
    def __init__(self, latent_dim=512, channels=32):
        super().__init__()
        
        self.init_size = 8
        self.l1 = nn.Linear(latent_dim, channels * 16 * self.init_size ** 2)

        def generator_block(in_filters, out_filters):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            # input size: 8 x 8
            generator_block(channels * 16, channels * 8),    # 16 x 16
            generator_block(channels * 8, channels * 4),     # 32 x 32
            generator_block(channels * 4, channels * 2),     # 64 x 64
            generator_block(channels * 2, channels),         # 128 x 128
            generator_block(channels, channels),             # 256 x 256
            nn.Conv2d(channels, 3, 3, stride=1, padding=1), # 256 x 256
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.model(out)
        return img

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_gan(dataloader, num_epochs=100, latent_dim=512, device="cuda"):
    # Initialize generator and discriminator
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Create directories for saving samples
    os.makedirs("images", exist_ok=True)

    # Training loop
    print("Starting training...")

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Configure input
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)

            # Adversarial ground truths
            valid = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(batch_size, latent_dim).to(device)

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] "
                    f"[Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] "
                    f"[G loss: {g_loss.item():.4f}]"
                )

                if i % 500 == 0:
                    # Save sample images
                    save_image(
                        gen_imgs.data[:25],
                        f"images/epoch_{epoch}_batch_{i}.png",
                        nrow=5,
                        normalize=True,
                    )

    return generator, discriminator

def main():
    # Hyperparameters
    latent_dim = 512
    batch_size = 16
    image_size = 256
    num_epochs = 200

    # Configure data loader
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root="data_cr", transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    generator, discriminator = train_gan(
        dataloader=dataloader,
        num_epochs=num_epochs,
        latent_dim=latent_dim,
        device=device
    )

    # Save the models
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Training finished! Models saved.")

if __name__ == "__main__":
    main()