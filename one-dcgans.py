import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from torchvision.utils import save_image
import time
import os
from torchvision.datasets import ImageFolder

# Data preprocessing
class DataPreprocessing:
    def __init__(self, image_size=64): # Changed image size to 244x244
        print("\n[INFO] Initializing Data Preprocessing...")
        self.image_size = image_size
        
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self, data_path, batch_size=32):
        print(f"\n[INFO] Loading dataset from: {data_path}")
        
        if not os.path.exists(data_path):
            raise RuntimeError(f"Dataset path {data_path} does not exist")
        
        # Load the entire dataset (without train/test split)
        dataset = ImageFolder(root=data_path, transform=self.transforms)
        print(f"[INFO] Found {len(dataset)} images")
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Print class distribution
        print("\n[INFO] Dataset Class Distribution:")
        for idx, class_name in enumerate(dataset.classes):
            n_samples = len([x for x, y in dataset.samples if y == idx])
            print(f"Class {idx}: {class_name} - {n_samples} images")
            
        return data_loader

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).squeeze(1)

def train_dcgan(data_loader, device, nz=100, num_epochs=325):
    # Create the generator and discriminator
    netG = Generator(latent_dim=nz).to(device)
    netD = Discriminator().to(device)
    
    # Initialize MSELoss function
    criterion = nn.MSELoss()

    # Create batch of latent vectors for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup optimizers
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.00005, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.00005, betas=(0.5, 0.999))
    
    # Training Loop
    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for i, data in enumerate(data_loader, 0):
            batch_size = data[0].size(0)
            
            # Train discriminator
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)
            
            real_cpu = data[0].to(device)
            optimizerD.zero_grad()
            real_output = netD(real_cpu)
            d_real_loss = criterion(real_output, real_labels)
            print(f"Discriminator real loss: {d_real_loss.item():.4f}")
            
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            fake_output = netD(fake.detach())
            d_fake_loss = criterion(fake_output, fake_labels)
            print(f"Discriminator fake loss: {d_fake_loss.item():.4f}")
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizerD.step()
            print(f"Discriminator total loss: {d_loss.item():.4f}")
            
            # Train generator
            optimizerG.zero_grad()
            fake_output = netD(fake)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizerG.step()
            print(f"Generator loss: {g_loss.item():.4f}")
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}')
        
        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise)
                save_image(fake.detach(),
                         f'generated_images_blight/fake_samples_epoch_{epoch+1}.png',
                         normalize=True)
                print(f"Saved generated images for epoch {epoch+1}")
    
    return netG

def main():
    # Create output directory
    os.makedirs("generated_images_blight", exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Initialize data preprocessing
    preprocessor = DataPreprocessing()
    data_loader = preprocessor.load_data('./data_blight')
    
    # Train DCGAN
    print("\n[INFO] Training DCGAN for data augmentation...")
    generator = train_dcgan(data_loader, device)
    
    print("[INFO] Training completed")

if __name__ == '__main__':
    main()