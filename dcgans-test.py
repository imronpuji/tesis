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
    def __init__(self, image_size=64): # Changed image size to 64x64 for DCGAN
        print("\n[INFO] Initializing Data Preprocessing...")
        self.image_size = image_size
        
        self.train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self, data_path, batch_size=32):
        print(f"\n[INFO] Loading dataset from: {data_path}")
        
        if not os.path.exists(data_path):
            raise RuntimeError(f"Dataset path {data_path} does not exist")
            
        train_path = os.path.join(data_path, 'train')
        if not os.path.exists(train_path):
            raise RuntimeError(f"Training data path {train_path} does not exist")
            
        train_dataset = ImageFolder(root=train_path, transform=self.train_transforms)
        print(f"[INFO] Found {len(train_dataset)} training images")
        
        test_path = os.path.join(data_path, 'test')
        if not os.path.exists(test_path):
            raise RuntimeError(f"Test data path {test_path} does not exist")
            
        test_dataset = ImageFolder(root=test_path, transform=self.test_transforms)
        print(f"[INFO] Found {len(test_dataset)} test images")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print("\n[INFO] Dataset Class Distribution:")
        for idx, class_name in enumerate(train_dataset.classes):
            n_samples = len([x for x, y in train_dataset.samples if y == idx])
            print(f"Class {idx}: {class_name} - {n_samples} images")
            
        return train_loader, test_loader

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def train_dcgan(train_loader, device, nz=100, num_epochs=50):
    # Create the generator and discriminator
    netG = Generator(latent_dim=nz).to(device)
    netD = Discriminator().to(device)
    
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup optimizers
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Training Loop
    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            # Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(1)  
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')
                
            if (epoch % 5 == 0) and (i == 0):
                with torch.no_grad():
                    fake = netG(fixed_noise)
                    save_image(fake.detach(),
                             f'generated_images/fake_samples_epoch_{epoch}.png',
                             normalize=True)
    
    return netG

def main():
    # Create output directory
    os.makedirs("generated_images", exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Initialize data preprocessing
    preprocessor = DataPreprocessing()
    train_loader, test_loader = preprocessor.load_data('./dataset')
    
    # Train DCGAN
    print("\n[INFO] Training DCGAN for data augmentation...")
    generator = train_dcgan(train_loader, device)
    
    print("[INFO] Training completed")

if __name__ == '__main__':
    main()