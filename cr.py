import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torchvision.datasets import ImageFolder
from torch.nn.utils import spectral_norm

class DataPreprocessing:
    def __init__(self, image_size=64):
        print("\n[INFO] Initializing Data Preprocessing...")
        self.image_size = image_size
        
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self, data_path, batch_size=32):
        print(f"\n[INFO] Loading dataset from: {data_path}")
        dataset = ImageFolder(root=data_path, transform=self.transforms)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        return data_loader

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)  # Flatten to [batch_size]

def train_dcgan(data_loader, device, nz=100, num_epochs=300):
    # Models
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)
    
    # Loss and Optimizers
    criterion = nn.BCELoss()
    
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, num_epochs)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, num_epochs)
    
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    real_label = 0.9  # Label smoothing
    fake_label = 0.0

    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        for i, (real_data, _) in enumerate(data_loader):
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            
            # Train with real
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            
            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(data_loader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

        # Update learning rates
        schedulerD.step()
        schedulerG.step()

        # Save images
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise)
                save_image(fake.detach(),
                         f'generated_images_cr/fake_samples_epoch_{epoch+1}.png',
                         normalize=True,
                         nrow=8)

    return netG

def main():
    # Create output directory
    os.makedirs("generated_images_cr", exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Initialize data preprocessing
    preprocessor = DataPreprocessing()
    data_loader = preprocessor.load_data('./data_cr')
    
    # Train DCGAN
    print("\n[INFO] Training DCGAN...")
    generator = train_dcgan(data_loader, device)
    
    print("[INFO] Training completed")

if __name__ == '__main__':
    main()