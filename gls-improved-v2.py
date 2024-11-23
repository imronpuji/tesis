import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torchvision.datasets import ImageFolder
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR  # Added this import

class DataPreprocessing:
    def __init__(self, image_size=64):
        print("\n[INFO] Initializing Data Preprocessing...")
        self.image_size = image_size
        
        self.transforms = transforms.Compose([
            transforms.Resize((image_size + 8, image_size + 8)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation(15),
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self, data_path, batch_size=16):
        dataset = ImageFolder(root=data_path, transform=self.transforms)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            pin_memory=True
        )
        return data_loader

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        std = torch.std(x, dim=0, keepdim=True)
        mean_std = std.mean()
        return torch.cat((x, mean_std.expand(x.shape[0], 1, x.shape[2], x.shape[3])), dim=1)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.2),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.2),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.2),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            MinibatchStdDev(),
            nn.Conv2d(ndf * 8 + 1, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_dcgan(data_loader, device, nz=100, num_epochs=400):
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    
    # Base learning rates
    lr_d = 0.0002
    lr_g = 0.0002
    
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.0, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.0, 0.999))
    
    try:
        # Try to use CosineAnnealingLR
        schedulerD = CosineAnnealingLR(optimizerD, num_epochs, eta_min=1e-5)
        schedulerG = CosineAnnealingLR(optimizerG, num_epochs, eta_min=1e-5)
        use_scheduler = True
    except:
        print("Warning: CosineAnnealingLR not available, using manual LR decay")
        use_scheduler = False
    
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        for i, (real_data, _) in enumerate(data_loader):
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Dynamic label smoothing
            real_target = 0.7 + 0.3 * np.random.random()
            fake_target = 0.0 + 0.2 * np.random.random()
            
            label_real = torch.full((batch_size,), real_target, device=device)
            label_fake = torch.full((batch_size,), fake_target, device=device)
            
            # Add noise to real images
            real_data = real_data + 0.05 * torch.randn_like(real_data)
            
            output_real = netD(real_data)
            errD_real = criterion(output_real, label_real)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Generate fake images
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            
            output_fake = netD(fake.detach())
            errD_fake = criterion(output_fake, label_fake)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            
            # Compute gradient penalty
            if epoch > 50:
                gp = compute_gradient_penalty(netD, real_data, fake.detach(), device)
                gp_loss = 10.0 * gp
                gp_loss.backward()
            
            errD = errD_real + errD_fake
            if errD.item() > 0.2:
                optimizerD.step()

            ############################
            # (2) Update G network
            ############################
            if i % 2 == 0:
                netG.zero_grad()
                output = netD(fake)
                errG = criterion(output, label_real)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(data_loader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

        # Update learning rates
        if use_scheduler:
            schedulerD.step()
            schedulerG.step()
        else:
            # Manual learning rate decay
            if (epoch + 1) % 100 == 0:
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.8
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.8

        # Save images
        save_freq = 1 if epoch < 50 else 5
        if (epoch + 1) % save_freq == 0:
            with torch.no_grad():
                fake = netG(fixed_noise)
                save_image(fake.detach(),
                         f'generated_images/fake_samples_epoch_{epoch+1}.png',
                         normalize=True,
                         nrow=8)

    return netG

def main():
    os.makedirs("generated_images", exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    preprocessor = DataPreprocessing()
    data_loader = preprocessor.load_data('./data')
    
    print("\n[INFO] Training DCGAN...")
    generator = train_dcgan(data_loader, device)
    
    print("[INFO] Training completed")

if __name__ == '__main__':
    main()