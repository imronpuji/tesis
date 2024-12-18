import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import os
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, resnet50, efficientnet_b0

class DataPreprocessing:
    def __init__(self, image_size=224):
        print("\n[INFO] Initializing Data Preprocessing...")
        self.image_size = image_size
    
        # Perkuat augmentasi untuk dataset kecil
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5), 
            transforms.RandomRotation(30),  # Perbesar rotasi
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Tambah translasi
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2, 
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self, data_path, batch_size=32, val_split=0.2):
        print(f"\n[INFO] Loading dataset from: {data_path}")
        
        if not os.path.exists(data_path):
            raise RuntimeError(f"Dataset path {data_path} does not exist")
        
        dataset = ImageFolder(root=data_path, transform=self.transforms)
        print(f"[INFO] Found {len(dataset)} images")
        
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print("\n[INFO] Dataset Class Distribution:")
        for idx, class_name in enumerate(dataset.classes):
            n_samples = len([x for x, y in dataset.samples if y == idx])
            print(f"Class {idx}: {class_name} - {n_samples} images")
            
        return train_loader, val_loader

class Generator(nn.Module):
    def __init__(self, backbone, latent_dim=100, output_size=224):
        super(Generator, self).__init__()
        self.backbone = backbone
        self.init_size = output_size // 16
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, backbone, input_size=224):
        super(Discriminator, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.input_size = input_size
        
        # Freeze the pre-trained backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.adv_layer = nn.Sequential(
            nn.Linear(self.get_num_features(), 1),
            nn.Sigmoid()
        )

    def get_num_features(self):
        # Use the forward method to get the number of features
        x = torch.randn(1, 3, self.input_size, self.input_size)
        features = self.backbone(x)
        return features.size(1)

    def forward(self, img):
        out = self.backbone(img)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity
        
def train_dcgan(train_loader, val_loader, device, latent_dim=100, num_epochs=200):
    # Load pre-trained backbones
    vgg = vgg16(pretrained=True).features.to(device)
    resnet = resnet50(pretrained=True).to(device)
    efficient = efficientnet_b0(pretrained=True).to(device)

    # Freeze the pre-trained backbones
    for param in vgg.parameters():
        param.requires_grad = False
    for param in resnet.parameters():
        param.requires_grad = False
    for param in efficient.parameters():
        param.requires_grad = False

    # Create DCGAN models with pre-trained backbones
    netG = Generator(efficient, latent_dim=latent_dim).to(device)
    netD = Discriminator(resnet).to(device)
    
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, latent_dim, device=device)

    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.1)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.1)

    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(train_loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Train Discriminator
            optimizerD.zero_grad()
            
            # Real images
            real_validity = netD(real_imgs)
            real_loss = criterion(real_validity, torch.ones_like(real_validity))
            
            # Fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = netG(z)
            fake_validity = netD(fake_imgs.detach())
            fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizerD.step()
            
            # Train Generator
            optimizerG.zero_grad()
            
            gen_imgs = netG(z)
            validity = netD(gen_imgs)
            g_loss = criterion(validity, torch.ones_like(validity))
            
            g_loss.backward()
            optimizerG.step()
            
            if (i+1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}')
        
        schedulerD.step()
        schedulerG.step()
        
        # Validation
        val_loss = 0
        netD.eval()
        netG.eval()
        with torch.no_grad():
            for real_imgs, _ in val_loader:
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(device)
                
                real_validity = netD(real_imgs)
                real_loss = criterion(real_validity, torch.ones_like(real_validity))
                
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_imgs = netG(z)
                fake_validity = netD(fake_imgs)
                fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))
                
                val_loss += (real_loss + fake_loss).item()
        
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
        
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                save_image(fake, f'generated_images_glsf/fake_samples_epoch_{epoch+1}.png', normalize=True)
    
    return netG

def main():
    os.makedirs("generated_images_glsf", exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    preprocessor = DataPreprocessing(image_size=224)
    train_loader, val_loader = preprocessor.load_data('./data')
    
    print("\n[INFO] Training DCGAN for data augmentation...")
    generator = train_dcgan(train_loader, val_loader, device, num_epochs=200)
    
    print("[INFO] Training completed")

if __name__ == '__main__':
    main()