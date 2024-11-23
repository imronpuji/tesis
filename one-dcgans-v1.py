import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from torchvision.utils import save_image
import time
import os
from torchvision.datasets import ImageFolder

class DataPreprocessing:
    def __init__(self, image_size=244):
        print("\n[INFO] Initializing Data Preprocessing...")
        print(f"[INFO] Image size set to: {image_size}x{image_size}")
        self.image_size = image_size
        
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        print("[INFO] Data transforms initialized")

    def load_data(self, data_path, batch_size=32):
        print(f"\n[INFO] Loading dataset from: {data_path}")
        print(f"[INFO] Batch size: {batch_size}")
        
        if not os.path.exists(data_path):
            raise RuntimeError(f"Dataset path {data_path} does not exist")
        
        dataset = ImageFolder(root=data_path, transform=self.transforms)
        print(f"[INFO] Total number of images: {len(dataset)}")
        
        # Create separate dataloaders for each class
        class_dataloaders = {}
        print("\n[INFO] Creating dataloaders for each class:")
        for class_idx, class_name in enumerate(dataset.classes):
            # Get indices for current class
            indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
            class_dataset = torch.utils.data.Subset(dataset, indices)
            
            class_dataloaders[class_name] = DataLoader(
                class_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            
            print(f"[INFO] Class {class_idx}: {class_name} - {len(indices)} images")
            
        return class_dataloaders, dataset.classes

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        print(f"\n[INFO] Initializing Generator with latent dim: {latent_dim}")
        
        # Menghitung initial size yang benar
        self.init_size = 244 // 32  # Mengubah pembagi menjadi 32
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=4),  # Mengubah scale factor
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=4),  # Mengubah scale factor
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )
        print("[INFO] Generator architecture initialized")

    def forward(self, z):
        # Reshape input properly
        z = z.view(z.size(0), -1)
        out = self.l1(z)
        out = out.view(out.size(0), 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        print("\n[INFO] Initializing Discriminator")

        def discriminator_block(in_filters, out_filters, bn=True, kernel_size=4):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout2d(0.25))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.AdaptiveAvgPool2d(1),  # Menambahkan adaptive pooling
            nn.Flatten(),  # Flatten output
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(512, 1),  # Mengubah input size
            nn.Sigmoid()
        )
        print("[INFO] Discriminator architecture initialized")

    def forward(self, img):
        features = self.model(img)
        validity = self.adv_layer(features)
        return validity

def train_dcgan_by_class(class_dataloaders, classes, device, output_dir, nz=100, num_epochs=200):
    print("\n[INFO] Starting DCGAN training process...")
    print(f"[INFO] {'='*50}")
    print(f"[INFO] Configuration:")
    print(f"[INFO] - Number of epochs: {num_epochs}")
    print(f"[INFO] - Latent dimension: {nz}")
    print(f"[INFO] - Output directory: {output_dir}")
    print(f"[INFO] - Device: {device}")
    print(f"[INFO] {'='*50}\n")
    
    # Create output directories for each class
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
        print(f"[INFO] Created output directory for class: {class_name}")
    
    # Track total progress
    total_classes = len(classes)
    current_class = 0
    
    for class_name, dataloader in class_dataloaders.items():
        current_class += 1
        print(f"\n[INFO] {'='*50}")
        print(f"[INFO] Processing class {current_class}/{total_classes}: {class_name}")
        print(f"[INFO] Number of samples: {len(dataloader.dataset)}")
        print(f"[INFO] Number of batches: {len(dataloader)}")
        print(f"[INFO] {'='*50}\n")
        
        # Initialize networks for this class
        print(f"[INFO] Initializing Generator and Discriminator for {class_name}")
        netG = Generator(latent_dim=nz).to(device)
        netD = Discriminator().to(device)
        print(f"[INFO] Networks moved to device: {device}")
        
        criterion = nn.MSELoss()
        fixed_noise = torch.randn(1, nz, 1, 1, device=device)
        print("[INFO] Created fixed noise for image generation")
        print(f"[DEBUG] Fixed noise shape: {fixed_noise.shape}")
        
        optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        print("[INFO] Initialized optimizers")
        
        print("\n[INFO] Starting training iterations...")
        # Training loop for this class
        total_steps = num_epochs * len(dataloader)
        current_step = 0
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\n[INFO] {'*'*30}")
            print(f"[INFO] Epoch {epoch+1}/{num_epochs}")
            print(f"[INFO] {'*'*30}")
            
            running_d_loss = 0.0
            running_g_loss = 0.0
            
            for i, (real_images, _) in enumerate(dataloader):
                current_step += 1
                batch_size = real_images.size(0)
                
                if i % 10 == 0:
                    print(f"\n[INFO] Batch Progress:")
                    print(f"[INFO] - Current batch: {i+1}/{len(dataloader)}")
                    print(f"[INFO] - Overall progress: {current_step}/{total_steps} steps")
                    print(f"[INFO] - Completion: {(current_step/total_steps)*100:.2f}%")
                    print(f"[DEBUG] Current batch size: {batch_size}")
                
                # Train Discriminator
                print("[DEBUG] Training Discriminator...")
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                
                # Train with real images
                real_images = real_images.to(device)
                print(f"[DEBUG] Real images shape: {real_images.shape}")
                
                optimizerD.zero_grad()
                output_real = netD(real_images)
                print(f"[DEBUG] Real output shape: {output_real.shape}")
                print(f"[DEBUG] Real labels shape: {real_labels.shape}")
                
                d_real_loss = criterion(output_real, real_labels)
                print(f"[DEBUG] D_real_loss: {d_real_loss.item():.4f}")
                
                # Train with fake images
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                print(f"[DEBUG] Noise shape: {noise.shape}")
                
                fake_images = netG(noise)
                print(f"[DEBUG] Generated fake images shape: {fake_images.shape}")
                
                output_fake = netD(fake_images.detach())
                print(f"[DEBUG] Fake output shape: {output_fake.shape}")
                
                d_fake_loss = criterion(output_fake, fake_labels)
                print(f"[DEBUG] D_fake_loss: {d_fake_loss.item():.4f}")
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                optimizerD.step()
                
                # Train Generator
                print("[DEBUG] Training Generator...")
                optimizerG.zero_grad()
                
                output_fake = netD(fake_images)
                g_loss = criterion(output_fake, real_labels)
                print(f"[DEBUG] G_loss: {g_loss.item():.4f}")
                
                g_loss.backward()
                optimizerG.step()
                
                running_d_loss += d_loss.item()
                running_g_loss += g_loss.item()
                
                if (i+1) % 10 == 0:
                    elapsed_time = time.time() - training_start_time
                    eta = (elapsed_time / current_step) * (total_steps - current_step)
                    
                    print(f"\n[INFO] Training Progress:")
                    print(f"[INFO] - Elapsed time: {elapsed_time/3600:.2f} hours")
                    print(f"[INFO] - Estimated time remaining: {eta/3600:.2f} hours")
                    print(f'[INFO] - D_Loss: {d_loss.item():.4f}')
                    print(f'[INFO] - G_Loss: {g_loss.item():.4f}')
            
            # Print epoch statistics
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_d_loss = running_d_loss / len(dataloader)
            epoch_g_loss = running_g_loss / len(dataloader)
            
            print(f'\n[INFO] Epoch [{epoch+1}/{num_epochs}] Summary:')
            print(f'[INFO] - Duration: {epoch_duration:.2f} seconds')
            print(f'[INFO] - Average D_Loss: {epoch_d_loss:.4f}')
            print(f'[INFO] - Average G_Loss: {epoch_g_loss:.4f}')
            
            # Generate and save one image every 5 epochs
            if (epoch+1) % 5 == 0:
                print(f"\n[INFO] {'~'*30}")
                print(f"[INFO] Generating sample image for epoch {epoch+1}")
                
                with torch.no_grad():
                    fake = netG(fixed_noise)
                    save_path = os.path.join(output_dir, class_name, f'generated_{epoch+1}.png')
                    save_image(fake[0], save_path, normalize=True)
                    print(f"[INFO] Saved generated image to: {save_path}")
                    
                print(f"[INFO] {'~'*30}")
        
        # Generate additional images after training
        print(f"\n[INFO] {'='*50}")
        print(f"[INFO] Training completed for class: {class_name}")
        print(f"[INFO] Generating additional images...")
        print(f"[INFO] {'='*50}")
        
        with torch.no_grad():
            for i in range(100):
                noise = torch.randn(1, nz, 1, 1, device=device)
                fake = netG(noise)
                save_path = os.path.join(output_dir, class_name, f'extra_generated_{i+1}.png')
                save_image(fake[0], save_path, normalize=True)
                
                if (i+1) % 10 == 0:
                    print(f"[INFO] Generated {i+1}/100 additional images")

        print(f"\n[INFO] {'='*50}")
        print(f"[INFO] Completed processing class: {class_name}")
        print(f"[INFO] {'='*50}")

    total_training_time = time.time() - training_start_time
    print(f"\n[INFO] {'='*50}")
    print(f"[INFO] Training Complete!")
    print(f"[INFO] Total training time: {total_training_time/3600:.2f} hours")
    print(f"[INFO] {'='*50}")

    return netG
def main():
    print("\n[INFO] {'='*50}")
    print("[INFO] Starting DCGAN Training Pipeline")
    print(f"[INFO] {'='*50}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    print("[INFO] Random seed set to: 42")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Initialize data preprocessing
    print("\n[INFO] Initializing data preprocessing...")
    preprocessor = DataPreprocessing()
    class_dataloaders, classes = preprocessor.load_data('./data')
    print(f"[INFO] Found {len(classes)} classes: {classes}")
    
    # Create output directory
    output_dir = "generated_images_by_class"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Created output directory: {output_dir}")
    
    # Train DCGAN for each class
    print("\n[INFO] Starting DCGAN training for data augmentation...")
    train_dcgan_by_class(class_dataloaders, classes, device, output_dir)
    
    print("\n[INFO] {'='*50}")
    print("[INFO] Training pipeline completed successfully")
    print(f"[INFO] {'='*50}")

if __name__ == '__main__':
    main()