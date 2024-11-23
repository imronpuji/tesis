import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg

# Add this class before the FIDScore class
class DataPreprocessing:
    def __init__(self, image_size=64):
        print("\n[INFO] Initializing Data Preprocessing...")
        self.image_size = image_size
        
        # Transformations optimized for GAN training
        self.train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(10),
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
            num_workers=2,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print("\n[INFO] Dataset Class Distribution:")
        class_names = train_dataset.classes
        for idx, class_name in enumerate(class_names):
            n_samples = len([x for x, y in train_dataset.samples if y == idx])
            print(f"Class {idx}: {class_name} - {n_samples} images")
            
        return train_loader, test_loader, len(train_dataset.classes)
# Add FID Score Calculator
# This is the class that needs to be fixed
class FIDScore:
    def __init__(self, device):
        self.device = device
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        self.inception_model.fc = nn.Identity()  # Remove final fc layer
        
        # Add resize transform for FID calculation
        self.resize_transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception v3 requires 299x299 input
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def calculate_activation_statistics(self, images):
        # Ensure we have a batch of images
        if images.shape[0] == 0:
            raise ValueError("Empty batch of images")
            
        # Print shape for debugging
        print(f"Input image shape: {images.shape}")
        
        # Resize images to 299x299 for Inception v3
        images = self.resize_transform(images)
        print(f"Resized image shape: {images.shape}")
        
        with torch.no_grad():
            try:
                features = self.inception_model(images)
                # Handle the case where inception returns a tuple
                if isinstance(features, tuple):
                    features = features[0]
                
                print(f"Features shape before reshape: {features.shape}")
                
                # Reshape features appropriately
                features = features.view(features.size(0), -1)
                print(f"Features shape after reshape: {features.shape}")
                
                # Convert to numpy
                features = features.cpu().numpy()
                
                # Calculate statistics
                mu = np.mean(features, axis=0)
                sigma = np.cov(features, rowvar=False)
                
                return mu, sigma
                
            except Exception as e:
                print(f"Error in inception model processing: {e}")
                print(f"Input tensor shape: {images.shape}")
                raise e

    def calculate_fid(self, real_images, fake_images):
        # Print shapes for debugging
        print(f"Real images shape: {real_images.shape}")
        print(f"Fake images shape: {fake_images.shape}")
        
        try:
            mu1, sigma1 = self.calculate_activation_statistics(real_images)
            mu2, sigma2 = self.calculate_activation_statistics(fake_images)
            
            ssdiff = np.sum((mu1 - mu2)**2.0)
            covmean = linalg.sqrtm(sigma1.dot(sigma2))
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            fid_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
            return fid_score
            
        except Exception as e:
            print(f"Error in FID calculation: {e}")
            raise e
# Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_fid = None
        self.early_stop = False

    def __call__(self, fid_score):
        if self.min_fid is None:
            self.min_fid = fid_score
        elif fid_score > self.min_fid - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.min_fid = fid_score
            self.counter = 0
        return self.early_stop

# Enhanced Generator with more BatchNorm
class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=3):
        super(Generator, self).__init__()
        
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        self.init_size = 64 // 4
        
        # Enhanced linear layer with BatchNorm
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2),
            nn.BatchNorm1d(128 * self.init_size ** 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Enhanced convolutional blocks with more BatchNorm
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Enhanced Discriminator with Gradient Penalty
class Discriminator(nn.Module):
    def __init__(self, n_classes=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, dropout=0.3):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        ds_size = 64 // 2 ** 5
        self.adv_layer = nn.Sequential(
            nn.Linear(256 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(256 * ds_size ** 2, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)[0]
    fake = torch.ones(real_samples.size(0), 1).to(device)
    
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

def train_cgan(train_loader, device, n_classes, latent_dim=100, n_epochs=300):
    # Initialize models
    generator = Generator(latent_dim, n_classes).to(device)
    discriminator = Discriminator(n_classes).to(device)
    
    # Initialize FID calculator and early stopping
    fid_calculator = FIDScore(device)
    early_stopping = EarlyStopping(patience=15)
    
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    
    # Optimizers with better learning rates
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', patience=5)
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', patience=5)
    
    # Create directories for samples
    os.makedirs("cgan_images", exist_ok=True)
    
    # Training loop
    for epoch in range(n_epochs):
        for i, (real_imgs, labels) in enumerate(train_loader):
            batch_size = real_imgs.shape[0]
            
            # Configure input
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            # Ground truths with label smoothing
            valid = torch.ones(batch_size, 1).to(device) * 0.9
            fake = torch.zeros(batch_size, 1).to(device) + 0.1
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate images
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, n_classes, (batch_size,)).to(device)
            gen_imgs = generator(z, gen_labels)
            
            # Generator loss
            validity, pred_label = discriminator(gen_imgs)
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real loss
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = 0.5 * (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels))
            
            # Fake loss
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = 0.5 * (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels))
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, gen_imgs.detach(), device)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss + 10 * gradient_penalty
            
            d_loss.backward()
            optimizer_D.step()
            
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )
                
        # Calculate FID score and check for early stopping
        if epoch % 10 == 0:
            with torch.no_grad():
                fake_images = generator(
                    torch.randn(100, latent_dim).to(device),
                    torch.randint(0, n_classes, (100,)).to(device)
                )
                fid_score = fid_calculator.calculate_fid(real_imgs[:100], fake_images)
                print(f"FID Score: {fid_score}")
                
                if early_stopping(fid_score):
                    print("Early stopping triggered")
                    break
                
                # Save generated images
                save_image(fake_images[:25], f"cgan_images/epoch_{epoch}.png", nrow=5, normalize=True)
        
        # Update learning rates
        scheduler_G.step(g_loss)
        scheduler_D.step(d_loss)
    
    return generator

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    latent_dim = 100
    n_epochs = 1000
    batch_size = 32
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Initialize data preprocessing and load data
    preprocessor = DataPreprocessing()  # Assuming you have this class from previous code
    train_loader, test_loader, n_classes = preprocessor.load_data('./dataset', batch_size)
    
    # Train CGAN
    print("\n[INFO] Training CGAN for data augmentation...")
    generator = train_cgan(train_loader, device, n_classes, latent_dim, n_epochs)
    
    print("[INFO] Training completed")

if __name__ == '__main__':
    main()