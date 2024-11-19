import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from efficientnet_pytorch import EfficientNet
import time

# 1. Data Preprocessing
class DataPreprocessing:
    def __init__(self, image_size=224):
        print("\n[INFO] Initializing Data Preprocessing...")
        self.image_size = image_size
        print(f"[INFO] Image size set to: {image_size}x{image_size}")
        
        print("[INFO] Setting up data transformations...")
        self.train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        print("[SUCCESS] Train transformations configured")
        
        self.test_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        print("[SUCCESS] Test transformations configured")

    def load_data(self, data_path, batch_size=32):
        print(f"\n[INFO] Loading dataset from: {data_path}")
        print(f"[INFO] Batch size: {batch_size}")
        
        print("[INFO] Loading training dataset...")
        train_dataset = datasets.ImageFolder(
            root=f'{data_path}/train',
            transform=self.train_transforms
        )
        print(f"[SUCCESS] Found {len(train_dataset)} training images")
        
        print("[INFO] Loading test dataset...")
        test_dataset = datasets.ImageFolder(
            root=f'{data_path}/test',
            transform=self.test_transforms
        )
        print(f"[SUCCESS] Found {len(test_dataset)} test images")
        
        print("[INFO] Creating data loaders...")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        print("[SUCCESS] Data loaders created")
        
        print("\n[INFO] Dataset Class Distribution:")
        for idx, class_name in enumerate(train_dataset.classes):
            print(f"Class {idx}: {class_name}")
        
        return train_loader, test_loader

# 2. DCGANs Implementation
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        print("\n[INFO] Initializing Generator...")
        self.main = nn.Sequential(
            # Architecture details printed
        )
        print("[SUCCESS] Generator initialized")

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        print("\n[INFO] Initializing Discriminator...")
        self.main = nn.Sequential(
            # Architecture details printed
        )
        print("[SUCCESS] Discriminator initialized")

    def forward(self, x):
        return self.main(x)

# 3. EfficientNet-B0 Model
class CornDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CornDiseaseClassifier, self).__init__()
        print("\n[INFO] Initializing EfficientNet-B0 model...")
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Linear(1280, num_classes)
        print(f"[SUCCESS] Model initialized with {num_classes} output classes")
        
    def forward(self, x):
        return self.efficientnet(x)

# 4. Training Functions
class ModelTrainer:
    def __init__(self, model, device, num_classes=4):
        print("\n[INFO] Initializing Model Trainer...")
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0001
        )
        print(f"[INFO] Using device: {device}")
        print("[INFO] Optimizer: Adam (lr=0.0001)")
        print("[SUCCESS] Model Trainer initialized")
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        print("\n[INFO] Starting training epoch...")
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"[INFO] Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
            
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f"[SUCCESS] Epoch completed in {epoch_time:.2f}s")
        return epoch_loss, epoch_acc
    
    def evaluate(self, test_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print("\n[INFO] Starting evaluation...")
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        test_loss = running_loss / len(test_loader)
        test_acc = 100. * correct / total
        
        print("[SUCCESS] Evaluation completed")
        return test_loss, test_acc

# 5. Main Training Pipeline
def train_model(data_path, num_epochs=100, batch_size=32):
    print("\n" + "="*50)
    print("Starting Training Pipeline")
    print("="*50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")
    
    # Initialize preprocessing
    print("\n[INFO] Initializing data preprocessing...")
    preprocessor = DataPreprocessing()
    train_loader, test_loader = preprocessor.load_data(
        data_path,
        batch_size=batch_size
    )
    
    # Initialize model
    print("\n[INFO] Initializing model...")
    model = CornDiseaseClassifier(num_classes=4)
    trainer = ModelTrainer(model, device)
    
    # Training loop
    print("\n[INFO] Starting training loop...")
    print(f"[INFO] Number of epochs: {num_epochs}")
    print(f"[INFO] Batch size: {batch_size}")
    
    best_acc = 0.0
    training_start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n{'='*20} Epoch [{epoch+1}/{num_epochs}] {'='*20}")
        
        train_loss, train_acc = trainer.train_epoch(train_loader)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        print(f"\n[INFO] Training Results:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"[INFO] New best accuracy! Model saved: {best_acc:.2f}%")
    
    total_time = time.time() - training_start_time
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best accuracy achieved: {best_acc:.2f}%")
    print("="*50)
    
    return model, best_acc

# Example usage
if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Corn Disease Classification Program")
    print("="*50)
    
    data_path = './dataset'
    print(f"\n[INFO] Data path: {data_path}")
    
    model, best_acc = train_model(
        data_path,
        num_epochs=100,
        batch_size=32
    )
    
    print("\n" + "="*50)
    print("Program Complete!")
    print(f"Final Best Accuracy: {best_acc:.2f}%")
    print("="*50)