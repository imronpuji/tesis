import sys
import os
import torch
import numpy as np
from datetime import datetime

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from dataset import prepare_data_loaders
from models.efficientnet import setup_efficientnet
from models.dcgan import Generator, Discriminator
from training import train_model
from evaluation import (
    evaluate_model, 
    create_evaluation_visualizations,
    plot_training_history
)
import json

def main():
    # Set device and random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Using device: {device}")
    
    # Create directories
    create_directories()
    
    # Define class names
    class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    
    # Prepare data
    print("Loading data...")
    train_loader, val_loader, test_loader = prepare_data_loaders(BASE_DIR)
    
    # Setup model
    print("Setting up model...")
    model = setup_efficientnet(NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("Starting training...")
    start_time = datetime.now()
    
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_dir=f'{RESULTS_DIR}/model_training'
    )
    
    training_time = datetime.now() - start_time
    print(f"Training completed in {training_time}")
    
    # Plot training history
    plot_training_history(history, f'{RESULTS_DIR}/model_training')
    
    # Evaluate model
    # In main.py
    pred_labels, true_labels = evaluate_model(model, test_loader, device)
    unique_classes = len(np.unique(true_labels))
    class_names = ['Class_0', 'Class_1', 'Class_2', 'Class_3'] # Generic names based on number of unique classes

    results, metrics = create_evaluation_visualizations(
        true_labels=true_labels,
        pred_labels=pred_labels,
        class_names=class_names,
        save_dir=f'{RESULTS_DIR}/model_evaluation'
    )
    # Save training history
    with open(f'{RESULTS_DIR}/model_training/history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Print final results
    print("\nFinal Results:")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")
    
    # Save execution information
    execution_info = {
        'training_time': str(training_time),
        'device': str(device),
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'final_metrics': metrics
    }
    
    with open(f'{RESULTS_DIR}/execution_info.json', 'w') as f:
        json.dump(execution_info, f, indent=4)

if __name__ == "__main__":
    main()