import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from itertools import cycle

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_roc_curves(true_labels, pred_probs, class_names, save_dir):
    os.makedirs(f'{save_dir}/roc_curves', exist_ok=True)
    
    # Get actual number of classes from the data
    unique_classes = np.unique(true_labels)
    n_classes = len(unique_classes)
    
    # Create one-hot encoded labels
    y_test = np.zeros((len(true_labels), n_classes))
    for i, label in enumerate(true_labels):
        y_test[i, label] = 1
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/roc_curves/roc_curves.png')
    plt.close()
    
    return roc_auc
def create_evaluation_visualizations(true_labels, pred_labels, class_names, save_dir):
    os.makedirs(f'{save_dir}/confusion_matrix', exist_ok=True)
    os.makedirs(f'{save_dir}/class_metrics', exist_ok=True)
    
    # Print debugging info
    print(f"Unique classes in true_labels: {np.unique(true_labels)}")
    print(f"Class names provided: {class_names}")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix/confusion_matrix.png')
    plt.close()
    
    # Classification report
    report = classification_report(true_labels, pred_labels,
                                 target_names=class_names,
                                 output_dict=True,
                                 zero_division=1)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f'{save_dir}/class_metrics/classification_report.csv')
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, average='weighted'),
        'recall': recall_score(true_labels, pred_labels, average='weighted'),
        'f1': f1_score(true_labels, pred_labels, average='weighted')
    }
    
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return df_report, metrics
    
def plot_training_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'{save_dir}/accuracy_plot.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'{save_dir}/loss_plot.png')
    plt.close()