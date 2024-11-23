from efficientnet_pytorch import EfficientNet
import torch.nn as nn

# Global variable to store the model instance
_model_instance = None

def setup_efficientnet(num_classes, force_new=False):
    global _model_instance
    
    # Return existing instance if available and not forced to create new
    if _model_instance is not None and not force_new:
        return _model_instance
    
    # Create new model instance
    model = EfficientNet.from_pretrained('efficientnet-b0')
    
    # Modify classifier for our number of classes
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    
    # Store the instance
    _model_instance = model
    
    return model