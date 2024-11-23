import os

# Path configurations
BASE_DIR = 'corn_leaf_disease'
RESULTS_DIR = 'results'

# Dataset parameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model parameters
NUM_CLASSES = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 1  # Increased for better training
EARLY_STOPPING_PATIENCE = 10

# DCGAN parameters
LATENT_DIM = 100
GEN_FEATURES = 64
DISC_FEATURES = 64
GAN_LEARNING_RATE = 0.0002
GAN_BETA1 = 0.5
GAN_EPOCHS = 200

# Results structure
RESULTS_STRUCTURE = {
    'dataset_analysis': ['distribution', 'statistics'],
    'preprocessing': ['normalized', 'augmented'],
    'dcgan_results': {
        'training': ['loss_plots', 'metrics'],
        'generated_samples': ['by_epoch', 'final'],
        'fid_scores': ['plots', 'values']
    },
    'augmented_dataset': ['distribution', 'samples'],
    'model_training': {
        'configurations': ['hyperparameters', 'architecture'],
        'checkpoints': ['best', 'periodic'],
        'metrics': ['loss', 'accuracy']
    },
    'model_evaluation': {
        'overall_metrics': ['confusion_matrix', 'roc_curves'],
        'per_class_analysis': ['precision', 'recall', 'f1']
    },
    'comparison_study': {
        'model_comparison': ['metrics', 'plots'],
        'computational_analysis': ['time', 'resources']
    }
}

def create_directories():
    def create_nested_dirs(base_path, structure):
        if isinstance(structure, list):
            for item in structure:
                os.makedirs(os.path.join(base_path, item), exist_ok=True)
        elif isinstance(structure, dict):
            for key, value in structure.items():
                path = os.path.join(base_path, key)
                os.makedirs(path, exist_ok=True)
                create_nested_dirs(path, value)
                
    create_nested_dirs(RESULTS_DIR, RESULTS_STRUCTURE)

def get_path(result_type, subtype=None):
    path = os.path.join(RESULTS_DIR, result_type)
    if subtype:
        path = os.path.join(path, subtype)
    return path