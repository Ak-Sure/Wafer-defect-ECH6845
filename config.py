"""
Model Configuration File for Wafer Defect Classification

This file contains hyperparameter configurations for all models.
Each model has default parameters that can be overridden.
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_CONFIG = {
    'dataset_path': 'mixedtype-wafer-defect-datasets',
    'image_size': 52,
    'num_classes': 38,
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    'random_seed': 42,
}

# ============================================================================
# TRAINING CONFIGURATION (GLOBAL DEFAULTS)
# ============================================================================
TRAINING_CONFIG = {
    'device': 'cuda',  # 'cuda' or 'cpu'
    'num_epochs': 20,
    'batch_size': 64,
    'learning_rate': 0.001,
    'optimizer': 'adam',  # 'adam' or 'sgd'
    'criterion': 'cross_entropy',
    'early_stopping_patience': 5,
    'verbose': True,
}

# ============================================================================
# MODEL 1: SIMPLE NEURAL NETWORK
# ============================================================================
SIMPLE_NN_CONFIG = {
    'name': 'Simple Neural Network',
    'description': 'Single hidden layer fully connected network',
    'architecture': '2704 -> 38',
    'training': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'optimizer': 'adam',
        'num_epochs': 20,
        'early_stopping_patience': 5,
    },
    'model_kwargs': {},  # No additional kwargs needed
}

# ============================================================================
# MODEL 2: MULTI-LAYER PERCEPTRON
# ============================================================================
MLP_CONFIG = {
    'name': 'Multi-Layer Perceptron',
    'description': 'Fully connected network with multiple hidden layers',
    'architecture': '2704 -> 512 -> 256 -> 128 -> 38',
    'training': {
        'learning_rate': 0.0005,
        'batch_size': 64,
        'optimizer': 'adam',
        'num_epochs': 25,
        'early_stopping_patience': 5,
    },
    'model_kwargs': {},
}

# ============================================================================
# MODEL 3: CONVOLUTIONAL NEURAL NETWORK
# ============================================================================
CNN_CONFIG = {
    'name': 'Convolutional Neural Network',
    'description': 'CNN with 3 convolutional layers and fully connected layers',
    'architecture': 'Conv(32) -> Conv(64) -> Conv(128) -> FC(38)',
    'training': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'optimizer': 'adam',
        'num_epochs': 20,
        'early_stopping_patience': 5,
    },
    'model_kwargs': {},
}

# ============================================================================
# MODEL 4: CNN WITH DATA AUGMENTATION
# ============================================================================
CNN_AUGMENTED_CONFIG = {
    'name': 'CNN with Data Augmentation',
    'description': 'CNN trained with data augmentation techniques',
    'architecture': 'Conv(32) -> Conv(64) -> Conv(128) -> FC(38)',
    'training': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'optimizer': 'adam',
        'num_epochs': 25,
        'early_stopping_patience': 5,
    },
    'model_kwargs': {},
    'augmentation': {
        'rotation_range': 10,
        'zoom_range': 0.1,
        'horizontal_flip': False,
        'vertical_flip': False,
        'elastic_deform': True,
    },
}

# ============================================================================
# MODEL 5: TRANSFER LEARNING (ResNet18)
# ============================================================================
TRANSFER_LEARNING_CONFIG = {
    'name': 'Transfer Learning (ResNet18)',
    'description': 'Pre-trained ResNet18 fine-tuned for wafer defect classification',
    'architecture': 'ResNet18 backbone -> FC(38)',
    'training': {
        'learning_rate': 0.0001,  # Lower learning rate for fine-tuning
        'batch_size': 32,  # Smaller batch size for transfer learning
        'optimizer': 'adam',
        'num_epochs': 15,  # Fewer epochs needed
        'early_stopping_patience': 5,
    },
    'model_kwargs': {
        'pretrained': True,
        'freeze_backbone': False,  # Set to True to freeze pre-trained weights
    },
}

# ============================================================================
# MODEL 6: LIGHTWEIGHT CNN (EDGE DEPLOYMENT)
# ============================================================================
LIGHTWEIGHT_CNN_CONFIG = {
    'name': 'Lightweight CNN',
    'description': 'Compressed CNN for edge deployment',
    'architecture': 'Conv(16) -> Conv(32) -> Conv(64) -> FC(38)',
    'training': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'optimizer': 'adam',
        'num_epochs': 20,
        'early_stopping_patience': 5,
    },
    'model_kwargs': {
        'compression_ratio': 0.5,  # 50% of original filters
    },
}

# ============================================================================
# MODEL REGISTRY
# ============================================================================
# Dictionary mapping model names to their configs
MODEL_CONFIGS = {
    'SimpleNN': SIMPLE_NN_CONFIG,
    'MLP': MLP_CONFIG,
    'CNN': CNN_CONFIG,
    'CNNAugmented': CNN_AUGMENTED_CONFIG,
    'TransferLearning': TRANSFER_LEARNING_CONFIG,
    'LightweightCNN': LIGHTWEIGHT_CNN_CONFIG,
}

# ============================================================================
# HYPERPARAMETER TUNING GRIDS
# ============================================================================
# Define parameter grids for each model's hyperparameter tuning

SIMPLE_NN_TUNING_GRID = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128],
    'optimizer': ['adam', 'sgd'],
}

MLP_TUNING_GRID = {
    'learning_rate': [0.0005, 0.0001, 0.00005],
    'batch_size': [32, 64],
    'optimizer': ['adam', 'sgd'],
}

CNN_TUNING_GRID = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128],
    'optimizer': ['adam'],
}

TRANSFER_LEARNING_TUNING_GRID = {
    'learning_rate': [0.0001, 0.00005, 0.00001],
    'batch_size': [32, 16],
    'optimizer': ['adam'],
}

TUNING_GRIDS = {
    'SimpleNN': SIMPLE_NN_TUNING_GRID,
    'MLP': MLP_TUNING_GRID,
    'CNN': CNN_TUNING_GRID,
    'TransferLearning': TRANSFER_LEARNING_TUNING_GRID,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_config(model_name):
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'SimpleNN', 'CNN')
    
    Returns:
        Configuration dictionary for the model
    
    Raises:
        ValueError: If model name not found
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


def get_training_config(model_name):
    """
    Get training parameters for a specific model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Training configuration dictionary
    """
    config = get_model_config(model_name)
    return config['training']


def get_tuning_grid(model_name):
    """
    Get hyperparameter tuning grid for a specific model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Parameter grid dictionary for hyperparameter tuning
    
    Raises:
        ValueError: If tuning grid not defined for model
    """
    if model_name not in TUNING_GRIDS:
        raise ValueError(f"Tuning grid not defined for '{model_name}'")
    return TUNING_GRIDS[model_name]


def print_all_configs():
    """Print all model configurations in a readable format."""
    print("\n" + "="*80)
    print("MODEL CONFIGURATIONS".center(80))
    print("="*80)
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{model_name}: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Architecture: {config['architecture']}")
        print(f"Training Parameters:")
        for key, value in config['training'].items():
            print(f"  - {key}: {value}")
        if 'augmentation' in config:
            print(f"Augmentation:")
            for key, value in config['augmentation'].items():
                print(f"  - {key}: {value}")
        if config['model_kwargs']:
            print(f"Model Kwargs:")
            for key, value in config['model_kwargs'].items():
                print(f"  - {key}: {value}")
    
    print("\n" + "="*80 + "\n")


# Example usage (uncomment to test)
if __name__ == "__main__":
    print_all_configs()
    
    # Example: Get specific model config
    simple_nn_config = get_model_config('SimpleNN')
    print(f"SimpleNN Training Config: {simple_nn_config['training']}")
    
    # Example: Get tuning grid
    cnn_grid = get_tuning_grid('CNN')
    print(f"\nCNN Tuning Grid: {cnn_grid}")
