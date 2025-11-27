# Wafer Defect Classification (Final project for ECH6845)

Deep learning to identify the defect type in semiconductor wafers using the WM-811K dataset with 38 unique defect classes.

**Contributors:** Akash Surendran, Binod L Perera, Shubham Ravan, Renee Wan

![38 defect types in semiconductor wafer chips](output.png)

## 📋 Overview

This project implements and compares multiple deep learning architectures for classifying defect patterns in semiconductor wafer maps. The dataset contains 38,015 wafer map samples (52×52 pixels) across 38 distinct defect classes, represented as binary patterns.

### Models Implemented

1. **Simple Neural Network** - Single-layer fully connected baseline
2. **Multi-Layer Perceptron (MLP)** - Deep fully connected architecture with dropout regularization
3. **Convolutional Neural Network (CNN)** - Custom CNN with batch normalization and max pooling
4. **Transfer Learning** - Fine-tuned ResNet18 and MobileNetV2 models

## 🚀 Step-by-Step Setup Guide

### Step 1: Load Conda Module and Create Environment

```bash
module load conda
conda create -n wafer_defect python=3.11 -y
conda activate wafer_defect
```

### Step 2: Install Dependencies

```bash
# Install core scientific libraries
conda install tqdm numpy pandas matplotlib scipy scikit-learn jupyter ipykernel -y

# Install OpenCV
conda install -c conda-forge opencv -y

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Register Jupyter kernel
python -m ipykernel install --user --name=wafer_defect --display-name="Python (Wafer Defect)"
```

### Step 3: Download Dataset

Place the `mixedtype-wafer-defect-datasets` folder containing `Wafer_Map_Datasets.npz` in the project root directory.


## 🔧 Usage

### 1. Load and Explore Dataset

```python
from data_loading import WaferDataLoader

# Initialize data loader
loader = WaferDataLoader('mixedtype-wafer-defect-datasets')

# Print dataset summary
loader.print_summary()

# Visualize class distribution
loader.plot_class_distribution()

# Display sample wafer maps from each class
loader.plot_sample_gallery()
```

### 2. Train a Model

```python
import torch
from models import SimpleNN, MLP, WaferCNN, WaferResNet18
from utility import setup_model_and_loaders, train_model

# Setup model and data loaders
result = setup_model_and_loaders(
    WaferCNN, X_train, X_val, X_test, 
    y_train, y_val, y_test,
    num_classes=38, 
    batch_size=64
)

# Define optimizer and loss
optimizer = torch.optim.Adam(result['model'].parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
history = train_model(
    result['model'], 
    result['train_loader'], 
    result['val_loader'],
    criterion, 
    optimizer,
    num_epochs=20,
    patience=5
)
```

### 3. Hyperparameter Tuning

```python
from utility import hyperparameter_tuning

# Define parameter grid
param_grid = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64],
    'optimizer': ['adam', 'sgd']
}

# Run grid search
tuning_results = hyperparameter_tuning(
    MLP, X_train, X_val, X_test, 
    y_train, y_val, y_test,
    param_grid,
    num_classes=38
)

print(tuning_results['best_params'])
print(tuning_results['summary_df'])
```

### 4. Evaluate Model

```python
from utility import evaluate_model, plot_confusion_matrix

# Evaluate on test set
test_acc, test_labels, test_preds = evaluate_model(
    model, test_loader, device='cuda'
)

print(f"Test Accuracy: {test_acc:.4f}")

# Plot confusion matrix
plot_confusion_matrix(test_labels, test_preds, num_classes=38)
```

## 📊 Model Configurations

All model configurations are centralized in `config.py`:

- **DATASET_CONFIG**: Dataset paths and split ratios
- **TRAINING_CONFIG**: Global training defaults
- **MODEL_CONFIGS**: Architecture-specific configurations
- **TUNING_GRIDS**: Hyperparameter search spaces

Example:
```python
from config import get_model_config, get_tuning_grid

# Get CNN configuration
cnn_config = get_model_config('CNN')
print(cnn_config['architecture'])  # Conv(32) -> Conv(64) -> Conv(128) -> FC(38)

# Get tuning grid for MLP
mlp_grid = get_tuning_grid('MLP')
```

## 📈 Key Features

- **Comprehensive Hyperparameter Tuning**: Grid search with k-fold cross-validation
- **Early Stopping**: Prevent overfitting with configurable patience
- **Model Checkpointing**: Save and load best models
- **Visualization Tools**: Training curves, confusion matrices, and wafer map galleries
- **Transfer Learning**: Pre-trained ResNet18 and MobileNetV2 with layer freezing options

## 🧪 Jupyter Notebooks

The repository includes several notebooks for experimentation:

1. **model_tuning_simplenn.ipynb** - Baseline simple neural network experiments
2. **model_tuning_mlp.ipynb** - Multi-layer perceptron with architecture search
3. **model_tuning_cnn.ipynb** - CNN hyperparameter optimization
4. **model_tuning_transfer_learning.ipynb** - Transfer learning with ResNet18/MobileNetV2
5. **model_comparison.ipynb** - Comprehensive comparison of all approaches

## 🔬 Dataset Information

- **Total Samples**: 38,015 wafer maps
- **Image Size**: 52 × 52 pixels (grayscale)
- **Number of Classes**: 38 unique defect patterns
- **Label Format**: 8-bit binary patterns, with each the 1's indicating a specific defect type (e.g., "10100000" for Class 9)
- **Split Ratio**: 70% train, 15% validation, 15% test

## 📄 License

This project was developed as part of the ECH6845 course.
