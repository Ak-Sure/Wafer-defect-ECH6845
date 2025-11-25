import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import kagglehub



def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=20, device='cuda', patience=5):
    """
    Train a PyTorch model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        device: Device to train on
        patience: Early stopping patience
    
    Returns:
        Dictionary containing training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best validation loss!")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation loss: {best_val_loss:.4f}")
    
    return history


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set.
    
    Returns:
        Tuple of (accuracy, all_labels, all_predictions)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_labels, all_preds


def train_model_simple(model, train_loader, criterion, optimizer, 
                       num_epochs=20, device='cuda'):
    """
    Train a PyTorch model WITHOUT validation set (simple training).
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
    
    Returns:
        Dictionary containing training history
    """
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    print(f"\nTraining complete!")
    return history


def plot_training_history(history, title="Training History"):
    """
    Plot training and validation loss/accuracy curves.
    Handles both histories with validation (from train_model) and without (from train_model_simple).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    has_validation = 'val_loss' in history
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if has_validation:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    if has_validation:
        axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(labels, predictions, num_classes, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def setup_model_and_loaders(model_class, X_train, X_val, X_test, y_train, y_val, y_test,
                           input_size=None, num_classes=38, device='cuda', batch_size=64,
                           model_kwargs=None, verbose=True):
    """
    Generalizable function to initialize a model and create data loaders.
    
    Args:
        model_class: PyTorch model class (e.g., SimpleNN, MLP, CNN)
        X_train: Training data (numpy array or tensor)
        X_val: Validation data (numpy array or tensor)
        X_test: Test data (numpy array or tensor)
        y_train: Training labels (numpy array or tensor)
        y_val: Validation labels (numpy array or tensor)
        y_test: Test labels (numpy array or tensor)
        input_size: Input size for model (if None, calculated as prod of X_train.shape[1:])
        num_classes: Number of output classes (default: 38)
        device: Device to place model on (default: 'cuda')
        batch_size: Batch size for data loaders (default: 64)
        model_kwargs: Additional keyword arguments to pass to model class
        verbose: Whether to print model info and loader stats (default: True)
    
    Returns:
        Dictionary containing:
            - 'model': Initialized model on specified device
            - 'train_loader': Training DataLoader
            - 'val_loader': Validation DataLoader
            - 'test_loader': Test DataLoader
            - 'total_params': Total number of trainable parameters
            - 'input_size': Input size used
            - 'batch_size': Batch size used
    
    Example:
        >>> result = setup_model_and_loaders(
        ...     SimpleNN, X_train, X_val, X_test, y_train, y_val, y_test,
        ...     input_size=2704, num_classes=38, device='cuda', batch_size=64
        ... )
        >>> model = result['model']
        >>> train_loader = result['train_loader']
        >>> print(f"Total params: {result['total_params']:,}")
    """
    # Default model kwargs
    if model_kwargs is None:
        model_kwargs = {}
    
    # Calculate input size if not provided
    if input_size is None:
        input_size = np.prod(X_train.shape[1:])
    
    # Initialize model
    # WaferCNN has different signature: (num_classes, input_channels=1)
    # Other models: (input_size, num_classes, ...)
    if model_class.__name__ == 'WaferCNN':
        model = model_class(num_classes, **model_kwargs).to(device)
    else:
        model = model_class(input_size, num_classes, **model_kwargs).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Print model info
    if verbose:
        print(f"{'='*70}")
        print(f"Model: {model_class.__name__}")
        print(f"{'='*70}")
        print(f"Architecture: {input_size} -> {num_classes}")
        print(f"Total parameters: {total_params:,}")
        print(f"\nModel structure:")
        print(model)
        print(f"{'='*70}\n")
    
    # Convert to tensors if needed
    X_train_tensor = torch.FloatTensor(X_train) if not isinstance(X_train, torch.Tensor) else X_train
    X_val_tensor = torch.FloatTensor(X_val) if not isinstance(X_val, torch.Tensor) else X_val
    X_test_tensor = torch.FloatTensor(X_test) if not isinstance(X_test, torch.Tensor) else X_test
    
    y_train_tensor = torch.LongTensor(y_train) if not isinstance(y_train, torch.Tensor) else y_train
    y_val_tensor = torch.LongTensor(y_val) if not isinstance(y_val, torch.Tensor) else y_val
    y_test_tensor = torch.LongTensor(y_test) if not isinstance(y_test, torch.Tensor) else y_test
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Print loader info
    if verbose:
        print(f"Data Loaders:")
        print(f"  Training batches: {len(train_loader)} (batch_size={batch_size})")
        print(f"  Validation batches: {len(val_loader)} (batch_size={batch_size})")
        print(f"  Test batches: {len(test_loader)} (batch_size={batch_size})")
        print(f"{'='*70}\n")
    
    return {
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'total_params': total_params,
        'input_size': input_size,
        'batch_size': batch_size
    }


def kfold_cross_validation(model_class, X, y, param_config, k=5, input_size=None, 
                          num_classes=38, device='cuda', num_epochs=20, 
                          patience=5, verbose=True):
    """
    Perform K-Fold Cross-Validation with specified hyperparameters.
    
    Splits data into K folds and trains model K times, each time using a different
    fold as validation set. Returns average performance across all folds.
    
    Args:
        model_class: PyTorch model class
        X: Full dataset features (numpy array)
        y: Full dataset labels (numpy array)
        param_config: Dictionary with hyperparameters:
            - 'learning_rate': Learning rate for optimizer
            - 'batch_size': Batch size for training
            - 'optimizer': Optimizer type ('adam' or 'sgd')
            - 'model_kwargs': Additional model arguments (optional)
        k: Number of folds (default: 5)
        input_size: Input feature size (auto-calculated if None)
        num_classes: Number of output classes (default: 38)
        device: Device to train on
        num_epochs: Maximum training epochs per fold
        patience: Early stopping patience
        verbose: Print progress information
    
    Returns:
        Dictionary containing:
            - 'fold_results': List of results for each fold
            - 'mean_train_acc': Mean training accuracy across folds
            - 'mean_val_acc': Mean validation accuracy across folds
            - 'std_train_acc': Std deviation of training accuracy
            - 'std_val_acc': Std deviation of validation accuracy
            - 'best_fold': Index of best performing fold
            - 'fold_models': List of trained models for each fold
            - 'summary_df': DataFrame with fold-wise results
    
    Example:
        >>> param_config = {
        ...     'learning_rate': 0.001,
        ...     'batch_size': 64,
        ...     'optimizer': 'adam'
        ... }
        >>> results = kfold_cross_validation(
        ...     SimpleNN, X, y, param_config, k=5, device='cuda'
        ... )
        >>> print(f"Mean Val Acc: {results['mean_val_acc']:.4f} ± {results['std_val_acc']:.4f}")
    """
    from sklearn.model_selection import StratifiedKFold
    
    # Calculate input size if not provided
    if input_size is None:
        input_size = np.prod(X.shape[1:])
    
    # Extract hyperparameters
    learning_rate = param_config['learning_rate']
    batch_size = param_config['batch_size']
    optimizer_type = param_config.get('optimizer', 'adam')
    model_kwargs = param_config.get('model_kwargs', {})
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"K-FOLD CROSS-VALIDATION (k={k})".center(70))
        print(f"{'='*70}")
        print(f"Model: {model_class.__name__}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Batch Size: {batch_size}")
        print(f"Optimizer: {optimizer_type}")
        print(f"{'='*70}\n")
    
    # Initialize K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    fold_models = []
    
    # Iterate through folds
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx}/{k}".center(70))
            print(f"{'='*70}")
        
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        if verbose:
            print(f"Train samples: {len(X_train_fold)}")
            print(f"Val samples:   {len(X_val_fold)}\n")
        
        # Setup model and loaders
        setup_result = setup_model_and_loaders(
            model_class, X_train_fold, X_val_fold, X_val_fold,  # Use val as dummy test
            y_train_fold, y_val_fold, y_val_fold,
            input_size=input_size, num_classes=num_classes, device=device,
            batch_size=batch_size, model_kwargs=model_kwargs, verbose=False
        )
        
        model = setup_result['model']
        train_loader = setup_result['train_loader']
        val_loader = setup_result['val_loader']
        
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=num_epochs, device=device, patience=patience
        )
        
        # Get final accuracies
        train_acc = history['train_acc'][-1]
        val_acc = history['val_acc'][-1]
        
        # Store results
        fold_result = {
            'fold': fold_idx,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'epochs_trained': len(history['train_loss']),
            'history': history
        }
        fold_results.append(fold_result)
        fold_models.append(model)
        
        if verbose:
            print(f"\nFold {fold_idx} Results:")
            print(f"  Train Acc: {train_acc:.4f}")
            print(f"  Val Acc:   {val_acc:.4f}")
            print(f"  Epochs:    {len(history['train_loss'])}")
    
    # Calculate statistics
    train_accs = [r['train_acc'] for r in fold_results]
    val_accs = [r['val_acc'] for r in fold_results]
    
    mean_train_acc = np.mean(train_accs)
    mean_val_acc = np.mean(val_accs)
    std_train_acc = np.std(train_accs)
    std_val_acc = np.std(val_accs)
    
    best_fold_idx = np.argmax(val_accs)
    
    # Create summary DataFrame
    summary_data = []
    for result in fold_results:
        summary_data.append({
            'Fold': result['fold'],
            'Train_Acc': result['train_acc'],
            'Val_Acc': result['val_acc'],
            'Epochs': result['epochs_trained']
        })
    summary_df = pd.DataFrame(summary_data)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"K-FOLD CROSS-VALIDATION RESULTS".center(70))
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"\n{'='*70}")
        print(f"Mean Train Acc: {mean_train_acc:.4f} ± {std_train_acc:.4f}")
        print(f"Mean Val Acc:   {mean_val_acc:.4f} ± {std_val_acc:.4f}")
        print(f"Best Fold:      Fold {best_fold_idx + 1} (Val Acc: {val_accs[best_fold_idx]:.4f})")
        print(f"{'='*70}\n")
    
    return {
        'fold_results': fold_results,
        'mean_train_acc': mean_train_acc,
        'mean_val_acc': mean_val_acc,
        'std_train_acc': std_train_acc,
        'std_val_acc': std_val_acc,
        'best_fold': best_fold_idx,
        'fold_models': fold_models,
        'summary_df': summary_df,
        'param_config': param_config
    }


def train_best_model(model_class, X_train, X_val, X_test, y_train, y_val, y_test,
                    best_params, input_size=None, num_classes=38, device='cuda',
                    num_epochs=20, patience=5, save_path=None, verbose=True):
    """
    Train a model with the best hyperparameters and optionally save it.
    
    Args:
        model_class: PyTorch model class
        X_train, X_val, X_test: Training, validation, test data
        y_train, y_val, y_test: Training, validation, test labels
        best_params: Dictionary with optimal hyperparameters:
            - 'learning_rate': Optimal learning rate
            - 'batch_size': Optimal batch size
            - 'optimizer': Optimizer type
            - 'model_kwargs': Additional model arguments (optional)
        input_size: Input feature size (auto-calculated if None)
        num_classes: Number of output classes (default: 38)
        device: Device to train on
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        save_path: Path to save best model (optional, e.g., 'models/best_model.pth')
        verbose: Print progress information
    
    Returns:
        Dictionary containing:
            - 'model': Trained model
            - 'history': Training history
            - 'train_acc': Final training accuracy
            - 'val_acc': Final validation accuracy
            - 'test_acc': Test accuracy
            - 'best_params': Parameters used
            - 'save_path': Path where model was saved (if save_path provided)
    
    Example:
        >>> best_params = {'learning_rate': 0.001, 'batch_size': 64, 'optimizer': 'adam'}
        >>> result = train_best_model(
        ...     SimpleNN, X_train, X_val, X_test, y_train, y_val, y_test,
        ...     best_params, save_path='models/simple_nn_best.pth'
        ... )
        >>> print(f"Model saved to: {result['save_path']}")
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING BEST MODEL".center(70))
        print(f"{'='*70}")
        print(f"Model: {model_class.__name__}")
        print(f"Parameters:")
        for key, value in best_params.items():
            if key != 'model_kwargs':
                print(f"  {key}: {value}")
        print(f"{'='*70}\n")
    
    # Calculate input size if not provided
    if input_size is None:
        input_size = np.prod(X_train.shape[1:])
    
    # Extract hyperparameters
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    optimizer_type = best_params.get('optimizer', 'adam')
    model_kwargs = best_params.get('model_kwargs', {})
    
    # Setup model and loaders
    setup_result = setup_model_and_loaders(
        model_class, X_train, X_val, X_test, y_train, y_val, y_test,
        input_size=input_size, num_classes=num_classes, device=device,
        batch_size=batch_size, model_kwargs=model_kwargs, verbose=verbose
    )
    
    model = setup_result['model']
    train_loader = setup_result['train_loader']
    val_loader = setup_result['val_loader']
    test_loader = setup_result['test_loader']
    
    # Setup optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, device=device, patience=patience
    )
    
    # Evaluate on test set
    test_acc, test_labels, test_preds = evaluate_model(model, test_loader, device)
    
    # Save model if path provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model_class.__name__,
            'hyperparameters': best_params,
            'input_size': input_size,
            'num_classes': num_classes,
            'train_acc': history['train_acc'][-1],
            'val_acc': history['val_acc'][-1],
            'test_acc': test_acc,
            'history': history
        }, save_path)
        if verbose:
            print(f"\n✓ Model saved to: {save_path}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE".center(70))
        print(f"{'='*70}")
        print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"Final Val Accuracy:   {history['val_acc'][-1]:.4f}")
        print(f"Test Accuracy:        {test_acc:.4f}")
        print(f"Epochs Trained:       {len(history['train_loss'])}")
        print(f"{'='*70}\n")
    
    return {
        'model': model,
        'history': history,
        'train_acc': history['train_acc'][-1],
        'val_acc': history['val_acc'][-1],
        'test_acc': test_acc,
        'test_labels': test_labels,
        'test_preds': test_preds,
        'best_params': best_params,
        'save_path': save_path,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }


def load_best_model(model_class, checkpoint_path, device='cuda', verbose=True):
    """
    Load a saved model from checkpoint.
    
    Args:
        model_class: PyTorch model class
        checkpoint_path: Path to saved model checkpoint
        device: Device to load model on
        verbose: Print loading information
    
    Returns:
        Dictionary containing:
            - 'model': Loaded model
            - 'hyperparameters': Hyperparameters used during training
            - 'train_acc': Training accuracy from checkpoint
            - 'val_acc': Validation accuracy from checkpoint
            - 'test_acc': Test accuracy from checkpoint
            - 'history': Training history
    
    Example:
        >>> loaded = load_best_model(SimpleNN, 'models/simple_nn_best.pth')
        >>> model = loaded['model']
        >>> print(f"Test Accuracy: {loaded['test_acc']:.4f}")
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOADING MODEL".center(70))
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract information
    input_size = checkpoint['input_size']
    num_classes = checkpoint['num_classes']
    hyperparameters = checkpoint['hyperparameters']
    model_kwargs = hyperparameters.get('model_kwargs', {})
    
    # Initialize model
    # WaferCNN has different signature: (num_classes, input_channels=1)
    # Other models: (input_size, num_classes, ...)
    if model_class.__name__ == 'WaferCNN':
        model = model_class(num_classes, **model_kwargs).to(device)
    else:
        model = model_class(input_size, num_classes, **model_kwargs).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if verbose:
        print(f"Model Class: {checkpoint['model_class']}")
        print(f"Input Size: {input_size}")
        print(f"Num Classes: {num_classes}")
        print(f"\nHyperparameters:")
        for key, value in hyperparameters.items():
            if key != 'model_kwargs':
                print(f"  {key}: {value}")
        print(f"\nPerformance:")
        print(f"  Train Acc: {checkpoint['train_acc']:.4f}")
        print(f"  Val Acc:   {checkpoint['val_acc']:.4f}")
        print(f"  Test Acc:  {checkpoint['test_acc']:.4f}")
        print(f"{'='*70}\n")
    
    return {
        'model': model,
        'hyperparameters': hyperparameters,
        'train_acc': checkpoint['train_acc'],
        'val_acc': checkpoint['val_acc'],
        'test_acc': checkpoint['test_acc'],
        'history': checkpoint.get('history', None),
        'input_size': input_size,
        'num_classes': num_classes
    }


def hyperparameter_tuning(model_class, X_train, X_val, X_test, y_train, y_val, y_test,
                         param_grid, input_size=None, num_classes=38, device='cuda',
                         criterion=None, num_epochs=20, patience=5, verbose=True):
    """
    Perform hyperparameter tuning using grid search across multiple configurations.
    
    Evaluates all combinations of hyperparameters and returns results sorted by validation accuracy.
    
    Args:
        model_class: PyTorch model class (e.g., SimpleNN, MLP, CNN)
        X_train, X_val, X_test: Training, validation, test data arrays
        y_train, y_val, y_test: Training, validation, test labels
        param_grid: Dictionary defining hyperparameters to tune
            Standard training parameters:
                'learning_rate': [0.001, 0.0001]
                'batch_size': [32, 64]
                'optimizer': ['adam', 'sgd']
                'num_epochs': [10, 20, 30]  # Epochs to train for
            
            MLP architecture parameters (for tuning network structure):
                'hidden_sizes': [[512, 256, 128], [256, 128]]  # Direct layer sizes
                'num_hidden_layers': [2, 3, 4]  # Number of hidden layers
                'neurons_per_layer': [256, 512]  # Neurons in each layer
                'dropout': [0.1, 0.2, 0.3]  # Dropout rate
            
            Example for learning rate and batch size:
            {
                'learning_rate': [0.001, 0.0001],
                'batch_size': [32, 64],
                'optimizer': ['adam', 'sgd']
            }
            
            Example for MLP with architecture tuning:
            {
                'learning_rate': [0.001],
                'batch_size': [64],
                'num_epochs': [20],
                'hidden_sizes': [[512, 256, 128], [256, 128]],
                'dropout': [0.1, 0.2]
            }
        input_size: Input feature size (auto-calculated if None)
        num_classes: Number of output classes (default: 38)
        device: Device to train on ('cuda' or 'cpu')
        criterion: Loss function (default: CrossEntropyLoss)
        num_epochs: Default maximum training epochs (default: 20)
        patience: Early stopping patience (default: 5)
        verbose: Print progress information (default: True)
    
    Returns:
        Dictionary containing:
            - 'results': List of dicts with params and metrics (sorted by val_acc descending)
            - 'best_params': Best hyperparameter configuration
            - 'best_model': Best trained model
            - 'best_history': Training history of best model
            - 'best_val_acc': Best validation accuracy
            - 'best_test_acc': Test accuracy of best model
            - 'summary_df': Pandas DataFrame of all results
    
    Examples:
        Example 1 - Basic training hyperparameters:
        >>> param_grid = {
        ...     'learning_rate': [0.001, 0.0001],
        ...     'batch_size': [32, 64],
        ...     'optimizer': ['adam', 'sgd']
        ... }
        >>> results = hyperparameter_tuning(
        ...     SimpleNN, X_train, X_val, X_test, y_train, y_val, y_test,
        ...     param_grid, input_size=2704, num_classes=38
        ... )
        
        Example 2 - MLP with architecture tuning (epochs, neurons, layers):
        >>> param_grid = {
        ...     'learning_rate': [0.001],
        ...     'batch_size': [32, 64],
        ...     'num_epochs': [10, 20],
        ...     'hidden_sizes': [[512, 256, 128], [256, 128]],
        ...     'dropout': [0.1, 0.2]
        ... }
        >>> results = hyperparameter_tuning(
        ...     MLP, X_train, X_val, X_test, y_train, y_val, y_test,
        ...     param_grid, input_size=2704, num_classes=38
        ... )
        >>> print(results['best_params'])
        >>> print(results['summary_df'])
    """
    import itertools
    
    # Set default criterion
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Calculate input size if not provided
    if input_size is None:
        input_size = np.prod(X_train.shape[1:])
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combinations = list(itertools.product(*param_values))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TUNING: Grid Search".center(70))
        print(f"{'='*70}")
        print(f"Model: {model_class.__name__}")
        print(f"Total combinations to evaluate: {len(param_combinations)}")
        print(f"Parameters: {', '.join(param_names)}")
        print(f"{'='*70}\n")
    
    results_list = []
    best_val_acc = -1
    best_model = None
    best_params = None
    best_history = None
    
    # Iterate through all parameter combinations
    for idx, param_values_tuple in enumerate(param_combinations, 1):
        params = dict(zip(param_names, param_values_tuple))
        
        if verbose:
            print(f"\n[{idx}/{len(param_combinations)}] Testing: {params}")
            print("-" * 70)
        
        try:
            # Extract hyperparameters
            learning_rate = params.get('learning_rate', 0.001)
            batch_size = params.get('batch_size', 64)
            optimizer_type = params.get('optimizer', 'adam')
            epochs_to_train = params.get('num_epochs', num_epochs)  # Get epochs from params if provided
            
            # Build model_kwargs with architecture parameters
            model_kwargs = {}
            
            # Handle hidden_sizes for MLP (number of neurons per layer)
            if 'hidden_sizes' in params:
                hidden_sizes = params['hidden_sizes']
                # If it's a string or list representation, parse it
                if isinstance(hidden_sizes, str):
                    # Handle formats like "512,256,128" or "[512,256,128]"
                    hidden_sizes = hidden_sizes.strip('[]').split(',')
                    hidden_sizes = [int(h.strip()) for h in hidden_sizes]
                model_kwargs['hidden_sizes'] = hidden_sizes
            
            # Handle hidden_layers for MLP (number of hidden layers)
            if 'num_hidden_layers' in params:
                num_layers = params['num_hidden_layers']
                # If not already specified, create default layer sizes
                if 'hidden_sizes' not in model_kwargs:
                    base_neurons = params.get('neurons_per_layer', 256)
                    model_kwargs['hidden_sizes'] = [base_neurons // (2 ** i) for i in range(num_layers)]
            
            # Handle neurons_per_layer for MLP
            if 'neurons_per_layer' in params and 'hidden_sizes' not in model_kwargs and 'num_hidden_layers' not in params:
                # Use default 3 hidden layers
                neurons = params['neurons_per_layer']
                model_kwargs['hidden_sizes'] = [neurons, neurons // 2, neurons // 4]
            
            # Handle dropout for MLP
            if 'dropout' in params and model_class.__name__ == 'MLP':
                model_kwargs['dropout'] = params['dropout']
            
            # Add any other parameters that aren't known training parameters
            for key in params:
                if key not in ['learning_rate', 'batch_size', 'optimizer', 'num_epochs', 
                              'hidden_sizes', 'num_hidden_layers', 'neurons_per_layer', 'dropout',
                              'kernel_size', 'padding']:
                    model_kwargs[key] = params[key]
            
            # Setup model and loaders
            setup_result = setup_model_and_loaders(
                model_class, X_train, X_val, X_test, y_train, y_val, y_test,
                input_size=input_size, num_classes=num_classes, device=device,
                batch_size=batch_size, model_kwargs=model_kwargs, verbose=False
            )
            
            model = setup_result['model']
            train_loader = setup_result['train_loader']
            val_loader = setup_result['val_loader']
            test_loader = setup_result['test_loader']
            
            # Setup optimizer
            if optimizer_type.lower() == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_type.lower() == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            else:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train model
            history = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                num_epochs=epochs_to_train, device=device, patience=patience
            )
            
            # Evaluate on validation set
            model.eval()
            val_acc = history['val_acc'][-1]
            val_loss = history['val_loss'][-1]
            
            # Evaluate on test set
            test_acc, _, _ = evaluate_model(model, test_loader, device)
            train_acc = history['train_acc'][-1]
            
            # Store results
            result = {
                'params': params,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'test_acc': test_acc,
                'epochs_trained': len(history['train_loss']),
                'total_params': setup_result['total_params']
            }
            results_list.append(result)
            
            if verbose:
                print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_params = params
                best_history = history
                if verbose:
                    print(f"✓ NEW BEST! (Val Acc: {val_acc:.4f})")
        
        except Exception as e:
            if verbose:
                print(f"✗ Error: {str(e)}")
            result = {
                'params': params,
                'train_acc': None,
                'val_acc': None,
                'val_loss': None,
                'test_acc': None,
                'epochs_trained': None,
                'total_params': None,
                'error': str(e)
            }
            results_list.append(result)
    
    # Sort results by validation accuracy (descending)
    results_list.sort(key=lambda x: x['val_acc'] if x['val_acc'] is not None else -1, reverse=True)
    
    # Create summary DataFrame
    summary_data = []
    for i, result in enumerate(results_list, 1):
        row = {'Rank': i}
        row.update(result['params'])
        row['Train_Acc'] = result['train_acc']
        row['Val_Acc'] = result['val_acc']
        row['Test_Acc'] = result['test_acc']
        row['Epochs'] = result['epochs_trained']
        row['Params'] = result['total_params']
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TUNING RESULTS SUMMARY".center(70))
        print(f"{'='*70}")
        print(f"\nTop 5 Configurations:")
        print(summary_df.head(5).to_string(index=False))
        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION:")
        print(f"{'='*70}")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
        print(f"Test Accuracy (Best Model): {results_list[0]['test_acc']:.4f}")
        print(f"{'='*70}\n")
    
    return {
        'results': results_list,
        'best_params': best_params,
        'best_model': best_model,
        'best_history': best_history,
        'best_val_acc': best_val_acc,
        'best_test_acc': results_list[0]['test_acc'],
        'summary_df': summary_df
    }


print("Helper functions defined successfully!")
