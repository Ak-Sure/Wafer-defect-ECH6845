import torch
import torch.nn as nn
from torchvision import models


# Simple Neural Network Model
class SimpleNN(nn.Module):
    """Single-layer neural network (logistic regression)."""
    
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        # Flatten the input: (batch_size, 52, 52) -> (batch_size, 2704)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Multi-Layer Perceptron Model
class MLP(nn.Module):
    """Multi-layer perceptron with hidden layers and activation functions."""
    
    def __init__(self, input_size, num_classes, hidden_sizes=[512, 256, 128], dropout=0.1):
        super(MLP, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return x
    
# Convolutional Neural Network Model
class WaferCNN(nn.Module):
    """Convolutional Neural Network for wafer defect classification."""
    
    def __init__(self, num_classes, input_channels=1, kernel_size=3, padding=1):
        super(WaferCNN, self).__init__()
        
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size dynamically based on kernel_size and padding
        # Input: 52x52
        self.flatten_size = self._calculate_flatten_size(52, kernel_size, padding)
        
        self.fc1 = nn.Linear(128 * self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def _calculate_flatten_size(self, input_size, kernel_size, padding):
        """Calculate output spatial dimensions after conv and pooling operations"""
        # Formula: output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
        # For each conv layer with stride=1
        size = input_size
        for _ in range(3):  # 3 conv blocks
            size = (size + 2 * padding - kernel_size) // 1 + 1
            size = size // 2  # MaxPool2d with stride 2
        return size * size
    
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # Reduce spatial dimensions
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # Reduce spatial dimensions
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # Reduce spatial dimensions
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

# Transfer learning

class WaferMobileNet(nn.Module):
    """Transfer learning with MobileNetV2 - lightweight and efficient."""
    
    def __init__(self, num_classes, pretrained=True, freeze_features=False):
        super(WaferMobileNet, self).__init__()
        
        # Load pre-trained MobileNetV2 (expects 3-channel RGB input)
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Optionally freeze feature extraction layers
        if freeze_features:
            for param in self.mobilenet.features.parameters():
                param.requires_grad = False
        
        # Replace final classifier layer
        # MobileNetV2 classifier is a Sequential with dropout and linear layer
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        # Always allow gradients in the classifier
        for param in self.mobilenet.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.mobilenet(x)


class WaferResNet18(nn.Module):
    """Transfer learning with ResNet18 - robust feature extraction with layer freezing."""
    
    def __init__(self, num_classes, pretrained=True, freeze_layers=0, dropout=0.5):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to load ImageNet pretrained weights
            freeze_layers: Number of residual layers to freeze (0-4)
                          0 = no freezing (train all layers)
                          1 = freeze layer1
                          2 = freeze layer1 + layer2
                          3 = freeze layer1 + layer2 + layer3
                          4 = freeze all conv layers (only train classifier)
            dropout: Dropout rate before final classification layer
        """
        super(WaferResNet18, self).__init__()
        
        # Load pre-trained ResNet18 (expects 3-channel RGB input)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        # Freeze initial conv and bn layers if any freezing is requested
        if freeze_layers > 0:
            for param in self.resnet.conv1.parameters():
                param.requires_grad = False
            for param in self.resnet.bn1.parameters():
                param.requires_grad = False
        
        # Freeze residual layers based on freeze_layers parameter
        layers_to_freeze = [self.resnet.layer1, self.resnet.layer2, 
                           self.resnet.layer3, self.resnet.layer4]
        
        for i in range(min(freeze_layers, 4)):
            for param in layers_to_freeze[i].parameters():
                param.requires_grad = False
        
        # Replace final fully connected layer
        # ResNet18 fc layer has 512 input features
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
        
        # Always allow gradients in the classifier
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
        # Store config for reference
        self.freeze_layers = freeze_layers
        self.num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.num_total = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        return self.resnet(x)
    
    def get_frozen_info(self):
        """Return information about frozen layers."""
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'freeze_layers': self.freeze_layers,
            'frozen_params': frozen,
            'trainable_params': trainable,
            'total_params': frozen + trainable,
            'trainable_percent': 100 * trainable / (frozen + trainable)
        }