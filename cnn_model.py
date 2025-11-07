"""
CNN Models for Pressure Sensor Classification
Separate models for:
- Object recognition (6 classes): ball, bottle, empty, pen, phone, hand
- Action recognition (6 classes): none, press, rotate, tap, touch, hold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PressureCNN(nn.Module):
    """
    Single-task CNN architecture for pressure sensor data
    Input: (batch, 1, 16, 16) - grayscale pressure map
    Output: (batch, num_classes)
    """
    
    def __init__(self, num_classes, dropout=0.5):
        super(PressureCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 2 pooling: 16 -> 8 -> 4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(-1, 256 * 4 * 4)
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AdvancedPressureCNN(nn.Module):
    """
    Advanced CNN with residual connections for better performance
    Single-task version for either object or action recognition
    """
    
    def __init__(self, num_classes=6):
        super(AdvancedPressureCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classification output
        out = self.fc(x)
        
        return out


def get_model(model_type='simple', task='object', **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: 'simple' or 'advanced'
        task: 'object' or 'action' (determines number of classes)
        **kwargs: Additional model parameters (e.g., custom num_classes)
    
    Returns:
        model: PyTorch model
    """
    # Default class numbers based on task
    if task == 'object':
        default_classes = 6  # ball, bottle, empty, pen, phone, hand
    elif task == 'action':
        default_classes = 6  # none, press, rotate, tap, touch, hold
    else:
        default_classes = kwargs.get('num_classes', 6)
    
    num_classes = kwargs.get('num_classes', default_classes)
    
    if model_type == 'simple':
        return PressureCNN(num_classes=num_classes)
    
    elif model_type == 'advanced':
        return AdvancedPressureCNN(num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'simple' or 'advanced'.")


if __name__ == '__main__':
    # Test models
    batch_size = 4
    x = torch.randn(batch_size, 1, 16, 16)
    
    print("Testing Simple CNN for Object Recognition:")
    model_obj = get_model('simple', task='object')
    out = model_obj(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_obj.parameters()):,}")
    
    print("\nTesting Simple CNN for Action Recognition:")
    model_act = get_model('simple', task='action')
    out = model_act(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_act.parameters()):,}")
    
    print("\nTesting Advanced CNN for Object Recognition:")
    model_obj_adv = get_model('advanced', task='object')
    out = model_obj_adv(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_obj_adv.parameters()):,}")
    
    print("\nTesting Advanced CNN for Action Recognition:")
    model_act_adv = get_model('advanced', task='action')
    out = model_act_adv(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_act_adv.parameters()):,}")

