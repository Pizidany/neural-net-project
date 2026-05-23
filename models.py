# models.py
import torch
import torch.nn as nn

# Import torchvision lazily inside AudioResNet so selecting the lightweight
# model doesn't require torchvision to be importable at module import time.

# Modello 1: CNN Personalizzata Leggera
class LightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 16 x 32 x 16
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 32 x 16 x 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Modello 2: ResNet18 modificata per input a 1 canale
class AudioResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioResNet, self).__init__()
        # Carichiamo una ResNet18 standard (import lazy)
        try:
            from torchvision import models as tv_models
        except Exception as e:
            raise ImportError("AudioResNet requires torchvision.models: " + str(e))

        self.resnet = tv_models.resnet18(num_classes=num_classes)
        # Sostituiamo il primo strato conv perché ResNet nasce per immagini a 3 canali (RGB), noi ne abbiamo 1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.resnet(x)