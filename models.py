# models.py
import torch
import torch.nn as nn

# Import torchvision lazily inside AudioResNet so selecting the lightweight
# model doesn't require torchvision to be importable at module import time.

# Modello 1: CNN Personalizzata Leggera
class LightCNN(nn.Module):
    def __init__(self, num_classes=10):
        # Il modello è volutamente semplice per essere leggero e veloce da addestrare,
        # ma con capacità di estrazione di caratteristiche decente
        super(LightCNN, self).__init__()
        # La struttura è ispirata a una classica CNN per immagini, ma con meno filtri e strati
        self.features = nn.Sequential(
            # Input: 1 x 64 x 32 (mel spectrogram con 64 mel bins e 32 frame)
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # BatchNorm aiuta a stabilizzare e accelerare l'addestramento,
            # soprattutto con batch size piccoli
            nn.BatchNorm2d(16),
            # ReLU introduce non linearità, essenziale per la capacità di apprendimento del modello
            nn.ReLU(),
            # MaxPool riduce la dimensione spaziale, aumentando la robustezza e riducendo i parametri
            nn.MaxPool2d(2), # Output: 16 x 32 x 16
            
            # Secondo blocco convoluzionale, con più filtri per catturare caratteristiche più complesse
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # BatchNorm e ReLU come prima
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # MaxPool finale per ridurre ulteriormente la dimensione spaziale prima del classificatore
            nn.MaxPool2d(2), # Output: 32 x 16 x 8
        )
        # Classificatore completamente connesso, con un layer nascosto per migliorare la capacità di apprendimento
        self.classifier = nn.Sequential(
            # Flatten per passare da 3D a 1D, necessario prima del layer lineare
            nn.Flatten(),
            # Il primo layer lineare riduce la dimensionalità a 128, un numero ragionevole per un modello leggero
            nn.Linear(32 * 16 * 8, 128),
            # ReLU per introdurre non linearità nel classificatore
            nn.ReLU(),
            # Dropout per ridurre l'overfitting, soprattutto con dataset piccoli o modelli leggeri
            nn.Dropout(0.3),
            # Layer finale che mappa a num_classes (10 in questo caso)
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
    
# Modello 3: MobileNetV2 modificata per input a 1 canale (opzionale, non usato nei training/eval principali)
class AudioMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioMobileNetV2, self).__init__()
        try:
            from torchvision import models as tv_models
        except Exception as e:
            raise ImportError("AudioMobileNetV2 requires torchvision.models: " + str(e))

        self.mobilenet = tv_models.mobilenet_v2(num_classes=num_classes)
        # Sostituiamo il primo strato conv per input a 1 canale
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.mobilenet(x)
    
# Modello 4: AlexNet modificata per input a 1 canale (opzionale, non usato nei training/eval principali)
class AudioAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioAlexNet, self).__init__()
        try:
            from torchvision import models as tv_models
        except Exception as e:
            raise ImportError("AudioAlexNet requires torchvision.models: " + str(e))

        self.alexnet = tv_models.alexnet(num_classes=num_classes)
        # Sostituiamo il primo strato conv per input a 1 canale
        self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        # AlexNet needs a larger, square spatial map than the raw audio features provide.
        # Upsample only for this backbone so the rest of the pipeline can stay unchanged.
        x = torch.nn.functional.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return self.alexnet(x)