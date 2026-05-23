# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from dataset import get_dataloaders
from models import LightCNN, AudioResNet

def train(model_name, epochs=10, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uso il dispositivo: {device}")
    
    train_loader, val_loader, _ = get_dataloaders(batch_size)
    
    # Selezione modello
    if model_name == 'light':
        model = LightCNN(num_classes=10).to(device)
    elif model_name == 'resnet':
        model = AudioResNet(num_classes=10).to(device)
    else:
        raise ValueError("Modello non riconosciuto. Scegli tra 'light' e 'resnet'")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        
        # Validazione
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"Epoca [{epoch+1}/{epochs}] - Loss Train: {epoch_loss:.4f} - Acc Train: {epoch_acc:.2f}% | Acc Val: {val_acc:.2f}%")
        
        # Salva il modello migliore
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")
            print("=> Modello migliore salvato!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Speech Commands Models")
    parser.add_argument('--model', type=str, required=True, choices=['light', 'resnet'], help="Scegli 'light' o 'resnet'")
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    train(model_name=args.model, epochs=args.epochs)