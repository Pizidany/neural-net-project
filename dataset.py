# dataset.py
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from audio_utils import build_audio_transforms, pad_or_crop_waveform, TARGET_SR

# Selezioniamo solo alcune classi per semplificare, o usa tutte le 35 classi del dataset
LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
label_to_index = {label: i for i, label in enumerate(LABELS)}

class SpeechCommandsPipeline(Dataset):
    def __init__(self, subset='training', root='./dataset_audio'):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset=subset)
        self.mel_transform, self.amp_to_db = build_audio_transforms(sample_rate=TARGET_SR)
        
        # Filtriamo il dataset per tenere solo le classi che ci interessano (opzionale)
        self.valid_indices = []
        for idx in range(len(self.dataset)):
            _, _, label, _, _ = self.dataset[idx]
            if label in label_to_index:
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        waveform, _, label, _, _ = self.dataset[actual_idx]
        
        # Padding/Cropping a 1 secondo (16000 campioni)
        waveform = pad_or_crop_waveform(waveform)
            
        # Trasformazione in Spettrogramma
        mel_spec = self.amp_to_db(self.mel_transform(waveform))
        
        # Mappa la label in numero
        label_idx = label_to_index[label]
        
        return mel_spec, label_idx

def get_dataloaders(batch_size=64):
    train_set = SpeechCommandsPipeline(subset='training')
    val_set = SpeechCommandsPipeline(subset='validation')
    test_set = SpeechCommandsPipeline(subset='testing')
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader