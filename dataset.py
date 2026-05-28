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
    def __init__(self, subset='training', root='./dataset_audio', return_path=False):

         # Scrica database se manca e inizializza il dataset di torchaudio
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset=subset)
        # Costruiamo le trasformazioni audio (mel spectrogram + db)
        self.mel_transform, self.amp_to_db = build_audio_transforms(sample_rate=TARGET_SR)
        # Opzione per restituire anche il path del file per l'analisi degli errori
        self.return_path = return_path
        
        # Filtro il datset per ridurre il tempo di training e valutazione,
        # mantenendo solo le classi di interesse, e memorizzo gli indici validi per __getitem__
        self.valid_indices = []
        for idx in range(len(self.dataset)):
            _, _, label, _, _ = self.dataset[idx]
            if label in label_to_index:
                self.valid_indices.append(idx)

    def __len__(self):
        # Restituisce la lunghezza del dataset filtrato
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Ottiene l'indice reale dal dataset filtrato
        actual_idx = self.valid_indices[idx]
        waveform, _, label, _, _ = self.dataset[actual_idx]
        
        # Normalizza e uniforma la lunghezza del waveform
        # a TARGET_SR e NUM_SAMPLES (1 secondo di audio a 16kHz)
        waveform = pad_or_crop_waveform(waveform)
            
        # Trasformazione in Spettrogramma
        mel_spec = self.amp_to_db(self.mel_transform(waveform))
        
        # Mappa la label in numero
        label_idx = label_to_index[label]

        # Se return_path è True, restituisce anche il path del file audio per l'analisi degli errori
        if self.return_path:
            file_path = self.dataset._walker[actual_idx]
            return mel_spec, label_idx, file_path

        return mel_spec, label_idx

def get_dataloaders(batch_size=64):
    # Crea i dataloader per training, validation e test set
    train_set = SpeechCommandsPipeline(subset='training')
    val_set = SpeechCommandsPipeline(subset='validation')
    test_set = SpeechCommandsPipeline(subset='testing')
    # drop_last=True per evitare batch incompleti che possono causare problemi con batchnorm o simili
    # shuffle=True solo per il training set, per mescolare i dati ad ogni epoca
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader