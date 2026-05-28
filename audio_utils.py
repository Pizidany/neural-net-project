import os
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

TARGET_SR = 16000
NUM_SAMPLES = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# Funzioni di utilità per il preprocessing audio, trasformazioni e caricamento da file

# Costruisce le trasformazioni per convertire un waveform in un mel spectrogram in dB
def build_audio_transforms(
    sample_rate: int = TARGET_SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> Tuple[T.MelSpectrogram, T.AmplitudeToDB]:
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    amp_to_db = T.AmplitudeToDB()
    return mel_transform, amp_to_db

# Pad o crop un waveform a una lunghezza fissa (num_samples), utile per uniformare l'input al modello
def pad_or_crop_waveform(waveform: torch.Tensor, num_samples: int = NUM_SAMPLES) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[1] < num_samples:
        pad_amount = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    else:
        waveform = waveform[:, :num_samples]
    return waveform

# Preprocessa un file audio dato il suo percorso,
# restituendo un tensore per essere passato al modello
def preprocess_waveform_np(
    waveform_np: np.ndarray,
    sr: int,
    target_sr: int = TARGET_SR,
    num_samples: int = NUM_SAMPLES,
) -> torch.Tensor:
    # Se l'audio ha più canali, convertilo in mono facendo la media tra i canali
    if waveform_np.ndim > 1:
        waveform_np = np.mean(waveform_np, axis=1)

    waveform = torch.from_numpy(waveform_np.astype(np.float32))

    # Se la frequenza di campionamento del file è diversa da quella target, resamplea l'audio
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

    waveform = pad_or_crop_waveform(waveform, num_samples=num_samples)
    return waveform

# Funzione per convertire un waveform in un mel spectrogram in dB usando le trasformazioni costruite
def waveform_to_mel_db(
    waveform: torch.Tensor,
    mel_transform: T.MelSpectrogram,
    amp_to_db: T.AmplitudeToDB,
) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return amp_to_db(mel_transform(waveform))

# Funzione per caricare un file audio da percorso,
# preprocessarlo e restituire un tensore per il modello
def load_audio_file(filepath: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File non trovato: {filepath}")

    waveform_np, file_sr = sf.read(filepath, dtype="float32")
    return preprocess_waveform_np(waveform_np, file_sr, target_sr=target_sr)
