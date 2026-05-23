#!/usr/bin/env python3
import argparse
import os
import time
import glob
import torch
from dataset import LABELS
from audio_utils import build_audio_transforms, load_audio_file, waveform_to_mel_db

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="wav file or directory with wavs")
    p.add_argument("--model", choices=["light","resnet"], default="light")
    p.add_argument("--model-path", default="best_light.pth")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sr", type=int, default=16000)
    return p.parse_args()

def load_model(name, path, device, num_classes):
    if name == "light":
        from models import LightCNN as M
    else:
        from models import AudioResNet as M
    model = M(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer_file(model, filepath, device, sr, mel_transform, amp_to_db, labels):
    wav_t = load_audio_file(filepath, target_sr=sr).to(device)
    with torch.no_grad():
        mel = waveform_to_mel_db(wav_t, mel_transform, amp_to_db)
        inp = mel.unsqueeze(0).to(device)
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        top_idx = int(probs.argmax())
        return labels[top_idx], probs[top_idx]

def main():
    args = build_args()
    device = torch.device(args.device)
    labels = LABELS
    model = load_model(args.model, args.model_path, device, num_classes=len(labels))
    mel_transform, amp_to_db = build_audio_transforms(sample_rate=args.sr)

    paths = []
    if os.path.isdir(args.path):
        paths = sorted(glob.glob(os.path.join(args.path, "*.wav")))
    else:
        paths = [args.path]

    if not paths:
        raise FileNotFoundError(f"Nessun file WAV trovato in: {args.path}")

    for p in paths:
        label, prob = infer_file(model, p, device, args.sr, mel_transform, amp_to_db, labels)
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {os.path.basename(p)} -> {label} ({prob*100:.1f}%)")

if __name__ == "__main__":
    main()