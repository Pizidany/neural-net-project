import argparse
import csv
import glob
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, precision_recall_fscore_support

from audio_utils import build_audio_transforms, load_audio_file, waveform_to_mel_db
from dataset import LABELS as CLASS_NAMES
from dataset import get_dataloaders
from models import AudioResNet, LightCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Valutazione e confronto modelli audio")
    parser.add_argument("--source", choices=["test", "custom", "both"], default="custom")
    parser.add_argument("--wav-dir", type=str, default=None, help="Cartella con file .wav (ricorsiva)")
    parser.add_argument("--models", nargs="+", choices=["light", "resnet"], default=["light", "resnet"])
    parser.add_argument("--light-path", type=str, default="best_light.pth")
    parser.add_argument("--resnet-path", type=str, default="best_resnet.pth")
    parser.add_argument("--output-dir", type=str, default="eval_outputs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--labels-from", choices=["none", "folder", "prefix"], default="none")
    parser.add_argument("--prefix-sep", type=str, default="_")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(model_name: str, checkpoint_path: str, device: torch.device, num_classes: int):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    if model_name == "light":
        model = LightCNN(num_classes=num_classes)
    elif model_name == "resnet":
        model = AudioResNet(num_classes=num_classes)
    else:
        raise ValueError(f"Modello non supportato: {model_name}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_loader(model, loader, device):
    y_true = []
    y_pred = []
    probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            p = torch.softmax(outputs, dim=1)
            pred = p.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            probs.extend(p.max(dim=1).values.cpu().numpy().tolist())

    return y_true, y_pred, probs


def predict_wav_files(model, wav_paths, device, sr, mel_transform, amp_to_db):
    results = []
    with torch.no_grad():
        for wav_path in wav_paths:
            waveform = load_audio_file(wav_path, target_sr=sr).to(device)
            mel = waveform_to_mel_db(waveform, mel_transform, amp_to_db)
            inp = mel.unsqueeze(0).to(device)
            out = model(inp)
            p = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(p))

            results.append(
                {
                    "file": wav_path,
                    "pred_idx": pred_idx,
                    "pred_label": CLASS_NAMES[pred_idx],
                    "confidence": float(p[pred_idx]),
                }
            )
    return results


def infer_true_label(wav_path: str, labels_from: str, prefix_sep: str):
    if labels_from == "none":
        return None

    if labels_from == "folder":
        candidate = os.path.basename(os.path.dirname(wav_path))
    else:
        base = os.path.basename(wav_path)
        candidate = base.split(prefix_sep)[0]

    if candidate in CLASS_NAMES:
        return candidate
    return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def accuracy_percent(y_true, y_pred):
    if not y_true:
        return 0.0
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return 100.0 * float((y_true_arr == y_pred_arr).mean())


def save_confusion_matrix(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_per_class_metrics(y_true, y_pred, title, output_path):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width=width, label="Precision")
    ax.bar(x, recall, width=width, label="Recall")
    ax.bar(x + width, f1, width=width, label="F1")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_pred_distribution(results, model_name, output_path):
    counts = Counter(r["pred_label"] for r in results)
    values = [counts.get(label, 0) for label in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(CLASS_NAMES, values)
    ax.set_title(f"Distribuzione predizioni - {model_name}")
    ax.set_ylabel("Numero file")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_confidence_hist(results, model_name, output_path):
    confs = [r["confidence"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confs, bins=20, range=(0.0, 1.0))
    ax.set_title(f"Distribuzione confidenza top-1 - {model_name}")
    ax.set_xlabel("Confidenza")
    ax.set_ylabel("Numero file")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_custom_comparison(light_results, resnet_results, output_dir):
    light_by_file = {r["file"]: r for r in light_results}
    resnet_by_file = {r["file"]: r for r in resnet_results}
    shared_files = sorted(set(light_by_file) & set(resnet_by_file))

    rows = []
    disagree_count = 0
    for f in shared_files:
        l = light_by_file[f]
        r = resnet_by_file[f]
        disagree = int(l["pred_label"] != r["pred_label"])
        if disagree:
            disagree_count += 1
        rows.append(
            {
                "file": f,
                "light_pred": l["pred_label"],
                "light_conf": round(l["confidence"], 4),
                "resnet_pred": r["pred_label"],
                "resnet_conf": round(r["confidence"], 4),
                "disagree": disagree,
                "min_conf": round(min(l["confidence"], r["confidence"]), 4),
            }
        )

    rows.sort(key=lambda x: (-x["disagree"], x["min_conf"]))
    csv_path = os.path.join(output_dir, "custom_critical_cases.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "light_pred",
                "light_conf",
                "resnet_pred",
                "resnet_conf",
                "disagree",
                "min_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Accordo", "Disaccordo"], [len(shared_files) - disagree_count, disagree_count])
    ax.set_title("Confronto modelli su file condivisi")
    ax.set_ylabel("Numero file")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "custom_model_disagreement.png"))
    plt.close(fig)

    print(f"Casi critici salvati: {csv_path}")


def evaluate_on_test(models, device, batch_size, output_dir):
    _, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    for split_name, loader in [("validation", val_loader), ("test", test_loader)]:
        print(f"\n=== Valutazione su {split_name} ===")
        for model_name, model in models.items():
            y_true, y_pred, _ = predict_loader(model, loader, device)
            acc = accuracy_percent(y_true, y_pred)
            print(f"[{model_name}] Accuracy: {acc:.2f}%")
            print(classification_report(y_true, y_pred, digits=2, target_names=CLASS_NAMES, zero_division=0))

            cm_path = os.path.join(output_dir, f"{split_name}_{model_name}_confusion.png")
            save_confusion_matrix(y_true, y_pred, f"Confusion Matrix - {model_name} ({split_name})", cm_path)

            metrics_path = os.path.join(output_dir, f"{split_name}_{model_name}_per_class_metrics.png")
            save_per_class_metrics(
                y_true,
                y_pred,
                f"Metriche per classe - {model_name} ({split_name})",
                metrics_path,
            )


def evaluate_on_custom(models, wav_dir, sr, output_dir, labels_from, prefix_sep):
    if not wav_dir:
        raise ValueError("Per source=custom devi passare --wav-dir")
    if not os.path.isdir(wav_dir):
        raise FileNotFoundError(f"Cartella WAV non trovata: {wav_dir}")

    wav_paths = sorted(glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True))
    if not wav_paths:
        raise FileNotFoundError(f"Nessun file .wav trovato in: {wav_dir}")

    mel_transform, amp_to_db = build_audio_transforms(sample_rate=sr)

    model_results = {}
    for model_name, model in models.items():
        print(f"\n=== Predizioni custom WAV - {model_name} ===")
        results = predict_wav_files(model, wav_paths, next(model.parameters()).device, sr, mel_transform, amp_to_db)
        model_results[model_name] = results

        save_pred_distribution(results, model_name, os.path.join(output_dir, f"custom_{model_name}_pred_distribution.png"))
        save_confidence_hist(results, model_name, os.path.join(output_dir, f"custom_{model_name}_confidence_hist.png"))

        avg_conf = np.mean([r["confidence"] for r in results])
        print(f"[{model_name}] File processati: {len(results)} | Confidenza media: {avg_conf:.3f}")

    if "light" in model_results and "resnet" in model_results:
        save_custom_comparison(model_results["light"], model_results["resnet"], output_dir)

    if labels_from != "none":
        print("\nModalita label attiva: genero anche error analysis supervisionata.")
        for model_name, results in model_results.items():
            y_true = []
            y_pred = []
            error_rows = []

            for row in results:
                true_label = infer_true_label(row["file"], labels_from, prefix_sep)
                if true_label is None:
                    continue
                true_idx = CLASS_NAMES.index(true_label)
                y_true.append(true_idx)
                y_pred.append(row["pred_idx"])
                if true_idx != row["pred_idx"]:
                    error_rows.append(
                        {
                            "file": row["file"],
                            "true_label": true_label,
                            "pred_label": row["pred_label"],
                            "confidence": round(row["confidence"], 4),
                        }
                    )

            if not y_true:
                print(f"[{model_name}] Nessuna label valida trovata dai file (labels_from={labels_from}).")
                continue

            acc = accuracy_percent(y_true, y_pred)
            print(f"[{model_name}] Accuracy custom (labeled): {acc:.2f}%")
            print(classification_report(y_true, y_pred, digits=2, target_names=CLASS_NAMES, zero_division=0))

            cm_path = os.path.join(output_dir, f"custom_{model_name}_confusion_labeled.png")
            save_confusion_matrix(y_true, y_pred, f"Confusion Matrix - {model_name} (custom labeled)", cm_path)

            metrics_path = os.path.join(output_dir, f"custom_{model_name}_per_class_metrics_labeled.png")
            save_per_class_metrics(
                y_true,
                y_pred,
                f"Metriche per classe - {model_name} (custom labeled)",
                metrics_path,
            )

            errors_path = os.path.join(output_dir, f"custom_{model_name}_errors.csv")
            with open(errors_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["file", "true_label", "pred_label", "confidence"])
                writer.writeheader()
                writer.writerows(error_rows)
            print(f"[{model_name}] Errori salvati in: {errors_path}")
    else:
        print("\nModalita senza label: niente accuracy/F1, usa i grafici e custom_critical_cases.csv per i casi critici.")


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    device = resolve_device(args.device)
    print(f"Uso il dispositivo: {device}")

    models = {}
    for model_name in args.models:
        ckpt = args.light_path if model_name == "light" else args.resnet_path
        try:
            models[model_name] = load_model(model_name, ckpt, device, num_classes=len(CLASS_NAMES))
            print(f"Caricato modello {model_name} da {ckpt}")
        except ImportError as e:
            print(f"Skipping {model_name}: {e}")
        except FileNotFoundError as e:
            print(f"Skipping {model_name}: {e}")

    if not models:
        raise RuntimeError("Nessun modello disponibile per la valutazione")

    if args.source in ["test", "both"]:
        evaluate_on_test(models, device, args.batch_size, args.output_dir)

    if args.source in ["custom", "both"]:
        evaluate_on_custom(models, args.wav_dir, args.sr, args.output_dir, args.labels_from, args.prefix_sep)


if __name__ == "__main__":
    main()