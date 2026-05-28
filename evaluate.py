import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from dataset import LABELS, SpeechCommandsPipeline
from models import AudioResNet, LightCNN

OUT_DIR = "eval_out"
CHECKPOINT_DIR = "checkpoints"


def evaluate(model_name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Carica il modello migliore salvato durante il training
    checkpoint = os.path.join(CHECKPOINT_DIR, f"best_{model_name}.pth")

    if model_name == "light":
        model = LightCNN(num_classes=len(LABELS))
    elif model_name == "resnet":
        model = AudioResNet(num_classes=len(LABELS))
    else:
        raise ValueError("Modello non riconosciuto. Scegli tra 'light' e 'resnet'")

    # Carica i pesi del modello e mettilo in modalità eval
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    # Crea un dataloader per il test set con return_path=True per ottenere i path dei file audio
    test_loader = DataLoader(SpeechCommandsPipeline(subset="testing", return_path=True), batch_size=64, shuffle=False)
    y_true, y_pred = [], []
    wrong_rows = []

    with torch.no_grad():
        # Il dataset di test è organizzato in sottocartelle per ogni classe,
        # quindi per l'analisi degli errori è utile avere il path completo del file audio
        dataset_root = os.path.join('dataset_audio', 'SpeechCommands', 'speech_commands_v0.02')

        # Iterazione sui batch di test
        for audio, labels, file_paths in test_loader:
            audio = audio.to(device)
            outputs = model(audio)
            # Calcolo delle probabilità e delle predizioni
            probs = torch.softmax(outputs, dim=1)
            preds_tensor = outputs.argmax(dim=1)
            preds = preds_tensor.cpu().tolist()
            # Ottieni la confidenza della predizione corretta per ogni campione
            pred_confidences = probs.gather(1, preds_tensor.unsqueeze(1)).squeeze(1).cpu().tolist()
            labels_list = labels.tolist()

            y_true.extend(labels_list)
            y_pred.extend(preds)

            # Analisi degli errori: se la predizione è errata, salva i dettagli in wrong_rows
            for i, pred in enumerate(preds):
                if pred != labels_list[i]:
                    # Costruiamo un percorso relativo al dataset per rendere i risultati più leggibili,
                    # ma se qualcosa va storto usiamo il path completo
                    raw_path = str(file_paths[i]).replace('\\', '/')
                    try:
                        rel_path = os.path.relpath(raw_path, start=dataset_root).replace('\\', '/')
                    except Exception:
                        rel_path = raw_path
                    wrong_rows.append([
                        os.path.basename(rel_path),
                        rel_path,
                        LABELS[labels_list[i]],
                        LABELS[pred],
                        f"{pred_confidences[i]*100:.1f}%",
                    ])

    print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

    # Salva i risultati degli errori in un file CSV per un'analisi più approfondita
    with open(os.path.join(OUT_DIR, f"wrong_{model_name}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "path", "true", "pred", "pred_confidence"])
        writer.writerows(wrong_rows)

    # Costruisci e salva la matrice di confusione
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix - {model_name}")
    accuracy = 100 * sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    plt.title(f"Confusion Matrix - {model_name}\nAccuracy: {accuracy:.2f}%")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"confusion_{model_name}.png"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valutazione semplice dei modelli audio")
    parser.add_argument("model", nargs="?", choices=["light", "resnet"], default="light")
    args = parser.parse_args()
    
    evaluate(model_name=args.model)
