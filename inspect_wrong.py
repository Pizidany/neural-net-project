import argparse
import csv
import shutil
from pathlib import Path
import sys
import os

CSV_TEMPLATE = "eval_out/wrong_{model}.csv"
DATASET_ROOT = Path("dataset_audio/SpeechCommands/speech_commands_v0.02")
OUT_DIR = Path("eval_out/wrong_samples")


def build_args():
    p = argparse.ArgumentParser(description="Ispeziona WAV misclassificati")
    p.add_argument('--model', type=str, required=True, choices=['light', 'resnet', 'mobilenet', 'alexnet'],
                   help="Scegli 'light', 'resnet', 'mobilenet' o 'alexnet'")
    return p.parse_args()


def find_file(dataset_root: Path, filename: str, rel_path: str | None = None) -> Path | None:
    if rel_path:
        rel = rel_path.strip().replace('\\', '/')
        if rel:
            candidate = dataset_root / Path(rel)
            if candidate.exists():
                return candidate

    for p in dataset_root.rglob(filename):
        return p
    return None


def normalize_conf(conf_str: str) -> str:
    s = (conf_str or '').strip()
    if not s:
        return ""
    if s.endswith('%'):
        return s
    try:
        f = float(s)
        if f <= 1.0:
            return f"{f*100:.1f}%"
        else:
            return f"{f:.1f}%"
    except Exception:
        return s


def main():
    args = build_args()
    csv_path = Path(args.csv.format(model=args.model))
    if not csv_path.exists():
        print(f"CSV non trovato: {csv_path}")
        sys.exit(1)

    dataset_root = DATASET_ROOT
    out_dir = OUT_DIR

    rows = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("Nessun campione misclassificato nel CSV.")
        return

    print(f"Trovati {len(rows)} campioni misclassificati in {csv_path}")

    for i, r in enumerate(rows, 1):
        fname = r.get('file') or r.get('filename')
        rel_path = r.get('path')
        true = r.get('true')
        pred = r.get('pred')
        conf = normalize_conf(r.get('pred_confidence', ''))

        print(f"[{i}/{len(rows)}] {fname}  true={true}  pred={pred}  conf={conf}")

        found = find_file(dataset_root, fname, rel_path)
        if not found:
            print(f"  -> File non trovato sotto {dataset_root}: {fname}")
            continue

        print(f"  -> localizzato in: {found}")

        if args.copy:
            safe_name = f"{i:04d}_{true}_as_{pred}_{conf.replace('%', 'pct')}_{fname}"
            dst = out_dir / safe_name
            try:
                shutil.copy2(found, dst)
                print(f"  -> copiato in: {dst}")
            except Exception as e:
                print(f"  -> copia fallita: {e}")

    print("Fatto.")


if __name__ == '__main__':
    main()
