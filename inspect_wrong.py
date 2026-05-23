#!/usr/bin/env python3
"""Inspect and optionally play or copy audio files listed in eval_out/wrong_<model>.csv

Usage examples:
    # copy misclassified WAVs to a folder for manual inspection (recommended on WSL)
    python3 inspect_wrong.py --model light --copy
"""
import argparse
import csv
import shutil
from pathlib import Path
import sys


def build_args():
    p = argparse.ArgumentParser(description="Inspect misclassified audio files")
    p.add_argument("--model", choices=["light", "resnet"], default="light")
    p.add_argument("--csv", default="eval_out/wrong_{model}.csv")
    p.add_argument("--dataset-root", default="dataset_audio/SpeechCommands/speech_commands_v0.02")
    p.add_argument("--out-dir", default="eval_out/wrong_samples")
    p.add_argument("--copy", action="store_true", help="Copy misclassified files to --out-dir for manual inspection")
    return p.parse_args()


def find_file(dataset_root: Path, filename: str, rel_path: str | None = None) -> Path | None:
    # Prefer exact relative path from CSV when available.
    if rel_path:
        rel = rel_path.strip().replace('\\', '/')
        if rel:
            candidate = dataset_root / Path(rel)
            if candidate.exists():
                return candidate

    # Fallback: search by filename under dataset_root (may be expensive).
    for p in dataset_root.rglob(filename):
        return p
    return None



def normalize_conf(conf_str: str) -> str:
    # conf may be like '87.3%' or '0.873'
    s = conf_str.strip()
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
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    if args.copy:
        out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print('No misclassified samples found in CSV.')
        return

    print(f'Found {len(rows)} misclassified samples in {csv_path}')

    for i, r in enumerate(rows, 1):
        fname = r.get('file') or r.get('filename')
        rel_path = r.get('path')
        true = r.get('true')
        pred = r.get('pred')
        conf = normalize_conf(r.get('pred_confidence', ''))

        print(f"[{i}/{len(rows)}] {fname}  true={true}  pred={pred}  conf={conf}")

        found = find_file(dataset_root, fname, rel_path)
        if not found:
            print(f"  -> File not found under {dataset_root}: {fname}")
            continue

        print(f"  -> located at: {found}")

        if args.copy:
            safe_name = f"{i:04d}_{true}_as_{pred}_{conf.replace('%', 'pct')}_{fname}"
            dst = out_dir / safe_name
            try:
                shutil.copy2(found, dst)
                print(f"  -> copied to: {dst}")
            except Exception as e:
                print(f"  -> failed to copy: {e}")

    print('Done.')


if __name__ == '__main__':
    main()
