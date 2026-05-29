from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audio_utils import TARGET_SR, build_audio_transforms, load_audio_file, waveform_to_mel_db


DEFAULT_DATASET_ROOT = Path("dataset_audio") / "SpeechCommands" / "speech_commands_v0.02"
DEFAULT_OUTPUT_PATH = Path("Report") / "figures" / "mel_spectrogram_example.png"


def find_example_wav(dataset_root: Path, label: str | None) -> Path:
    if label:
        candidates = sorted((dataset_root / label).glob("*.wav"))
        if not candidates:
            raise SystemExit(f"No WAV files found for label '{label}' in {dataset_root / label}")
        return candidates[0]

    for wav_path in sorted(dataset_root.rglob("*.wav")):
        if "_background_noise_" not in wav_path.parts:
            return wav_path

    raise SystemExit(f"No WAV files found under {dataset_root}")


def create_figure(wav_path: Path, output_path: Path) -> None:
    waveform = load_audio_file(str(wav_path), target_sr=TARGET_SR)
    mel_transform, amp_to_db = build_audio_transforms(sample_rate=TARGET_SR)
    mel_db = waveform_to_mel_db(waveform, mel_transform, amp_to_db).squeeze(0).numpy()

    duration_seconds = waveform.shape[-1] / TARGET_SR
    time_axis = np.linspace(0.0, duration_seconds, mel_db.shape[1])

    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[time_axis[0], time_axis[-1], 0, mel_db.shape[0]],
    )

    ax.set_title(f"Mel spectrogram - {wav_path.parent.name}", pad=12, fontweight="semibold")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Bande Mel")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Intensità (dB)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a nice Mel spectrogram figure for the report.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of SpeechCommands speech_commands_v0.02",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional class label to pick a sample from, e.g. yes or stop",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path",
    )
    args = parser.parse_args()

    wav_path = find_example_wav(args.dataset_root, args.label)
    create_figure(wav_path, args.output)

    print(f"Saved {args.output} from {wav_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())