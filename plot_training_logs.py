"""Generate plots from the training history CSV files.

The script scans a directory for files named ``training_history_*.csv`` and
creates a compact set of figures that summarize the training behaviour of each
model.

Default output:
- ``training_history_grid.png``: per-model curves in a 2x2 grid
- ``validation_accuracy_comparison.png``: validation accuracy comparison
- ``validation_loss_comparison.png``: validation loss comparison
- ``training_loss_comparison.png``: training loss comparison
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("training_logs")
DEFAULT_OUTPUT_DIR = Path("Report") / "figures"


def pretty_model_name(model_name: str) -> str:
    mapping = {
        "light": "LightCNN",
        "resnet": "AudioResNet",
        "mobilenet": "AudioMobileNetV2",
        "alexnet": "AudioAlexNet",
    }
    return mapping.get(model_name, model_name.capitalize())


def load_history(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{csv_path} is empty")

    return {
        "epoch": [int(row["epoch"]) for row in rows],
        "train_loss": [float(row["train_loss"]) for row in rows],
        "validation_loss": [float(row["validation_loss"]) for row in rows],
        "validation_accuracy": [float(row["validation_accuracy"]) for row in rows],
    }


def summarize_history(model_name: str, history: dict[str, list[float]]) -> str:
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    validation_loss = history["validation_loss"]
    validation_accuracy = history["validation_accuracy"]

    best_index = max(range(len(validation_accuracy)), key=validation_accuracy.__getitem__)
    return (
        f"{pretty_model_name(model_name):<18} "
        f"final_epoch={epochs[-1]:>2} "
        f"final_train_loss={train_loss[-1]:.6f} "
        f"final_val_loss={validation_loss[-1]:.6f} "
        f"final_val_acc={validation_accuracy[-1]:.4f} "
        f"best_val_acc={validation_accuracy[best_index]:.4f} "
        f"best_epoch={epochs[best_index]:>2}"
    )


def plot_grid(histories: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    model_order = ["light", "resnet", "mobilenet", "alexnet"]
    colors = {"train": "#1f77b4", "val_loss": "#d62728", "val_acc": "#2ca02c"}

    for ax, model_name in zip(axes.flat, model_order):
        history = histories[model_name]
        epochs = history["epoch"]

        ax2 = ax.twinx()
        ax.plot(epochs, history["train_loss"], marker="o", label="Train loss", color=colors["train"])
        ax.plot(epochs, history["validation_loss"], marker="s", label="Val loss", color=colors["val_loss"])
        ax2.plot(
            epochs,
            history["validation_accuracy"],
            marker="^",
            linestyle="--",
            label="Val accuracy",
            color=colors["val_acc"],
        )

        ax.set_title(pretty_model_name(model_name))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax2.set_ylabel("Validation accuracy (%)")
        ax.grid(True, alpha=0.25)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="center right", fontsize=8)

    fig.suptitle("Training history per modello")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(
    histories: dict[str, dict[str, list[float]]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = {
        "light": "#1f77b4",
        "resnet": "#d62728",
        "mobilenet": "#2ca02c",
        "alexnet": "#9467bd",
    }

    for model_name, history in histories.items():
        ax.plot(
            history["epoch"],
            history[metric],
            marker="o",
            label=pretty_model_name(model_name),
            color=colors.get(model_name),
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate plots from training logs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with training_history_*.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where plots are saved")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("training_history_*.csv"))
    if not csv_files:
        raise SystemExit(f"No training_history_*.csv files found in {input_dir}")

    histories: dict[str, dict[str, list[float]]] = {}
    for csv_path in csv_files:
        model_name = csv_path.stem.replace("training_history_", "")
        histories[model_name] = load_history(csv_path)

    expected_order = ["light", "resnet", "mobilenet", "alexnet"]
    ordered_histories = {name: histories[name] for name in expected_order if name in histories}
    for name in histories:
        if name not in ordered_histories:
            ordered_histories[name] = histories[name]

    print("Training log summary")
    print("-" * 110)
    for model_name in ordered_histories:
        print(summarize_history(model_name, ordered_histories[model_name]))

    plot_grid(ordered_histories, output_dir / "training_history_grid.png")
    plot_metric_comparison(
        ordered_histories,
        metric="validation_accuracy",
        title="Validation accuracy by epoch",
        ylabel="Validation accuracy (%)",
        output_path=output_dir / "validation_accuracy_comparison.png",
    )
    plot_metric_comparison(
        ordered_histories,
        metric="validation_loss",
        title="Validation loss by epoch",
        ylabel="Validation loss",
        output_path=output_dir / "validation_loss_comparison.png",
    )
    plot_metric_comparison(
        ordered_histories,
        metric="train_loss",
        title="Training loss by epoch",
        ylabel="Train loss",
        output_path=output_dir / "training_loss_comparison.png",
    )

    print(f"\nFigures saved in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())