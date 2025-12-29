import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"
TRAINING_DIR = RESULTS_DIR / "training"
HISTORY_PATH = TRAINING_DIR / "training_history_complete.json"
OUTPUT_PATH = TRAINING_DIR / "training_history_plots.png"


def load_history(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def plot_stage(ax_loss, ax_auc, stage_name: str, records):
    epochs = [r["epoch"] for r in records]
    train_loss = [r["train_loss"] for r in records]
    val_loss = [r["val_loss"] for r in records]
    train_auc = [r["train_auc"] for r in records]
    val_auc = [r["val_auc"] for r in records]

    # Loss
    ax_loss.plot(epochs, train_loss, label="Train loss", marker="o")
    ax_loss.plot(epochs, val_loss, label="Val loss", marker="s")
    ax_loss.set_title(f"{stage_name} – Loss")
    ax_loss.set_xlabel("Epoka")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)

    # AUC
    ax_auc.plot(epochs, train_auc, label="Train AUC", marker="o")
    ax_auc.plot(epochs, val_auc, label="Val AUC", marker="s")
    ax_auc.set_title(f"{stage_name} – AUC")
    ax_auc.set_xlabel("Epoka")
    ax_auc.set_ylabel("AUC-ROC")
    ax_auc.set_ylim(0.4, 1.0)
    ax_auc.grid(True, alpha=0.3)


def main():
    history = load_history(HISTORY_PATH)

    stage1 = history.get("stage1", [])
    stage2 = history.get("stage2", [])
    stage3 = history.get("stage3", [])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    (ax_s1_loss, ax_s1_auc), (ax_s2_loss, ax_s2_auc), (ax_s3_loss, ax_s3_auc) = axes

    if stage1:
        plot_stage(ax_s1_loss, ax_s1_auc, "Etap 1", stage1)
    if stage2:
        plot_stage(ax_s2_loss, ax_s2_auc, "Etap 2", stage2)
    if stage3:
        plot_stage(ax_s3_loss, ax_s3_auc, "Etap 3", stage3)

    # Wspólne legendy
    for row in axes:
        for ax in row:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=8)

    fig.suptitle("Historia trenowania – loss i AUC (Etapy 1–3)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Zapisano wykres do:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
