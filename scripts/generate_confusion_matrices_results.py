import json
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
RESULTS_DIR = ANALYSIS_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
IMPORTANT_DIR = RESULTS_DIR / "important"

METRICS_JSON_PATH = METRICS_DIR / "test_metrics_by_threshold.json"
OUTPUT_METRICS_PNG = METRICS_DIR / "confusion_matrices.png"
OUTPUT_IMPORTANT_PNG = IMPORTANT_DIR / "confusion_matrices.png"


ORDER = [
    ("threshold_default", "Threshold default"),
    ("threshold_optimal_youden", "Threshold optimal Youden"),
    ("threshold_optimal_f1", "Threshold optimal f1"),
    ("threshold_optimal_recall_70", "Threshold optimal recall 70"),
    ("threshold_optimal_recall_80", "Threshold optimal recall 80"),
    ("threshold_optimal_recall_90", "Threshold optimal recall 90"),
]


def _load_metrics(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _extract_matrix(entry: dict) -> list[list[int]]:
    # Preferred: matrix already stored in JSON.
    cm = entry.get("confusion_matrix")
    if isinstance(cm, dict) and "matrix" in cm:
        return cm["matrix"]

    # Fallback: reconstruct from tp/tn/fp/fn.
    tn = int(entry["tn"])
    fp = int(entry["fp"])
    fn = int(entry["fn"])
    tp = int(entry["tp"])
    return [[tn, fp], [fn, tp]]


def generate(output_paths: list[Path]) -> None:
    if not METRICS_JSON_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {METRICS_JSON_PATH}")

    data = _load_metrics(METRICS_JSON_PATH)
    thresholds = data.get("thresholds", {})

    items = []
    for key, label in ORDER:
        if key in thresholds:
            items.append((label, thresholds[key]))

    if not items:
        raise ValueError("Brak wpisów progów w test_metrics_by_threshold.json")

    n = len(items)
    cols = 3 if n > 4 else 2
    rows = ceil(n / cols)

    sns.set_style("white")
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows), constrained_layout=True)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, (label, entry) in enumerate(items):
        r = i // cols
        c = i % cols
        ax = axes[r][c]

        matrix = _extract_matrix(entry)
        thr = entry.get("threshold")
        title = label if thr is None else f"{label}\n(threshold={thr:.3f})"

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=1,
            linecolor="white",
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"], rotation=0)

    # Disable any unused axes
    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    for out in output_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)

    plt.close(fig)


def main() -> None:
    generate([OUTPUT_METRICS_PNG, OUTPUT_IMPORTANT_PNG])
    print(f"Zapisano: {OUTPUT_METRICS_PNG}")
    print(f"Zapisano: {OUTPUT_IMPORTANT_PNG}")


if __name__ == "__main__":
    main()
