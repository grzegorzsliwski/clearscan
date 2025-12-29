import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
RESULTS_DIR = ANALYSIS_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
DATA_SPLITS_DIR = ANALYSIS_DIR / "data_splits"

PREDICTIONS_PATH = PREDICTIONS_DIR / "all_predictions.csv"
TEST_SPLIT_PATH = DATA_SPLITS_DIR / "test_split.csv"
THRESHOLD_CONFIG_PATH = METRICS_DIR / "threshold_optimization.json"
OUTPUT_PATH = METRICS_DIR / "test_metrics_by_threshold.json"


def compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else None

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "clinical_metrics": {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
        },
    }


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z predykcjami: {PREDICTIONS_PATH}")
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku test_split: {TEST_SPLIT_PATH}")
    if not THRESHOLD_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z konfiguracją progów: {THRESHOLD_CONFIG_PATH}")

    # Wczytanie predykcji
    preds = pd.read_csv(PREDICTIONS_PATH)

    # Wczytanie prawdziwych etykiet z test_split (to źródło prawdy)
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    test_labels = test_df[["Image Index", "Mass"]].rename(columns={"Mass": "true_label_from_test"})

    merged = preds.merge(
        test_labels,
        left_on="image_index",
        right_on="Image Index",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(preds):
        raise ValueError(
            f"Po złączeniu z test_split liczba wierszy się zmieniła: "
            f"{len(preds)} -> {len(merged)}. Sprawdź spójność danych."
        )

    if "true_label" in merged.columns:
        mismatch = (merged["true_label"].astype(int) != merged["true_label_from_test"].astype(int)).sum()
        if mismatch > 0:
            print(f"Ostrzeżenie: {mismatch} przykładów ma różne etykiety w all_predictions i test_split. "
                  f"Do metryk używam etykiet z test_split.")

    y_true = merged["true_label_from_test"].astype(int).to_numpy()
    y_score = merged["predicted_prob"].astype(float).to_numpy()

    # AUC jest niezależne od progu
    auc_roc = roc_auc_score(y_true, y_score)

    # Wczytanie progów
    with open(THRESHOLD_CONFIG_PATH, "r") as f:
        thr_cfg = json.load(f)

    thresholds = {
        "threshold_default": float(thr_cfg.get("default_threshold", 0.5)),
    }

    if "optimal_threshold_youden" in thr_cfg:
        thresholds["threshold_optimal_youden"] = float(thr_cfg["optimal_threshold_youden"])
    if "optimal_threshold_f1" in thr_cfg:
        thresholds["threshold_optimal_f1"] = float(thr_cfg["optimal_threshold_f1"])
    if "optimal_threshold_recall_70" in thr_cfg:
        thresholds["threshold_optimal_recall_70"] = float(thr_cfg["optimal_threshold_recall_70"])

    results = {
        "auc_roc": auc_roc,
        "thresholds": {},
    }

    for name, thr in thresholds.items():
        y_pred = (y_score >= thr).astype(int)
        metrics = compute_confusion_metrics(y_true, y_pred)
        metrics["threshold"] = thr
        results["thresholds"][name] = metrics

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Zapisano metryki do: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
