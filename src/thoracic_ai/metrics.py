from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    candidates = grid if grid is not None else np.arange(0.05, 0.96, 0.05)
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)

    for class_index in range(targets.shape[1]):
        class_targets = targets[:, class_index]
        if np.unique(class_targets).size < 2:
            continue
        scores = [
            f1_score(class_targets, probabilities[:, class_index] >= threshold, zero_division=0)
            for threshold in candidates
        ]
        thresholds[class_index] = float(candidates[int(np.argmax(scores))])
    return thresholds


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    thresholds: np.ndarray | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    _validate_arrays(targets, probabilities, classes)
    thresholds = thresholds if thresholds is not None else np.full(len(classes), 0.5)
    predictions = probabilities >= thresholds.reshape(1, -1)

    rows = []
    for index, class_name in enumerate(classes):
        class_targets = targets[:, index]
        class_probabilities = probabilities[:, index]
        class_predictions = predictions[:, index]
        rows.append(
            {
                "class": class_name,
                "auroc": _safe_metric(roc_auc_score, class_targets, class_probabilities),
                "auprc": _safe_metric(
                    average_precision_score, class_targets, class_probabilities
                ),
                "precision": precision_score(
                    class_targets, class_predictions, zero_division=0
                ),
                "recall": recall_score(class_targets, class_predictions, zero_division=0),
                "f1": f1_score(class_targets, class_predictions, zero_division=0),
                "threshold": float(thresholds[index]),
                "positives": int(class_targets.sum()),
            }
        )

    class_metrics = pd.DataFrame(rows)
    summary = {
        "macro_auroc": float(class_metrics["auroc"].mean()),
        "macro_auprc": float(class_metrics["auprc"].mean()),
        "macro_f1": float(class_metrics["f1"].mean()),
        "micro_auroc": _safe_metric(roc_auc_score, targets.ravel(), probabilities.ravel()),
        "micro_auprc": _safe_metric(
            average_precision_score, targets.ravel(), probabilities.ravel()
        ),
        "micro_f1": float(f1_score(targets.ravel(), predictions.ravel(), zero_division=0)),
    }
    return summary, class_metrics


def save_metrics(
    output_dir: str | Path,
    summary: dict[str, float],
    class_metrics: pd.DataFrame,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    class_metrics.to_csv(destination / "class_metrics.csv", index=False)


def _safe_metric(metric, targets: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(targets).size < 2:
        return float("nan")
    return float(metric(targets, scores))


def _validate_arrays(
    targets: np.ndarray, probabilities: np.ndarray, classes: list[str]
) -> None:
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have identical shapes")
    if targets.ndim != 2:
        raise ValueError("targets and probabilities must be two-dimensional")
    if targets.shape[1] != len(classes):
        raise ValueError("The array width must match the number of classes")

