from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_training_history(history_path: str | Path, output_path: str | Path) -> None:
    history = pd.read_csv(history_path)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Train")
    axes[0].plot(
        history["epoch"], history["validation_loss"], marker="o", label="Validation"
    )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE loss")
    axes[0].legend()
    axes[1].plot(
        history["epoch"], history["validation_macro_auroc"], marker="o", label="AUROC"
    )
    axes[1].plot(
        history["epoch"], history["validation_macro_auprc"], marker="o", label="AUPRC"
    )
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_roc_curves(
    targets: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    output_path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    for index, class_name in enumerate(classes):
        if np.unique(targets[:, index]).size < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            targets[:, index], probabilities[:, index]
        )
        axis.plot(false_positive_rate, true_positive_rate, label=class_name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Test ROC curves")
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_precision_recall_curves(
    targets: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    output_path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    for index, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(targets[:, index], probabilities[:, index])
        axis.plot(recall, precision, label=class_name)
    axis.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curves")
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_class_metrics(class_metrics: pd.DataFrame, output_path: str | Path) -> None:
    melted = class_metrics.melt(
        id_vars="class", value_vars=["auroc", "auprc", "f1"], var_name="metric"
    )
    figure, axis = plt.subplots(figsize=(12, 5))
    sns.barplot(data=melted, x="class", y="value", hue="metric", ax=axis)
    axis.set(xlabel="Disease", ylabel="Score", title="Disease-level test performance", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    _save_figure(figure, output_path)


def _save_figure(figure, output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)

