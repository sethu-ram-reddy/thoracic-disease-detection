from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed model evaluations")
    parser.add_argument("--densenet-dir", required=True)
    parser.add_argument("--vit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = {
        "DenseNet-121": Path(args.densenet_dir) / "evaluation",
        "ViT-B/16": Path(args.vit_dir) / "evaluation",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    class_frames = []
    for model_name, evaluation_dir in experiments.items():
        summary = json.loads((evaluation_dir / "summary_metrics.json").read_text(encoding="utf-8"))
        summaries.append({"model": model_name, **summary})
        class_frame = pd.read_csv(evaluation_dir / "class_metrics.csv")
        class_frame.insert(0, "model", model_name)
        class_frames.append(class_frame)

    summary_frame = pd.DataFrame(summaries)
    class_frame = pd.concat(class_frames, ignore_index=True)
    summary_frame.to_csv(output_dir / "model_summary.csv", index=False)
    class_frame.to_csv(output_dir / "disease_level_comparison.csv", index=False)

    melted = summary_frame.melt(
        id_vars="model",
        value_vars=["macro_auroc", "macro_auprc", "macro_f1"],
        var_name="metric",
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="model", ax=axis)
    axis.set(title="DenseNet-121 vs ViT-B/16", xlabel="Metric", ylabel="Score", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()

