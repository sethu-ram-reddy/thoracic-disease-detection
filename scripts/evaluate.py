from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thoracic_ai.config import ExperimentConfig
from thoracic_ai.data import create_dataloaders, prepare_splits
from thoracic_ai.engine import load_trained_model, run_inference
from thoracic_ai.metrics import calculate_metrics, save_metrics, tune_thresholds
from thoracic_ai.models import create_model
from thoracic_ai.utils import seed_everything, select_device
from thoracic_ai.visualization import (
    plot_class_metrics,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained thoracic disease classifier")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to <output_dir>/best.pt")
    parser.add_argument("--data-root", help="Override the NIH dataset root")
    parser.add_argument("--output-dir", help="Override the experiment output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
        config.data.image_root = str(root)
        config.data.metadata_csv = str(root / "Data_Entry_2017.csv")
        config.data.train_val_list = str(root / "train_val_list.txt")
        config.data.test_list = str(root / "test_list.txt")
    if args.output_dir:
        config.output_dir = str(Path(args.output_dir).expanduser().resolve())
    config.validate()
    seed_everything(config.seed)

    experiment_dir = Path(config.output_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else experiment_dir / "best.pt"
    evaluation_dir = experiment_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_splits(config.data, config.seed, experiment_dir / "splits")
    loaders = create_dataloaders(splits, config.data)
    device = select_device()
    model = create_model(
        config.model.name,
        len(config.data.classes),
        pretrained=False,
        dropout=config.model.dropout,
    )
    load_trained_model(model, checkpoint_path, device)

    amp_enabled = config.training.mixed_precision and device.type == "cuda"
    _, validation_targets, validation_probabilities, _ = run_inference(
        model, loaders["validation"], device, amp_enabled=amp_enabled, description="Validation"
    )
    thresholds = tune_thresholds(validation_targets, validation_probabilities)

    _, test_targets, test_probabilities, image_names = run_inference(
        model, loaders["test"], device, amp_enabled=amp_enabled, description="Test"
    )
    summary, class_metrics = calculate_metrics(
        test_targets, test_probabilities, config.data.classes, thresholds
    )
    save_metrics(evaluation_dir, summary, class_metrics)
    (evaluation_dir / "thresholds.json").write_text(
        json.dumps(
            dict(zip(config.data.classes, thresholds.astype(float), strict=True)), indent=2
        ),
        encoding="utf-8",
    )

    predictions = pd.DataFrame({"image_name": image_names})
    for index, class_name in enumerate(config.data.classes):
        predictions[f"target_{class_name}"] = test_targets[:, index].astype(int)
        predictions[f"probability_{class_name}"] = test_probabilities[:, index]
    predictions.to_csv(evaluation_dir / "test_predictions.csv", index=False)

    plot_roc_curves(
        test_targets, test_probabilities, config.data.classes, evaluation_dir / "roc_curves.png"
    )
    plot_precision_recall_curves(
        test_targets,
        test_probabilities,
        config.data.classes,
        evaluation_dir / "precision_recall_curves.png",
    )
    plot_class_metrics(class_metrics, evaluation_dir / "class_metrics.png")
    history_path = experiment_dir / "history.csv"
    if history_path.exists():
        plot_training_history(history_path, evaluation_dir / "training_history.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
