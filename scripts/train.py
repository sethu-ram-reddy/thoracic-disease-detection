from __future__ import annotations

import argparse
from pathlib import Path

import torch

from thoracic_ai.config import ExperimentConfig
from thoracic_ai.data import (
    compute_positive_weights,
    create_dataloaders,
    prepare_splits,
    split_summary,
)
from thoracic_ai.engine import train_model
from thoracic_ai.models import create_model
from thoracic_ai.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a thoracic disease classifier")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--data-root", help="Override the NIH dataset root")
    parser.add_argument("--output-dir", help="Override the experiment output directory")
    parser.add_argument("--force-splits", action="store_true", help="Rebuild cached data splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    apply_overrides(config, args.data_root, args.output_dir)
    seed_everything(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "resolved_config.json")

    splits = prepare_splits(
        config=config.data,
        seed=config.seed,
        cache_dir=output_dir / "splits",
        force=args.force_splits,
    )
    summary = split_summary(splits, config.data.classes)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    print(summary.to_string(index=False))

    loaders = create_dataloaders(splits, config.data)
    device = select_device()
    model = create_model(
        name=config.model.name,
        num_classes=len(config.data.classes),
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    ).to(device)

    positive_weights = compute_positive_weights(splits["train"], config.data.classes).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    print(f"Training {config.model.name} on {device} with {len(config.data.classes)} labels")

    result = train_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        config=config,
        device=device,
    )
    print(result)


def apply_overrides(
    config: ExperimentConfig,
    data_root: str | None,
    output_dir: str | None,
) -> None:
    if data_root:
        root = Path(data_root).expanduser().resolve()
        config.data.image_root = str(root)
        config.data.metadata_csv = str(root / "Data_Entry_2017.csv")
        config.data.train_val_list = str(root / "train_val_list.txt")
        config.data.test_list = str(root / "test_list.txt")
    if output_dir:
        config.output_dir = str(Path(output_dir).expanduser().resolve())
    config.validate()


if __name__ == "__main__":
    main()

