from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from thoracic_ai.config import ExperimentConfig
from thoracic_ai.data import ChestXrayDataset, build_transforms, prepare_splits
from thoracic_ai.engine import load_trained_model
from thoracic_ai.explainability import GradCAM, save_gradcam_figure
from thoracic_ai.models import create_model, gradcam_target_layer
from thoracic_ai.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DenseNet-121 Grad-CAM examples")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--samples", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if config.model.name != "densenet121":
        raise ValueError("Grad-CAM generation requires the DenseNet-121 config")
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
        config.data.image_root = str(root)
        config.data.metadata_csv = str(root / "Data_Entry_2017.csv")
        config.data.train_val_list = str(root / "train_val_list.txt")
        config.data.test_list = str(root / "test_list.txt")
    if args.output_dir:
        config.output_dir = str(Path(args.output_dir).expanduser().resolve())
    seed_everything(config.seed)

    experiment_dir = Path(config.output_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else experiment_dir / "best.pt"
    gradcam_dir = experiment_dir / "evaluation" / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_splits(config.data, config.seed, experiment_dir / "splits")
    _, evaluation_transform = build_transforms(config.data.image_size)
    dataset = ChestXrayDataset(splits["test"], config.data.classes, evaluation_transform)

    device = select_device()
    model = create_model(
        config.model.name,
        len(config.data.classes),
        pretrained=False,
        dropout=config.model.dropout,
    )
    load_trained_model(model, checkpoint_path, device)
    model.eval()
    gradcam = GradCAM(model, gradcam_target_layer(model, config.model.name))

    rng = np.random.default_rng(config.seed)
    positive_rows = splits["test"][config.data.classes].sum(axis=1).to_numpy() > 0
    candidate_indices = np.flatnonzero(positive_rows)
    selected = rng.choice(
        candidate_indices, size=min(args.samples, len(candidate_indices)), replace=False
    )

    for dataset_index in selected:
        sample = dataset[int(dataset_index)]
        image = sample["image"].unsqueeze(0).to(device)
        labels = sample["labels"].numpy()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(image))[0].cpu().numpy()

        positive_classes = np.flatnonzero(labels > 0)
        target_index = int(positive_classes[np.argmax(probabilities[positive_classes])])
        heatmap = gradcam(image, target_index)
        image_stem = Path(str(sample["image_name"])).stem
        save_gradcam_figure(
            image=sample["image"],
            heatmap=heatmap,
            class_name=config.data.classes[target_index],
            probability=float(probabilities[target_index]),
            output_path=gradcam_dir / f"{image_stem}_{config.data.classes[target_index]}.png",
        )

    gradcam.close()
    print(f"Saved {len(selected)} Grad-CAM figures to {gradcam_dir}")


if __name__ == "__main__":
    main()

