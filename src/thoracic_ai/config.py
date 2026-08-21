from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    metadata_csv: str
    image_root: str
    train_val_list: str
    test_list: str
    classes: list[str]
    validation_fraction: float = 0.15
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 8

    def expand_paths(self) -> None:
        self.metadata_csv = _expand(self.metadata_csv)
        self.image_root = _expand(self.image_root)
        self.train_val_list = _expand(self.train_val_list)
        self.test_list = _expand(self.test_list)


@dataclass
class ModelConfig:
    name: str
    pretrained: bool = True
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    epochs: int = 8
    freeze_backbone_epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    mixed_precision: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    output_dir: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        config = cls(
            experiment_name=raw["experiment_name"],
            seed=int(raw.get("seed", 42)),
            output_dir=_expand(raw["output_dir"]),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
        )
        config.data.expand_paths()
        config.validate()
        return config

    def validate(self) -> None:
        if not self.data.classes:
            raise ValueError("At least one target class is required.")
        if len(self.data.classes) != len(set(self.data.classes)):
            raise ValueError("Target classes must be unique.")
        if not 0 < self.data.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.data.batch_size < 1 or self.data.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers cannot be negative.")
        if self.model.name not in {"densenet121", "vit_b16"}:
            raise ValueError(f"Unsupported model: {self.model.name}")
        if self.training.epochs < 1:
            raise ValueError("epochs must be at least 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _expand(value: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(value)))

