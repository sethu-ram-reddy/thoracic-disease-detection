from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from thoracic_ai.config import DataConfig
from thoracic_ai.constants import IMAGENET_MEAN, IMAGENET_STD


def build_image_index(image_root: str | Path, cache_path: str | Path) -> pd.DataFrame:
    root = Path(image_root)
    cache = Path(cache_path)

    if cache.exists():
        indexed = pd.read_csv(cache)
        if not indexed.empty and Path(indexed.iloc[0]["image_path"]).exists():
            return indexed

    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    paths = [
        path
        for suffix in ("*.png", "*.jpg", "*.jpeg")
        for path in root.rglob(suffix)
        if path.is_file()
    ]
    if not paths:
        raise FileNotFoundError(f"No chest X-ray images were found under {root}")

    indexed = pd.DataFrame(
        {"Image Index": [path.name for path in paths], "image_path": [str(path) for path in paths]}
    )
    duplicates = indexed["Image Index"].duplicated().sum()
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate image filenames under {root}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    indexed.to_csv(cache, index=False)
    return indexed


def encode_labels(metadata: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    required = {"Image Index", "Patient ID", "Finding Labels"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    encoded = metadata[["Image Index", "Patient ID", "Finding Labels"]].copy()
    label_sets = encoded["Finding Labels"].fillna("").str.split("|").apply(set)
    for class_name in classes:
        encoded[class_name] = label_sets.apply(
            lambda labels, target=class_name: float(target in labels)
        )
    return encoded


def prepare_splits(
    config: DataConfig,
    seed: int,
    cache_dir: str | Path,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    cache = Path(cache_dir)
    split_paths = {name: cache / f"{name}.csv" for name in ("train", "validation", "test")}

    if not force and all(path.exists() for path in split_paths.values()):
        splits = {name: pd.read_csv(path) for name, path in split_paths.items()}
        _validate_split_integrity(splits)
        return splits

    metadata_path = Path(config.metadata_csv)
    if not metadata_path.exists():
        raise FileNotFoundError(f"NIH metadata file does not exist: {metadata_path}")

    image_index = build_image_index(config.image_root, cache / "image_index.csv")
    metadata = encode_labels(pd.read_csv(metadata_path), config.classes)
    metadata = metadata.merge(image_index, on="Image Index", how="inner", validate="one_to_one")

    train_val_names = _read_name_list(config.train_val_list)
    test_names = _read_name_list(config.test_list)
    overlap = train_val_names.intersection(test_names)
    if overlap:
        raise ValueError(f"Official split files overlap on {len(overlap)} images")

    train_val = metadata[metadata["Image Index"].isin(train_val_names)].copy()
    test = metadata[metadata["Image Index"].isin(test_names)].copy()

    patient_targets = train_val.groupby("Patient ID")[config.classes].max().reset_index()
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=config.validation_fraction,
        random_state=seed,
    )
    train_indices, validation_indices = next(
        splitter.split(np.zeros((len(patient_targets), 1)), patient_targets[config.classes].values)
    )
    train_patients = set(patient_targets.iloc[train_indices]["Patient ID"])
    validation_patients = set(patient_targets.iloc[validation_indices]["Patient ID"])

    train = train_val[train_val["Patient ID"].isin(train_patients)].copy()
    validation = train_val[train_val["Patient ID"].isin(validation_patients)].copy()
    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    _validate_split_integrity(splits)

    cache.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(split_paths[name], index=False)
    return splits


class ChestXrayDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, classes: list[str], transform=None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.classes = classes
        self.transform = transform
        self.image_paths = self.frame["image_path"].tolist()
        self.labels = self.frame[classes].to_numpy(dtype=np.float32)
        self.image_names = self.frame["Image Index"].tolist()
        self.patient_ids = self.frame["Patient ID"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        with Image.open(self.image_paths[index]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "labels": torch.from_numpy(self.labels[index]),
            "image_name": self.image_names[index],
            "patient_id": self.patient_ids[index],
        }


def create_dataloaders(
    splits: dict[str, pd.DataFrame],
    config: DataConfig,
) -> dict[str, DataLoader]:
    train_transform, evaluation_transform = build_transforms(config.image_size)
    datasets = {
        "train": ChestXrayDataset(splits["train"], config.classes, train_transform),
        "validation": ChestXrayDataset(
            splits["validation"], config.classes, evaluation_transform
        ),
        "test": ChestXrayDataset(splits["test"], config.classes, evaluation_transform),
    }

    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        common["prefetch_factor"] = 4
    return {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=False, **common),
        "validation": DataLoader(datasets["validation"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def compute_positive_weights(train_frame: pd.DataFrame, classes: list[str]) -> torch.Tensor:
    positives = train_frame[classes].sum(axis=0).to_numpy(dtype=np.float32)
    negatives = len(train_frame) - positives
    return torch.tensor(negatives / np.clip(positives, 1.0, None), dtype=torch.float32)


def split_summary(splits: dict[str, pd.DataFrame], classes: Iterable[str]) -> pd.DataFrame:
    rows = []
    for split_name, frame in splits.items():
        row: dict[str, object] = {
            "split": split_name,
            "images": len(frame),
            "patients": frame["Patient ID"].nunique(),
        }
        row.update({class_name: int(frame[class_name].sum()) for class_name in classes})
        rows.append(row)
    return pd.DataFrame(rows)


def _read_name_list(path: str | Path) -> set[str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Official NIH split file does not exist: {source}")
    return {
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _validate_split_integrity(splits: dict[str, pd.DataFrame]) -> None:
    patient_sets = {name: set(frame["Patient ID"].unique()) for name, frame in splits.items()}
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        overlap = patient_sets[left].intersection(patient_sets[right])
        if overlap:
            raise ValueError(f"Patient leakage detected between {left} and {right}: {len(overlap)}")
