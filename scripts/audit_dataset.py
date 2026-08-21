from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thoracic_ai.constants import NIH_14_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an NIH ChestXray14 dataset directory")
    parser.add_argument("--data-root", required=True)
    return parser.parse_args()


def main() -> None:
    root = Path(parse_args().data_root).expanduser().resolve()
    metadata_path = root / "Data_Entry_2017.csv"
    train_val_path = root / "train_val_list.txt"
    test_path = root / "test_list.txt"
    required = [metadata_path, train_val_path, test_path]
    missing_files = [str(path) for path in required if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required dataset files: {missing_files}")

    metadata = pd.read_csv(metadata_path)
    image_paths = list(root.rglob("*.png"))
    image_names = {path.name for path in image_paths}
    missing_images = set(metadata["Image Index"]).difference(image_names)

    print(f"Dataset root: {root}")
    print(f"Metadata rows: {len(metadata):,}")
    print(f"Unique patients: {metadata['Patient ID'].nunique():,}")
    print(f"PNG images found: {len(image_paths):,}")
    print(f"Metadata images missing on disk: {len(missing_images):,}")
    print(f"Official train/validation images: {_line_count(train_val_path):,}")
    print(f"Official test images: {_line_count(test_path):,}")
    print("\nDisease prevalence")
    for class_name in NIH_14_CLASSES:
        count = metadata["Finding Labels"].fillna("").str.split("|").apply(
            lambda labels: class_name in labels
        ).sum()
        print(f"{class_name:20s} {count:>7,}")

    if missing_images:
        raise RuntimeError("Dataset audit failed because metadata images are missing")
    print("\nDataset audit passed.")


def _line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    main()

