from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from thoracic_ai.config import ExperimentConfig
from thoracic_ai.metrics import calculate_metrics
from thoracic_ai.models import freeze_backbone, unfreeze_model


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.training.freeze_backbone_epochs > 0:
        freeze_backbone(model, config.model.name)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.epochs
    )
    amp_enabled = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_macro_auroc = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch_index in range(config.training.epochs):
        epoch_number = epoch_index + 1
        if epoch_index == config.training.freeze_backbone_epochs:
            unfreeze_model(model)

        start_time = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            gradient_clip_norm=config.training.gradient_clip_norm,
            amp_enabled=amp_enabled,
        )
        validation_loss, targets, probabilities, _ = run_inference(
            model=model,
            loader=loaders["validation"],
            device=device,
            criterion=criterion,
            amp_enabled=amp_enabled,
            description="Validation",
        )
        validation_summary, _ = calculate_metrics(
            targets=targets,
            probabilities=probabilities,
            classes=config.data.classes,
        )
        scheduler.step()

        record = {
            "epoch": float(epoch_number),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_macro_auroc": validation_summary["macro_auroc"],
            "validation_macro_auprc": validation_summary["macro_auprc"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "duration_seconds": float(time.perf_counter() - start_time),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        checkpoint = {
            "epoch": epoch_number,
            "model_name": config.model.name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": config.data.classes,
            "validation_summary": validation_summary,
            "config": config.to_dict(),
        }
        torch.save(checkpoint, output_dir / "last.pt")

        current_score = validation_summary["macro_auroc"]
        if np.isfinite(current_score) and current_score > best_macro_auroc:
            best_macro_auroc = current_score
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch_number:02d}/{config.training.epochs} | "
            f"train loss {train_loss:.4f} | validation loss {validation_loss:.4f} | "
            f"macro AUROC {validation_summary['macro_auroc']:.4f} | "
            f"macro AUPRC {validation_summary['macro_auprc']:.4f}"
        )

        if epochs_without_improvement >= config.training.early_stopping_patience:
            print(f"Early stopping after {epoch_number} epochs.")
            break

    return {
        "best_validation_macro_auroc": float(best_macro_auroc),
        "epochs_completed": float(len(history)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    gradient_clip_norm: float,
    amp_enabled: bool,
) -> float:
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().item())
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    amp_enabled: bool = False,
    description: str = "Inference",
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    losses = []
    target_batches = []
    probability_batches = []
    image_names: list[str] = []

    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            if criterion is not None:
                losses.append(float(criterion(logits, labels).item()))

        target_batches.append(labels.cpu().numpy())
        probability_batches.append(torch.sigmoid(logits).cpu().numpy())
        image_names.extend(batch["image_name"])

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    targets = np.concatenate(target_batches, axis=0)
    probabilities = np.concatenate(probability_batches, axis=0)
    return mean_loss, targets, probabilities, image_names


def load_trained_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint

