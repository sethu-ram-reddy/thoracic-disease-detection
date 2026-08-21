from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from thoracic_ai.constants import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def __call__(self, image: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[:, class_index].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = functional.relu(heatmap)
        heatmap = functional.interpolate(
            heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        heatmap = heatmap[0, 0]
        heatmap -= heatmap.min()
        heatmap /= heatmap.max().clamp_min(1e-8)
        return heatmap.detach().cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _save_activations(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    restored = (image * std + mean).clamp(0, 1)
    return restored.detach().cpu().permute(1, 2, 0).numpy()


def save_gradcam_figure(
    image: torch.Tensor,
    heatmap: np.ndarray,
    class_name: str,
    probability: float,
    output_path: str | Path,
) -> None:
    original = denormalize(image)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Chest X-ray")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(original, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title(f"{class_name}: {probability:.3f}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)

