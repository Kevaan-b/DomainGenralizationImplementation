"""Evaluation and statistically correct multi-seed aggregation."""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.nn import functional as functional


@torch.no_grad()
def evaluate(method, loader, device: torch.device) -> dict[str, float]:  # type: ignore[no-untyped-def]
    method.eval()
    correct = total = 0
    losses, class_correct, class_total = [], torch.zeros(10), torch.zeros(10)
    for batch in loader:
        images, labels = batch["image"].to(device, non_blocking=device.type == "cuda"), batch["label"].to(device, non_blocking=device.type == "cuda")
        logits = method.predict(images)
        losses.append(functional.cross_entropy(logits, labels, reduction="sum").item())
        predicted = logits.argmax(1)
        correct += (predicted == labels).sum().item()
        total += len(labels)
        for class_id in range(10):
            mask = labels == class_id
            class_total[class_id] += mask.sum().cpu()
            class_correct[class_id] += (predicted[mask] == labels[mask]).sum().cpu()
    return {"accuracy": correct / total, "cross_entropy": sum(losses) / total,
            "mean_per_class_accuracy": (class_correct / class_total.clamp_min(1)).mean().item()}


def aggregate(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return {"mean": mean, "std": math.sqrt(variance), "standard_error": math.sqrt(variance / len(values))}
