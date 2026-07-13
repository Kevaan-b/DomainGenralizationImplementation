"""Seed, device, and environment metadata handling."""
from __future__ import annotations

import platform
import random
import sys
from typing import Any

import numpy as np
import torch
import torchvision


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no NVIDIA CUDA device is available.")
    return device


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def environment_metadata(device: torch.device, deterministic: bool) -> dict[str, Any]:
    return {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "platform": platform.platform(), "deterministic": deterministic}
