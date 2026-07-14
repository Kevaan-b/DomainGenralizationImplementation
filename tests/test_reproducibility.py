import random

import numpy as np
import torch
import yaml

from dg.training.reproducibility import environment_metadata, resolve_device, seed_everything


def test_seed_controls_python_numpy_and_torch_and_records_environment():
    seed_everything(42)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    seed_everything(42)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == second
    device = resolve_device("auto")
    metadata = environment_metadata(device, deterministic=True)
    assert metadata["torch"] == str(torch.__version__)
    assert "torchvision" in metadata
    yaml.safe_dump(metadata)
