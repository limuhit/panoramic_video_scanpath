import json
from pathlib import Path

import torch


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def safe_torch_load(checkpoint_path, map_location, allow_unsafe=False):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if allow_unsafe:
            return torch.load(checkpoint_path, map_location=map_location)
        raise RuntimeError(
            "Safe checkpoint loading requires a PyTorch version with weights_only support. "
            "Upgrade PyTorch, or pass --allow-unsafe-checkpoint only for checkpoints you trust."
        ) from exc
    except Exception as exc:
        if allow_unsafe:
            return torch.load(checkpoint_path, map_location=map_location)
        raise RuntimeError(
            f"Could not safely load checkpoint {checkpoint_path}. "
            "Use only trusted checkpoints, or pass --allow-unsafe-checkpoint for legacy files."
        ) from exc
