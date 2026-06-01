from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def list_images(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    files: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_image_by_stem(directory: str | Path, stem: str, valid_exts: Iterable[str] = IMAGE_EXTS) -> Path:
    directory = Path(directory)
    for ext in valid_exts:
        for candidate in (directory / f"{stem}{ext}", directory / f"{stem}{ext.upper()}"):
            if candidate.exists():
                return candidate
    matches = [p for p in directory.glob(f"{stem}.*") if p.suffix.lower() in {e.lower() for e in valid_exts}]
    if matches:
        return sorted(matches)[0]
    raise FileNotFoundError(f"No image found for stem '{stem}' in {directory}")


def tensor_to_pil(tensor: torch.Tensor, mode: str = "L") -> Image.Image:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(-1, 1)
    arr = ((tensor + 1.0) * 127.5).clamp(0, 255).byte().numpy()
    return Image.fromarray(arr, mode=mode)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
