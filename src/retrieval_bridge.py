from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class RetrievalImageDataset(Dataset):
    def __init__(self, frame, image_col: str, id_col: str, kind: str, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_col = image_col
        self.id_col = id_col
        self.kind = kind
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx].to_dict()
        img = Image.open(row[self.image_col]).convert("RGB")
        return {
            "image": self.transform(img),
            "item_id": str(row[self.id_col]),
            "path": row[self.image_col],
        }


def load_retrieval_model(cfg: DictConfig, device: torch.device):
    use_pseudo = getattr(cfg, "mode", None) == "pseudo_label"
    section = cfg.pseudo_label if use_pseudo else cfg.evaluate
    repo = Path(section.retrieval_repo)
    config_name = section.retrieval_config_name
    checkpoint_path = section.retrieval_checkpoint
    if checkpoint_path is None:
        raise ValueError("Set CMR_CHECKPOINT or pseudo_label/evaluate.retrieval_checkpoint.")

    # Both projects use a top-level package named "src". Temporarily clear the
    # current project's src modules so the retrieval repo can import its own.
    saved_src_modules = {name: mod for name, mod in sys.modules.items() if name == "src" or name.startswith("src.")}
    for name in list(saved_src_modules):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(repo))
    try:
        from src.dataset import PadToSquare
        from src.models import DualEncoder
    finally:
        try:
            sys.path.remove(str(repo))
        except ValueError:
            pass

    with initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        retrieval_cfg = compose(config_name=config_name)

    model = DualEncoder(retrieval_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    normalize = T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    color = T.Grayscale(3) if retrieval_cfg.data.transforms.use_grayscale else (lambda x: x)
    schema_t = T.Compose([color, PadToSquare(fill=0), T.Resize((224, 224)), T.ToTensor(), normalize])
    seal_t = T.Compose([color, T.Resize((224, 224)), T.ToTensor(), normalize])
    for name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
        sys.modules.pop(name, None)
    for name, mod in saved_src_modules.items():
        sys.modules[name] = mod
    return model, schema_t, seal_t


@torch.no_grad()
def encode_images(model, loader: DataLoader, device: torch.device, kind: str):
    embeddings = []
    ids = []
    paths = []
    for batch in loader:
        x = batch["image"].to(device)
        z = model.encode_schema(x) if kind == "schema" else model.encode_seal(x)
        embeddings.append(z.cpu())
        ids.extend(batch["item_id"])
        paths.extend(batch["path"])
    return torch.cat(embeddings, dim=0), ids, paths
