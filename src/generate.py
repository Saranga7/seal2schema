from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SealOnlyDataset, load_split_frame, unpaired_seal_dataframe
from .models.dino_decoder import make_dino_decoder
from .models.diffusion import make_diffusion
from .models.dpt_decoder import make_dpt_decoder
from .models.pix2pix import Pix2PixGenerator
from .models.resnet_unet import make_resnet_unet
from .models.sam_decoder import make_sam_decoder
from .models.sd import make_stable_controlnet_sampler
from .postprocess import postprocess_batch
from .utils import get_device, tensor_to_pil


def load_generator_from_checkpoint(cfg: DictConfig, checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", cfg.model.name)
    saved_cfg = checkpoint.get("cfg")
    model_cfg = cfg
    if saved_cfg:
        model_cfg = OmegaConf.create(saved_cfg)
    if model_name == "pix2pix":
        model = Pix2PixGenerator(model_cfg.model.in_channels, model_cfg.model.out_channels, model_cfg.model.base_channels)
        model.load_state_dict(checkpoint["generator"])
        sampler = None
    elif model_name == "resnet_unet":
        model = make_resnet_unet(model_cfg)
        model.load_state_dict(checkpoint["generator"])
        sampler = None
    elif model_name == "diffusion":
        model, sampler = make_diffusion(model_cfg, device)
        model.load_state_dict(checkpoint["model"])
    elif model_name == "dino_decoder":
        model = make_dino_decoder(model_cfg)
        model.load_state_dict(checkpoint["generator"], strict=not bool(checkpoint.get("partial_generator", False)))
        sampler = None
    elif model_name == "sam_decoder":
        model = make_sam_decoder(model_cfg)
        model.load_state_dict(checkpoint["generator"], strict=not bool(checkpoint.get("partial_generator", False)))
        sampler = None
    elif model_name == "dpt_decoder":
        model = make_dpt_decoder(model_cfg)
        model.load_state_dict(checkpoint["generator"], strict=not bool(checkpoint.get("partial_generator", False)))
        sampler = None
    elif model_name == "stable_diffusion_controlnet":
        model = None
        sampler = make_stable_controlnet_sampler(model_cfg, checkpoint_path, device)
    else:
        raise ValueError(f"Unknown checkpoint model_name: {model_name}")
    if model is not None:
        model.to(device).eval()
    return model_name, model, sampler, model_cfg


def _postprocess_enabled(cfg: DictConfig) -> bool:
    return bool(cfg.evaluate.get("postprocess", {}).get("enabled", False))


@torch.no_grad()
def generate(cfg: DictConfig) -> None:
    checkpoint_path = cfg.train.checkpoint_path or cfg.evaluate.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("Set train.checkpoint_path or evaluate.checkpoint_path for generation.")
    device = get_device(cfg.device)
    model_name, model, sampler, checkpoint_cfg = load_generator_from_checkpoint(cfg, checkpoint_path, device)

    if cfg.evaluate.generated_dir is None:
        out_dir = Path(cfg.output_dir) / "generated"
    else:
        out_dir = Path(cfg.evaluate.generated_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.generate.dataset == "test":
        frame = load_split_frame(cfg, "test")
        id_col = "monogram_id"
    elif cfg.generate.dataset == "unpaired":
        frame = unpaired_seal_dataframe(cfg)
        id_col = "unpaired_id"
    else:
        raise ValueError(f"Unknown generate.dataset: {cfg.generate.dataset}")
    if cfg.data.limit_test is not None:
        frame = frame.head(int(cfg.data.limit_test))
    data_cfg = checkpoint_cfg.data if "data" in checkpoint_cfg else cfg.data
    dataset = SealOnlyDataset(
        frame,
        data_cfg.image_size,
        float(data_cfg.crop_padding),
        str(data_cfg.get("input_mode", cfg.data.input_mode)),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.data.batch_size), shuffle=False, num_workers=int(cfg.data.num_workers))
    rows = []
    for batch in tqdm(loader, desc="generate"):
        seal = batch["input"].to(device)
        if model_name == "diffusion":
            pred = sampler.sample(model, seal)
        elif model_name == "stable_diffusion_controlnet":
            pred = sampler.sample(batch, device)
        else:
            pred = model(seal).clamp(-1, 1)
        raw_pred = pred
        if _postprocess_enabled(cfg):
            pred = postprocess_batch(pred, cfg.evaluate.postprocess)
        for i, item_id in enumerate(batch["item_id"]):
            path = out_dir / f"{item_id}.png"
            raw_path = None
            if _postprocess_enabled(cfg) and bool(cfg.evaluate.postprocess.get("save_raw", True)):
                raw_dir = out_dir / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / f"{item_id}.png"
                tensor_to_pil(raw_pred[i]).save(raw_path)
            tensor_to_pil(pred[i]).save(path)
            row = {
                "item_id": item_id,
                id_col: item_id,
                "seal_path": batch["seal_path"][i],
                "mask_path": batch.get("mask_path", [""] * len(batch["item_id"]))[i],
                "collection": batch.get("collection", [""] * len(batch["item_id"]))[i],
                "generated_schema_path": str(path),
            }
            if raw_path is not None:
                row["raw_generated_schema_path"] = str(raw_path)
            if cfg.generate.dataset == "test":
                row["monogram_id"] = item_id
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"Saved {len(rows)} generated schemas to {out_dir}")
