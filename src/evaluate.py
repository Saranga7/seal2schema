from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .data import SealSchemaDataset, load_split_frame, schema_transform
from .generate import load_generator_from_checkpoint
from .metrics import reconstruction_metrics_per_item
from .postprocess import postprocess_batch
from .retrieval_bridge import RetrievalImageDataset, encode_images, load_retrieval_model
from .utils import get_device


def _batch_value(batch: dict, key: str, idx: int, default: str = "") -> str:
    values = batch.get(key)
    if values is None:
        return default
    if torch.is_tensor(values):
        return str(values[idx].item())
    return str(values[idx])


def _postprocess_enabled(cfg: DictConfig) -> bool:
    return bool(cfg.evaluate.get("postprocess", {}).get("enabled", False))


def _apply_postprocess_if_enabled(pred: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    if not _postprocess_enabled(cfg):
        return pred
    return postprocess_batch(pred, cfg.evaluate.postprocess)


def _load_generated_prediction(path: str | Path, image_size: int, binarize: bool) -> torch.Tensor:
    if binarize:
        return schema_transform(image_size)(Image.open(path).convert("L")).unsqueeze(0)
    image = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.array(image).astype("float32") / 255.0
    return (torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2.0 - 1.0).clamp(-1, 1)


def _quality_key(value: str) -> str:
    text = str(value).strip().lower()
    if text == "":
        return "unknown"
    return text if text.startswith("q") else f"q{text}"


def _add_quality_example(
    examples: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    quality: str,
    target: torch.Tensor,
    pred: torch.Tensor,
    max_per_quality: int,
) -> None:
    key = _quality_key(quality)
    bucket = examples.setdefault(key, [])
    if len(bucket) >= max_per_quality:
        return
    bucket.append((target.detach().cpu(), pred.detach().cpu()))


def _save_quality_grids(
    examples: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    output_dir: Path,
) -> None:
    if not examples:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for quality, pairs in sorted(examples.items()):
        if not pairs:
            continue
        targets = torch.stack([pair[0].squeeze(0) if pair[0].ndim == 4 else pair[0] for pair in pairs], dim=0)
        preds = torch.stack([pair[1].squeeze(0) if pair[1].ndim == 4 else pair[1] for pair in pairs], dim=0)
        grid = torch.cat([(targets + 1) / 2, (preds + 1) / 2], dim=0)
        save_image(grid, output_dir / f"{quality}_target_generated_grid.png", nrow=len(pairs))


@torch.no_grad()
def evaluate(cfg: DictConfig) -> None:
    device = get_device(cfg.device)
    generated_dir = Path(cfg.evaluate.generated_dir) if cfg.evaluate.generated_dir else None
    quality_examples: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    quality_viz_cfg = cfg.evaluate.get("quality_visualizations", {})
    quality_viz_enabled = bool(quality_viz_cfg.get("enabled", True))
    max_quality_examples = int(quality_viz_cfg.get("max_per_quality", 8))

    if generated_dir is not None and (generated_dir / "manifest.csv").exists():
        manifest = pd.read_csv(generated_dir / "manifest.csv")
        eval_df = load_split_frame(cfg, str(cfg.evaluate.split))
        if cfg.data.limit_test is not None:
            eval_df = eval_df.head(int(cfg.data.limit_test))
        df = eval_df.merge(manifest[["monogram_id", "generated_schema_path"]], on="monogram_id", how="inner")
        rows = []
        transform = schema_transform(cfg.data.image_size)
        for row in tqdm(df.to_dict("records"), desc="evaluate files"):
            pred = _load_generated_prediction(
                row["generated_schema_path"],
                int(cfg.data.image_size),
                binarize=not _postprocess_enabled(cfg),
            )
            pred = _apply_postprocess_if_enabled(pred, cfg)
            target = transform(Image.open(row["schema_path"]).convert("L")).unsqueeze(0)
            quality = row.get("quality_label", "")
            rows.append(
                {
                    "monogram_id": row["monogram_id"],
                    "quality_label": quality,
                    **reconstruction_metrics_per_item(pred, target, cfg.evaluate.threshold)[0],
                }
            )
            if quality_viz_enabled:
                _add_quality_example(quality_examples, str(quality), target[0], pred[0], max_quality_examples)
    else:
        checkpoint_path = cfg.evaluate.checkpoint_path or cfg.train.checkpoint_path
        if checkpoint_path is None:
            raise ValueError("Set evaluate.checkpoint_path or evaluate.generated_dir.")
        _, model, sampler, checkpoint_cfg = load_generator_from_checkpoint(cfg, checkpoint_path, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_name = checkpoint.get("model_name", cfg.model.name)
        test_df = load_split_frame(cfg, str(cfg.evaluate.split))
        if cfg.data.limit_test is not None:
            test_df = test_df.head(int(cfg.data.limit_test))
        data_cfg = checkpoint_cfg.data if "data" in checkpoint_cfg else cfg.data
        dataset = SealSchemaDataset(
            test_df,
            data_cfg.image_size,
            crop_padding=float(data_cfg.crop_padding),
            input_mode=str(data_cfg.get("input_mode", cfg.data.input_mode)),
        )
        loader = DataLoader(dataset, batch_size=int(cfg.data.batch_size), shuffle=False, num_workers=int(cfg.data.num_workers))
        rows = []
        qualitative = []
        for batch_idx, batch in enumerate(tqdm(loader, desc="evaluate checkpoint")):
            seal = batch["input"].to(device)
            target = batch["schema"].to(device)
            if model_name == "diffusion":
                pred = sampler.sample(model, seal)
            elif model_name == "stable_diffusion_controlnet":
                pred = sampler.sample(batch, device)
            else:
                pred = model(seal).clamp(-1, 1)
            pred = _apply_postprocess_if_enabled(pred, cfg)
            batch_metrics = reconstruction_metrics_per_item(pred, target, cfg.evaluate.threshold)
            for idx, item_id in enumerate(batch["monogram_id"]):
                quality = _batch_value(batch, "quality_label", idx)
                rows.append(
                    {
                        "monogram_id": item_id,
                        "quality_label": quality,
                        **batch_metrics[idx],
                    }
                )
                if quality_viz_enabled:
                    _add_quality_example(quality_examples, quality, target[idx], pred[idx], max_quality_examples)
            if batch_idx < 3:
                qualitative.append(torch.cat([(target[:4].cpu() + 1) / 2, (pred[:4].cpu() + 1) / 2], dim=0))
        if qualitative:
            sample_dir = Path(cfg.output_dir) / "qualitative"
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_image(torch.cat(qualitative, dim=0), sample_dir / "test_target_generated_grid.png", nrow=4)

    if quality_viz_enabled:
        _save_quality_grids(quality_examples, Path(cfg.output_dir) / "qualitative" / "by_quality")

    result = pd.DataFrame(rows)
    metric_cols = [
        "mae",
        "ssim",
        "binary_iou",
        "foreground_recall",
        "skeleton_precision",
        "skeleton_recall",
        "skeleton_f1",
        "skeleton_iou",
        "skeleton_chamfer",
    ]
    summary = result[metric_cols].mean().to_dict()
    out_path = Path(cfg.output_dir) / cfg.evaluate.metrics_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    pd.DataFrame([summary]).to_csv(Path(cfg.output_dir) / "metrics_summary.csv", index=False)
    if "quality_label" in result.columns and result["quality_label"].astype(str).str.len().sum() > 0:
        by_quality = result.groupby("quality_label", dropna=False)[metric_cols].mean().reset_index()
        by_quality.to_csv(Path(cfg.output_dir) / "metrics_by_quality.csv", index=False)
    print(summary)
    print(f"Saved metrics to {out_path}")

    if generated_dir is not None and (generated_dir / "manifest.csv").exists() and cfg.evaluate.retrieval_checkpoint is not None:
        manifest = pd.read_csv(generated_dir / "manifest.csv")
        test_ids = set(load_split_frame(cfg, "test")["monogram_id"].astype(str))
        manifest_ids = set(manifest.get("monogram_id", pd.Series(dtype=str)).astype(str))
        if test_ids & manifest_ids:
            retrieval = retrieval_self_consistency(cfg, generated_dir, device)
            pd.DataFrame([retrieval]).to_csv(Path(cfg.output_dir) / "retrieval_self_consistency.csv", index=False)
            print(retrieval)
        expansion = retrieval_expanded_gallery(cfg, generated_dir, device)
        pd.DataFrame([expansion]).to_csv(Path(cfg.output_dir) / "retrieval_expanded_gallery.csv", index=False)
        print(expansion)


@torch.no_grad()
def retrieval_self_consistency(cfg: DictConfig, generated_dir: Path, device: torch.device) -> dict[str, float]:
    manifest = pd.read_csv(generated_dir / "manifest.csv")
    test_df = load_split_frame(cfg, "test")
    df = test_df.merge(manifest[["monogram_id", "generated_schema_path"]], on="monogram_id", how="inner")
    model, schema_t, seal_t = load_retrieval_model(cfg, device)
    gen_ds = RetrievalImageDataset(df, "generated_schema_path", "monogram_id", "schema", schema_t)
    seal_ds = RetrievalImageDataset(df, "seal_path", "monogram_id", "seal", seal_t)
    gen_loader = DataLoader(gen_ds, batch_size=int(cfg.data.batch_size), shuffle=False, num_workers=int(cfg.data.num_workers))
    seal_loader = DataLoader(seal_ds, batch_size=int(cfg.data.batch_size), shuffle=False, num_workers=int(cfg.data.num_workers))
    z_gen, gen_ids, _ = encode_images(model, gen_loader, device, kind="schema")
    z_seal, seal_ids, _ = encode_images(model, seal_loader, device, kind="seal")
    sim = z_gen @ z_seal.T
    rankings = torch.argsort(sim, dim=1, descending=True)
    seal_index = {pid: i for i, pid in enumerate(seal_ids)}
    ranks = []
    for i, pid in enumerate(gen_ids):
        gt = seal_index[pid]
        rank = (rankings[i] == gt).nonzero(as_tuple=False)[0, 0].item() + 1
        ranks.append(rank)
    ranks_t = torch.tensor(ranks, dtype=torch.float32)
    return {
        "generated_schema_to_seal_R@1": float((ranks_t <= 1).float().mean()),
        "generated_schema_to_seal_R@5": float((ranks_t <= 5).float().mean()),
        "generated_schema_to_seal_R@10": float((ranks_t <= 10).float().mean()),
        "generated_schema_to_seal_MRR": float((1.0 / ranks_t).mean()),
        "generated_schema_to_seal_MedianRank": float(ranks_t.median()),
    }


@torch.no_grad()
def retrieval_expanded_gallery(cfg: DictConfig, generated_dir: Path, device: torch.device) -> dict[str, float]:
    manifest = pd.read_csv(generated_dir / "manifest.csv")
    if "generated_schema_path" not in manifest.columns:
        raise ValueError(f"{generated_dir / 'manifest.csv'} must contain generated_schema_path")
    paired = load_split_frame(cfg, "test").copy()
    paired_gallery = paired[["monogram_id", "schema_path"]].rename(columns={"monogram_id": "gallery_id"})
    paired_gallery["gallery_source"] = "paired_schema"
    generated_gallery = manifest.copy()
    generated_gallery["gallery_id"] = generated_gallery.get("item_id", generated_gallery.index.astype(str))
    generated_gallery = generated_gallery[["gallery_id", "generated_schema_path"]].rename(
        columns={"generated_schema_path": "schema_path"}
    )
    generated_gallery["gallery_source"] = "generated_unpaired_schema"
    gallery = pd.concat([paired_gallery, generated_gallery], ignore_index=True, sort=False)
    gallery_manifest = generated_dir / "expanded_gallery_manifest.csv"
    gallery.to_csv(gallery_manifest, index=False)

    model, schema_t, seal_t = load_retrieval_model(cfg, device)
    gallery_ds = RetrievalImageDataset(gallery, "schema_path", "gallery_id", "schema", schema_t)
    query_ds = RetrievalImageDataset(paired, "seal_path", "monogram_id", "seal", seal_t)
    gallery_loader = DataLoader(gallery_ds, batch_size=int(cfg.data.batch_size), shuffle=False, num_workers=int(cfg.data.num_workers))
    query_loader = DataLoader(query_ds, batch_size=int(cfg.data.batch_size), shuffle=False, num_workers=int(cfg.data.num_workers))
    z_gallery, gallery_ids, _ = encode_images(model, gallery_loader, device, kind="schema")
    z_query, query_ids, _ = encode_images(model, query_loader, device, kind="seal")
    sim = z_query @ z_gallery.T
    rankings = torch.argsort(sim, dim=1, descending=True)
    gallery_index = {pid: i for i, pid in enumerate(gallery_ids)}
    ranks = []
    for i, pid in enumerate(query_ids):
        gt = gallery_index[pid]
        rank = (rankings[i] == gt).nonzero(as_tuple=False)[0, 0].item() + 1
        ranks.append(rank)
    ranks_t = torch.tensor(ranks, dtype=torch.float32)
    return {
        "expanded_gallery_size": float(len(gallery)),
        "paired_test_queries": float(len(query_ids)),
        "seal_to_schema_expanded_R@1": float((ranks_t <= 1).float().mean()),
        "seal_to_schema_expanded_R@5": float((ranks_t <= 5).float().mean()),
        "seal_to_schema_expanded_R@10": float((ranks_t <= 10).float().mean()),
        "seal_to_schema_expanded_MRR": float((1.0 / ranks_t).mean()),
        "seal_to_schema_expanded_MedianRank": float(ranks_t.median()),
    }
