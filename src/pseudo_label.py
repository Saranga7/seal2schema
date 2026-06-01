from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .data import unpaired_seal_dataframe, validate_paired_metadata
from .retrieval_bridge import RetrievalImageDataset, encode_images, load_retrieval_model
from .utils import get_device


@torch.no_grad()
def pseudo_label(cfg: DictConfig) -> None:
    device = get_device(cfg.device)
    model, schema_t, seal_t = load_retrieval_model(cfg, device)
    paired = validate_paired_metadata(cfg)
    unpaired = unpaired_seal_dataframe(cfg)

    gallery_ds = RetrievalImageDataset(paired, "schema_path", "monogram_id", "schema", schema_t)
    query_ds = RetrievalImageDataset(unpaired, "seal_path", "unpaired_id", "seal", seal_t)
    gallery_loader = DataLoader(
        gallery_ds,
        batch_size=int(cfg.pseudo_label.batch_size),
        shuffle=False,
        num_workers=int(cfg.pseudo_label.num_workers),
    )
    query_loader = DataLoader(
        query_ds,
        batch_size=int(cfg.pseudo_label.batch_size),
        shuffle=False,
        num_workers=int(cfg.pseudo_label.num_workers),
    )
    z_schema, schema_ids, schema_paths = encode_images(model, gallery_loader, device, kind="schema")
    z_seal, seal_ids, seal_paths = encode_images(model, query_loader, device, kind="seal")
    sim = z_seal @ z_schema.T
    top_scores, top_idx = torch.topk(sim, k=min(int(cfg.pseudo_label.top_k), sim.size(1)), dim=1)

    rows = []
    for i, seal_id in enumerate(seal_ids):
        best = int(top_idx[i, 0])
        rows.append(
            {
                "unpaired_id": seal_id,
                "seal_path": seal_paths[i],
                "mask_path": unpaired.iloc[i].get("mask_path", ""),
                "ann_path": unpaired.iloc[i].get("ann_path", ""),
                "pseudo_monogram_id": schema_ids[best],
                "pseudo_schema_path": schema_paths[best],
                "score": float(top_scores[i, 0]),
                "topk_schema_ids": "|".join(schema_ids[int(j)] for j in top_idx[i]),
                "topk_scores": "|".join(f"{float(s):.6f}" for s in top_scores[i]),
                "collection": unpaired.iloc[i].get("collection", Path(seal_paths[i]).parent.name),
            }
        )

    out_path = Path(cfg.pseudo_label.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} pseudo-pairs to {out_path}")
