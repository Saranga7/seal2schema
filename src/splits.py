from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .data import validate_paired_metadata


def create_splits(cfg: DictConfig) -> None:
    df = validate_paired_metadata(cfg)
    out_dir = Path(cfg.data.splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stratify = df["quality_label"] if "quality_label" in df.columns else None
    train_df, temp_df = train_test_split(
        df,
        train_size=float(cfg.data.split.train),
        random_state=int(cfg.seed),
        shuffle=True,
        stratify=stratify,
    )
    rel_val = float(cfg.data.split.val) / (float(cfg.data.split.val) + float(cfg.data.split.test))
    temp_stratify = temp_df["quality_label"] if "quality_label" in temp_df.columns else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=rel_val,
        random_state=int(cfg.seed),
        shuffle=True,
        stratify=temp_stratify,
    )

    for name, split_df in {"train": train_df, "val": val_df, "test": test_df}.items():
        split_df.sort_values("monogram_id").to_csv(out_dir / f"{name}.csv", index=False)

    stats_rows = []
    for name, split_df in {"train": train_df, "val": val_df, "test": test_df}.items():
        row = {"split": name, "count": len(split_df)}
        if "quality_label" in split_df.columns:
            for quality, count in split_df["quality_label"].value_counts().sort_index().items():
                row[f"q{quality}_count"] = int(count)
        stats_rows.append(row)
    stats = pd.DataFrame(stats_rows).fillna(0)
    stats.to_csv(out_dir / "stats.csv", index=False)
    print(stats.to_string(index=False))
