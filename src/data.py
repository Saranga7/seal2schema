from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .utils import IMAGE_EXTS, find_image_by_stem, list_images


def _quality_key(value: Any) -> str:
    text = str(value).strip().lower()
    return text if text.startswith("q") else f"q{text}"


def _excluded_quality_keys(cfg: DictConfig) -> set[str]:
    return {_quality_key(value) for value in getattr(cfg.data, "exclude_quality_labels", [])}


def _annotation_path(ann_dir: Path, image_path: Path) -> Path:
    candidates = [
        ann_dir / f"{image_path.name}.json",
        ann_dir / f"{image_path.stem}.json",
        ann_dir / f"{image_path.stem}{image_path.suffix}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(ann_dir.glob(f"{image_path.stem}*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No annotation JSON found for '{image_path.name}' in {ann_dir}")


def _bbox_from_mask(mask: Image.Image) -> tuple[int, int, int, int] | None:
    return mask.convert("L").point(lambda x: 255 if x > 0 else 0).getbbox()


def _bbox_from_annotation(path: str | Path) -> tuple[int, int, int, int] | None:
    with Path(path).open() as f:
        payload = json.load(f)
    points: list[list[float]] = []
    for obj in payload.get("objects", []):
        if obj.get("classTitle") != "Monogram":
            continue
        points.extend(obj.get("points", {}).get("exterior", []))
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_frac: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    pad = int(max(right - left, bottom - top) * padding_frac)
    return max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad)


def _jitter_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    translate_frac: float,
    scale_frac: float,
) -> tuple[int, int, int, int]:
    if translate_frac <= 0 and scale_frac <= 0:
        return bbox
    left, top, right, bottom = bbox
    box_w = max(right - left, 1)
    box_h = max(bottom - top, 1)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    if translate_frac > 0:
        center_x += float(torch.empty(1).uniform_(-translate_frac, translate_frac).item()) * box_w
        center_y += float(torch.empty(1).uniform_(-translate_frac, translate_frac).item()) * box_h
    if scale_frac > 0:
        scale = float(torch.empty(1).uniform_(1.0 - scale_frac, 1.0 + scale_frac).item())
        box_w *= scale
        box_h *= scale
    left = int(round(center_x - box_w / 2.0))
    right = int(round(center_x + box_w / 2.0))
    top = int(round(center_y - box_h / 2.0))
    bottom = int(round(center_y + box_h / 2.0))
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height
    left = max(0, left)
    top = max(0, top)
    right = min(image_width, max(left + 1, right))
    bottom = min(image_height, max(top + 1, bottom))
    return left, top, right, bottom


def _crop_pair(
    image: Image.Image,
    mask: Image.Image,
    ann_path: str | Path | None,
    image_size: int,
    crop_padding: float,
    augment: DictConfig | None = None,
) -> tuple[Image.Image, Image.Image]:
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    bbox = _bbox_from_mask(mask)
    if bbox is None and ann_path is not None:
        bbox = _bbox_from_annotation(ann_path)
    if bbox is None:
        bbox = (0, 0, image.width, image.height)
    bbox = _pad_bbox(bbox, image.width, image.height, crop_padding)
    if augment is not None and bool(augment.get("enabled", False)):
        bbox = _jitter_bbox(
            bbox,
            image.width,
            image.height,
            float(augment.get("crop_jitter", 0.0)),
            float(augment.get("crop_scale_jitter", 0.0)),
        )
    image_crop = image.crop(bbox).resize((image_size, image_size), Image.Resampling.BICUBIC)
    mask_crop = mask.crop(bbox).resize((image_size, image_size), Image.Resampling.NEAREST)
    return image_crop, mask_crop


def _compose_model_input(seal: torch.Tensor, mask: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode == "seal_mask":
        return torch.cat([seal, mask], dim=0)
    if input_mode == "mask_only":
        return mask
    if input_mode == "seal_only":
        return seal
    raise ValueError(f"Unknown data.input_mode: {input_mode}")


def _augment_enabled(cfg: DictConfig) -> bool:
    return bool(cfg.data.get("augment", {}).get("enabled", False))


def _sample_affine_params(aug: DictConfig) -> dict[str, Any]:
    max_rotation = float(aug.get("rotation", 0.0))
    max_translate = float(aug.get("translate", 0.0))
    scale_min = float(aug.get("scale_min", 1.0))
    scale_max = float(aug.get("scale_max", 1.0))
    max_shear = float(aug.get("shear", 0.0))
    return {
        "angle": float(torch.empty(1).uniform_(-max_rotation, max_rotation).item()),
        "translate_frac": (
            float(torch.empty(1).uniform_(-max_translate, max_translate).item()),
            float(torch.empty(1).uniform_(-max_translate, max_translate).item()),
        ),
        "scale": float(torch.empty(1).uniform_(scale_min, scale_max).item()),
        "shear": [float(torch.empty(1).uniform_(-max_shear, max_shear).item()), 0.0],
    }


def _apply_input_affine(
    seal: Image.Image,
    mask: Image.Image,
    aug: DictConfig,
) -> tuple[Image.Image, Image.Image]:
    params = _sample_affine_params(aug)
    width, height = seal.size
    translate = (
        int(round(params["translate_frac"][0] * width)),
        int(round(params["translate_frac"][1] * height)),
    )
    seal = TF.affine(
        seal,
        angle=params["angle"],
        translate=translate,
        scale=params["scale"],
        shear=params["shear"],
        interpolation=T.InterpolationMode.BICUBIC,
        fill=0,
    )
    mask = TF.affine(
        mask,
        angle=params["angle"],
        translate=translate,
        scale=params["scale"],
        shear=params["shear"],
        interpolation=T.InterpolationMode.NEAREST,
        fill=0,
    )
    return seal, mask


def _apply_seal_photometric(seal: Image.Image, aug: DictConfig) -> Image.Image:
    brightness = float(aug.get("seal_brightness", 0.0))
    contrast = float(aug.get("seal_contrast", 0.0))
    if brightness > 0:
        factor = float(torch.empty(1).uniform_(max(0.0, 1.0 - brightness), 1.0 + brightness).item())
        seal = TF.adjust_brightness(seal, factor)
    if contrast > 0:
        factor = float(torch.empty(1).uniform_(max(0.0, 1.0 - contrast), 1.0 + contrast).item())
        seal = TF.adjust_contrast(seal, factor)
    return seal


def _add_tensor_noise(seal_t: torch.Tensor, aug: DictConfig) -> torch.Tensor:
    noise_std = float(aug.get("seal_noise_std", 0.0))
    if noise_std <= 0:
        return seal_t
    return (seal_t + torch.randn_like(seal_t) * noise_std).clamp(-1, 1)


def validate_paired_metadata(cfg: DictConfig) -> pd.DataFrame:
    paired_dir = Path(cfg.data.paired_dir)
    metadata = pd.read_csv(cfg.data.metadata_csv)
    if "monogram_id" not in metadata.columns:
        raise ValueError(f"{cfg.data.metadata_csv} must contain a monogram_id column")
    if "quality_label" not in metadata.columns:
        raise ValueError(f"{cfg.data.metadata_csv} must contain a quality_label column")
    excluded = _excluded_quality_keys(cfg)
    metadata = metadata[~metadata["quality_label"].map(_quality_key).isin(excluded)].copy()

    seal_dir = paired_dir / cfg.data.seal_dir_name
    schema_dir = paired_dir / cfg.data.schema_dir_name
    mask_dir = paired_dir / cfg.data.mask_dir_name
    ann_dir = paired_dir / cfg.data.ann_dir_name
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for row in metadata.to_dict("records"):
        monogram_id = str(row["monogram_id"]).strip()
        try:
            seal_path = find_image_by_stem(seal_dir, monogram_id, cfg.data.valid_exts)
            schema_path = find_image_by_stem(schema_dir, monogram_id, cfg.data.valid_exts)
            mask_path = find_image_by_stem(mask_dir, monogram_id, cfg.data.valid_exts)
            ann_path = _annotation_path(ann_dir, seal_path)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        rows.append(
            {
                **row,
                "monogram_id": monogram_id,
                "seal_path": str(seal_path),
                "schema_path": str(schema_path),
                "mask_path": str(mask_path),
                "ann_path": str(ann_path),
            }
        )

    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Metadata has {len(missing)} missing image pairs. First failures:\n{preview}")
    expected = int(cfg.data.expected_paired_count)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} usable metadata-backed pairs, found {len(rows)}")
    return pd.DataFrame(rows)


def unpaired_seal_dataframe(cfg: DictConfig) -> pd.DataFrame:
    root = Path(cfg.data.unpaired_dir)
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for image_dir in sorted(root.glob("*/img")):
        collection_dir = image_dir.parent
        mask_dir = collection_dir / cfg.data.mask_dir_name
        ann_dir = collection_dir / cfg.data.ann_dir_name
        collection = collection_dir.name
        for image_path in list_images(image_dir, recursive=False):
            try:
                mask_path = find_image_by_stem(mask_dir, image_path.stem, cfg.data.valid_exts)
                ann_path = _annotation_path(ann_dir, image_path)
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            item_id = f"{collection}__{image_path.stem}".replace("/", "__")
            rows.append(
                {
                    "unpaired_id": item_id,
                    "item_id": item_id,
                    "seal_path": str(image_path),
                    "mask_path": str(mask_path),
                    "ann_path": str(ann_path),
                    "collection": collection,
                }
            )
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Unpaired set has {len(missing)} missing masks/annotations. First failures:\n{preview}")
    expected = int(cfg.data.expected_unpaired_count)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} unpaired seal images, found {len(rows)}")
    return pd.DataFrame(rows)


def seal_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def mask_transform() -> T.Compose:
    return T.Compose(
        [
            T.Grayscale(1),
            T.ToTensor(),
            T.Lambda(lambda x: (x > 0.5).float()),
            T.Normalize([0.5], [0.5]),
        ]
    )


def schema_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Grayscale(1),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            T.Lambda(lambda x: (x > 0.5).float()),
            T.Normalize([0.5], [0.5]),
        ]
    )


class SealSchemaDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
        pseudo_loss_weight: float = 1.0,
        crop_padding: float = 0.08,
        input_mode: str = "seal_mask",
        augment: DictConfig | None = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.seal_t = seal_transform(image_size)
        self.mask_t = mask_transform()
        self.schema_t = schema_transform(image_size)
        self.image_size = image_size
        self.pseudo_loss_weight = pseudo_loss_weight
        self.crop_padding = crop_padding
        self.input_mode = input_mode
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx].to_dict()
        seal = Image.open(row["seal_path"]).convert("RGB")
        mask_path = row.get("mask_path")
        mask = Image.open(mask_path).convert("L") if isinstance(mask_path, str) and mask_path else Image.new("L", seal.size, 255)
        seal, mask = _crop_pair(seal, mask, row.get("ann_path"), self.image_size, self.crop_padding, self.augment)
        schema = Image.open(row["schema_path"]).convert("L")
        if self.augment is not None and bool(self.augment.get("enabled", False)) and bool(self.augment.get("input_affine_enabled", False)):
            seal, mask = _apply_input_affine(seal, mask, self.augment)
        if self.augment is not None and bool(self.augment.get("enabled", False)):
            seal = _apply_seal_photometric(seal, self.augment)
        source = row.get("pair_source", "real")
        weight = 1.0 if source == "real" else float(row.get("loss_weight", self.pseudo_loss_weight))
        seal_t = self.seal_t(seal)
        if self.augment is not None and bool(self.augment.get("enabled", False)):
            seal_t = _add_tensor_noise(seal_t, self.augment)
        mask_t = self.mask_t(mask)
        model_input = _compose_model_input(seal_t, mask_t, self.input_mode)
        return {
            "input": model_input,
            "seal": model_input,
            "seal_rgb": seal_t,
            "mask": mask_t,
            "schema": self.schema_t(schema),
            "monogram_id": str(row.get("monogram_id", row.get("unpaired_id", idx))),
            "seal_path": row["seal_path"],
            "schema_path": row["schema_path"],
            "mask_path": row.get("mask_path", ""),
            "quality_label": row.get("quality_label", ""),
            "pair_source": source,
            "loss_weight": torch.tensor(weight, dtype=torch.float32),
        }


class SealOnlyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, crop_padding: float = 0.08, input_mode: str = "seal_mask"):
        self.frame = frame.reset_index(drop=True)
        self.seal_t = seal_transform(image_size)
        self.mask_t = mask_transform()
        self.image_size = image_size
        self.crop_padding = crop_padding
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx].to_dict()
        seal = Image.open(row["seal_path"]).convert("RGB")
        mask_path = row.get("mask_path")
        mask = Image.open(mask_path).convert("L") if isinstance(mask_path, str) and mask_path else Image.new("L", seal.size, 255)
        seal, mask = _crop_pair(seal, mask, row.get("ann_path"), self.image_size, self.crop_padding)
        seal_t = self.seal_t(seal)
        mask_t = self.mask_t(mask)
        model_input = _compose_model_input(seal_t, mask_t, self.input_mode)
        return {
            "input": model_input,
            "seal": model_input,
            "seal_rgb": seal_t,
            "mask": mask_t,
            "item_id": str(row.get("unpaired_id", row.get("monogram_id", idx))),
            "seal_path": row["seal_path"],
            "mask_path": row.get("mask_path", ""),
            "collection": row.get("collection", ""),
        }


def load_split_frame(cfg: DictConfig, split: str) -> pd.DataFrame:
    split_path = Path(cfg.data.splits_dir) / f"{split}.csv"
    if not split_path.exists():
        from .splits import create_splits

        create_splits(cfg)
    return pd.read_csv(split_path)


def apply_limit(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def build_train_val_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    train_df = load_split_frame(cfg, "train")
    val_df = load_split_frame(cfg, "val")
    train_df["pair_source"] = "real"
    val_df["pair_source"] = "real"

    if cfg.data.training_set in {"real332+pseudo_all", "real332+pseudo_confident", "real350+pseudo_all", "real350+pseudo_confident"}:
        pseudo_path = Path(cfg.data.pseudo_pairs_csv)
        if not pseudo_path.exists():
            raise FileNotFoundError(f"Pseudo-pair CSV not found: {pseudo_path}")
        pseudo_df = pd.read_csv(pseudo_path)
        if cfg.data.training_set in {"real332+pseudo_confident", "real350+pseudo_confident"}:
            pseudo_df = pseudo_df[pseudo_df["score"] >= float(cfg.data.pseudo_confidence_threshold)]
        pseudo_df = pseudo_df.rename(columns={"pseudo_schema_path": "schema_path"})
        pseudo_df["pair_source"] = "pseudo"
        pseudo_df["loss_weight"] = float(cfg.data.pseudo_loss_weight)
        train_df = pd.concat([train_df, pseudo_df], ignore_index=True, sort=False)
    elif cfg.data.training_set not in {"real332", "real350"}:
        raise ValueError(f"Unknown training_set: {cfg.data.training_set}")

    train_ds = SealSchemaDataset(
        train_df,
        cfg.data.image_size,
        cfg.data.pseudo_loss_weight,
        float(cfg.data.crop_padding),
        str(cfg.data.input_mode),
        cfg.data.augment if _augment_enabled(cfg) else None,
    )
    val_ds = SealSchemaDataset(
        val_df,
        cfg.data.image_size,
        cfg.data.pseudo_loss_weight,
        float(cfg.data.crop_padding),
        str(cfg.data.input_mode),
        None,
    )
    return apply_limit(train_ds, cfg.data.limit_train), apply_limit(val_ds, cfg.data.limit_val)


def build_test_dataset(cfg: DictConfig) -> Dataset:
    test_df = load_split_frame(cfg, "test")
    test_df["pair_source"] = "real"
    return apply_limit(
        SealSchemaDataset(
            test_df,
            cfg.data.image_size,
            crop_padding=float(cfg.data.crop_padding),
            input_mode=str(cfg.data.input_mode),
        ),
        cfg.data.limit_test,
    )
