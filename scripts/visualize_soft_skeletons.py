from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import SealSchemaDataset  # noqa: E402
from src.generate import load_generator_from_checkpoint  # noqa: E402
from src.loss import soft_skeletonize  # noqa: E402
from src.utils import get_device  # noqa: E402


def denorm01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def tensor_to_pil(x: torch.Tensor, size: int = 180) -> Image.Image:
    x = x.detach().float().cpu()
    if x.ndim == 3 and x.shape[0] == 3:
        arr = (denorm01(x).permute(1, 2, 0).numpy() * 255).astype("uint8")
        img = Image.fromarray(arr, mode="RGB")
    else:
        if x.ndim == 3:
            x = x[:1]
        arr = (x.squeeze().clamp(0, 1).numpy() * 255).astype("uint8")
        img = Image.fromarray(arr, mode="L").convert("RGB")
    return img.resize((size, size), Image.Resampling.NEAREST)


def labeled_tile(label: str, image: Image.Image, width: int, label_h: int) -> Image.Image:
    tile = Image.new("RGB", (width, width + label_h), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((6, 8), label, fill=(25, 32, 44))
    tile.paste(image, (0, label_h))
    return tile


def make_row(label_image_pairs: list[tuple[str, Image.Image]], tile_size: int, label_h: int, gap: int = 10) -> Image.Image:
    tiles = [labeled_tile(label, image, tile_size, label_h) for label, image in label_image_pairs]
    row = Image.new("RGB", (len(tiles) * tile_size + (len(tiles) - 1) * gap, tile_size + label_h), (245, 243, 239))
    x = 0
    for tile in tiles:
        row.paste(tile, (x, 0))
        x += tile_size + gap
    return row


def stack_rows(rows: list[Image.Image], gap: int = 16) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), (235, 232, 226))
    y = 0
    for row in rows:
        canvas.paste(row, ((width - row.width) // 2, y))
        y += row.height + gap
    return canvas


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_frame(split: str, limit: int | None) -> pd.DataFrame:
    path = ROOT / "splits" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    frame = pd.read_csv(path)
    return frame.head(limit) if limit is not None else frame


def make_dataset(frame: pd.DataFrame, cfg, input_mode: str | None) -> SealSchemaDataset:
    data_cfg = cfg.data
    return SealSchemaDataset(
        frame,
        int(data_cfg.image_size),
        crop_padding=float(data_cfg.crop_padding),
        input_mode=input_mode or str(data_cfg.get("input_mode", "seal_mask")),
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the exact soft skeleton maps used by the training loss.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sweep", default="1,5,10,20", help="Comma-separated skeleton iteration counts for target sweep.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional checkpoint for prediction skeleton visualization.")
    parser.add_argument("--output-dir", default="outputs/soft_skeleton_debug")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(ROOT / "configs" / "data" / "default.yaml")
    cfg = OmegaConf.create({"data": base_cfg, "model": {"name": "pix2pix"}, "device": args.device})
    model = None
    sampler = None
    model_name = None
    checkpoint_cfg = None
    device = torch.device("cpu")
    if args.checkpoint_path:
        device = get_device(args.device)
        _, model, sampler, checkpoint_cfg = load_generator_from_checkpoint(cfg, args.checkpoint_path, device)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        model_name = checkpoint.get("model_name", "unknown")
        if "data" in checkpoint_cfg:
            cfg.data = checkpoint_cfg.data

    frame = load_frame(args.split, args.limit)
    dataset = make_dataset(frame, cfg, input_mode=str(cfg.data.get("input_mode", "seal_mask")))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = parse_int_list(args.sweep)
    all_rows = []
    saved = 0
    for batch in loader:
        inputs = batch["input"]
        target = batch["schema"]
        target_prob = denorm01(target)
        pred = None
        pred_prob = None
        if model_name == "diffusion":
            pred = sampler.sample(model, inputs.to(device)).cpu()
        elif model_name == "stable_diffusion_controlnet":
            pred = sampler.sample(batch, device).cpu()
        elif model is not None:
            pred = model(inputs.to(device)).clamp(-1, 1).cpu()
        if pred is not None:
            pred_prob = denorm01(pred)

        for idx in range(target.size(0)):
            item_id = str(batch["monogram_id"][idx])
            target_skel = soft_skeletonize(target_prob[idx : idx + 1], args.iterations)[0]
            pairs = [
                ("seal crop", tensor_to_pil(batch["seal_rgb"][idx])),
                ("mask", tensor_to_pil(denorm01(batch["mask"][idx]))),
                ("target", tensor_to_pil(target_prob[idx])),
                (f"target soft skel {args.iterations}", tensor_to_pil(target_skel)),
            ]
            for count in sweep:
                skel = soft_skeletonize(target_prob[idx : idx + 1], count)[0]
                pairs.append((f"target skel {count}", tensor_to_pil(skel)))
            if pred_prob is not None:
                pred_skel = soft_skeletonize(pred_prob[idx : idx + 1], args.iterations)[0]
                diff = torch.abs(pred_skel - target_skel)
                pairs.extend(
                    [
                        ("prediction", tensor_to_pil(pred_prob[idx])),
                        (f"pred soft skel {args.iterations}", tensor_to_pil(pred_skel)),
                        ("skel abs diff", tensor_to_pil(diff)),
                    ]
                )
            rows_needed = math.ceil(len(pairs) / 5)
            chunk_rows = [make_row(pairs[start : start + 5], 180, 32) for start in range(0, rows_needed * 5, 5) if pairs[start : start + 5]]
            sheet = stack_rows(chunk_rows)
            path = out_dir / f"{saved:03d}_{item_id}_soft_skeleton.png"
            sheet.save(path)
            all_rows.append(path)
            saved += 1
            if saved >= args.limit:
                break
        if saved >= args.limit:
            break

    index = out_dir / "index.html"
    cards = "\n".join(
        f"<figure><img src='{path.name}'><figcaption>{path.name}</figcaption></figure>" for path in all_rows
    )
    index.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        "body{font-family:sans-serif;background:#f5f2eb;color:#1f2933;margin:24px}"
        "figure{background:white;border:1px solid #ddd;padding:12px;margin:0 0 18px}"
        "img{max-width:100%;height:auto;display:block}figcaption{margin-top:8px;color:#667085}"
        "</style></head><body>"
        f"<h1>Soft skeleton debug</h1><p>Iterations used for training view: {args.iterations}. "
        "Target skeleton sweep is shown to inspect iteration behavior.</p>"
        + cards
        + "</body></html>",
        encoding="utf-8",
    )
    print(f"Saved {saved} soft skeleton visualizations to {out_dir}")
    print(f"Open {index}")


if __name__ == "__main__":
    main()
