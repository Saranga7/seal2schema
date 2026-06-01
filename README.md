# Seal-to-Schema Material Reconstruction

This project reconstructs idealized Byzantine monogram schemas from degraded seal
impressions. The pipeline supports paired training on the 332 usable q0-q2
seal-schema-mask pairs, retrieval-based pseudo-labeling of the 285 unpaired seals,
and evaluation of generated schemas.

## Data

Defaults point to the local dataset layout:

```text
/scratch/mahantas/datasets/MonogramSchema_Seal_pairs/
  metadata.csv
  seals/
  schemas_standardized_224/
  binary_masks/
  ann/

/scratch/mahantas/datasets/285_monograms_segmentations/
  */img/
  */binary_masks/
  */ann/
```

`metadata.csv` is canonical. Image paths are derived from `monogram_id`; q3
rows are excluded because they lack masks, and the paired loader validates
exactly 332 metadata-backed q0-q2 pairs. The unpaired loader discovers the 285
collection images from each `img/` directory and requires a matching
`binary_masks/` image and Supervisely annotation JSON.

## Setup

```bash
uv sync
```

## Core Commands

Create deterministic train/val/test splits:

```bash
uv run python main.py mode=create_splits
```

The default split is stratified 80/10/10, yielding roughly 265 train, 33 val,
and 34 test examples.

Smoke-test the full pix2pix path:

```bash
uv run python main.py mode=smoke_test data.num_workers=0
```

Train pix2pix:

```bash
uv run python main.py mode=train model=pix2pix
```

Input ablations are controlled by `data.input_mode`:

```bash
# Default: RGB seal crop + binary mask, 4 channels
uv run python main.py mode=train model=pix2pix data.input_mode=seal_mask model.in_channels=4

# Mask-only geometry baseline, 1 channel
uv run python main.py mode=train model=pix2pix data.input_mode=mask_only model.in_channels=1

# Seal-only texture baseline, 3 channels
uv run python main.py mode=train model=pix2pix data.input_mode=seal_only model.in_channels=3
```

Train a stronger ResNet-encoder U-Net generator:

```bash
uv run python main.py mode=train model=resnet_unet data.batch_size=16
```

`resnet_unet` uses ImageNet-pretrained torchvision weights by default. This
machine already has ResNet50 weights cached. If weights are not cached on a
different cluster node, either pre-cache them or run with:

```bash
uv run python main.py mode=train model=resnet_unet model.pretrained=false data.batch_size=16
```

Train the compact conditional diffusion baseline:

```bash
uv run python main.py mode=train model=diffusion data.batch_size=2
```

Enable Weights & Biases logging for any training run:

```bash
uv run wandb login
uv run python main.py mode=train model=resnet_unet data.batch_size=16 \
  train.wandb.enabled=true \
  train.wandb.project=seal2schema-sumac
```

Training logs `train/*` losses, `val/*` reconstruction metrics, best-checkpoint
selection values, resolved config, and validation sample grids. Use
`train.wandb.mode=offline` if the compute node has no network access, then sync
the run later with `uv run wandb sync`.

Generate schemas from a trained checkpoint:

```bash
uv run python main.py mode=generate \
  train.checkpoint_path=/path/to/recon/checkpoints/best.pth
```

Generate schemas for the 285 unpaired segmented seals:

```bash
uv run python main.py mode=generate generate.dataset=unpaired \
  train.checkpoint_path=/path/to/recon/checkpoints/best.pth
```

Evaluate a checkpoint directly:

```bash
uv run python main.py mode=evaluate \
  evaluate.checkpoint_path=/path/to/recon/checkpoints/best.pth
```

Evaluate saved generated schemas:

```bash
uv run python main.py mode=evaluate \
  evaluate.generated_dir=/path/to/generated
```

## Retrieval Pseudo-Labels

Pseudo-labeling reuses the retrieval model in
`/scratch/mahantas/cross_modal_retrieval`. Set a retrieval checkpoint either
with `CMR_CHECKPOINT` or as a Hydra override:

```bash
export CMR_CHECKPOINT=/scratch/mahantas/cross_modal_retrieval/.../best_model.pth
uv run python main.py mode=pseudo_label
```

This writes:

```text
outputs/pseudo_labels/pseudo_pairs.csv
```

Train with all pseudo-pairs:

```bash
uv run python main.py mode=train model=pix2pix \
  data.training_set=real332+pseudo_all \
  data.pseudo_pairs_csv=outputs/pseudo_labels/pseudo_pairs.csv
```

Train with confident pseudo-pairs only:

```bash
uv run python main.py mode=train model=pix2pix \
  data.training_set=real332+pseudo_confident \
  data.pseudo_confidence_threshold=0.25 \
  data.pseudo_pairs_csv=outputs/pseudo_labels/pseudo_pairs.csv
```

## Outputs

Hydra writes each run under:

```text
outputs/YYYY-MM-DD/HH-MM-SS_<mode>_<model>/
```

Training runs use 224x224 images by default to match the cross-view retrieval
work. Runs include checkpoints, `history.csv`, samples, and the resolved
configuration. Evaluation writes per-item metrics plus a summary. If evaluating
a generated directory and a retrieval checkpoint is configured, retrieval
self-consistency metrics are also written.
