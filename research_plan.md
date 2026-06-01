# Byzantine Monogram Research Plan
## Two-Stage: SUMAC 2026 → CVPR 2027

---

## Datasets

### Dataset A — Adolfo-Eidelstein Collection (paired, primary)
```
/scratch/mahantas/datasets/MonogramSchema_Seal_pairs
```
- **350** Byzantine seal images, each with a corresponding **schema** (clean B&W expert line drawing of the monogram)
- **332** of the 350 have **segmentation masks** on Supervisely (polygon + binary mask), covering quality labels q0, q1, q2
- **18** are q3 (very low quality) — have schemas but no segmentation masks; exclude from training
- Quality distribution: q0=48, q1=159, q2=125, q3=18. Metadata of quality labels is in /scratch/mahantas/datasets/MonogramSchema_Seal_pairs/metadata.csv (quality_label is the correct column). 
- From each of the 332 annotated samples:
  - (a) full seal image (datasets/MonogramSchema_Seal_pairs/seals)
  - (b) monogram region crop (polygon coordinates) (datasets/MonogramSchema_Seal_pairs/ann)
  - (c) binary mask aligned to the crop (datasets/MonogramSchema_Seal_pairs/binary_masks)

### Dataset B — Other Collections (unpaired, secondary)
```
/scratch/mahantas/datasets/285_monograms_segmentations
```
- **285** monogram seal images from collections other than Adolfo-Eidelstein
- Each has a **monogram segmentation** (so crops are available), but **no schema**
- Used only at inference time — never for training
- Purpose: generate schemas for these 285 seals to expand the retrieval gallery

---

## Paper 1: SUMAC 2026
**Deadline: July 16, 2026**
**Venue: SUMAC Workshop @ ACM Multimedia 2026, November 10–14, Rio de Janeiro, Brazil**
**Format: ACM, 8+2 pages (full paper). Downgrade to 4+2 short paper only if diffusion results are not ready — decide at end of Stage 4.**

### Title (working)
*"Monogram-to-Schema Translation for Byzantine Seal Analysis via Conditional Image Generation"*

### Problem
Given a degraded seal photograph containing a monogram, automatically generate its clean canonical schema — the expert line-drawing representation used by historians to read the monogram. This replaces a manual, expert-intensive process and extends the usability of the existing cross-view retrieval system (HIP 2026 paper) to seals that have never been manually annotated with a schema.

### Data Used
- **Training/validation/test**: 332 pairs from Dataset A (seal crop + binary mask → schema)
- **Inference only**: 285 seals from Dataset B (generate schemas, never train on them)

### Train/Val/Test Split
- Stratify by quality label to ensure all quality levels are represented in each split
- 80/10/10 split → ~265 train / ~33 val / ~34 test
- Keep q0 samples proportionally represented in test (fewest but cleanest — best for qualitative evaluation)

### Model Pipeline

**Input**: monogram crop (RGB or grayscale, ~256×256) + binary segmentation mask as conditioning signal

**Baseline — pix2pix with modern backbone**
- Pretrained ResNet-50 or ViT-S encoder (ImageNet weights), U-Net decoder
- Input: monogram crop concatenated channel-wise with binary mask
- Output: schema image
- Loss: L1 + adversarial (PatchGAN discriminator)
- Augmentation on seal crops only: brightness/contrast jitter, Gaussian blur, elastic distortion, random affine — schemas get only mild affine transforms to preserve geometry

**Main model — ControlNet-style diffusion fine-tuning**
- Lightweight diffusion backbone (Stable Diffusion) with LoRA fine-tuning (~20M parameter update, start with rank r=16)
- Control signal: binary segmentation mask (structural boundary information)
- Image conditioning: monogram crop via IP-Adapter or cross-attention injection
- Fine-tune on ~265 training pairs; validate on ~33 val pairs

### Evaluation Metrics
1. **FID** on held-out test schemas — distributional quality
2. **SSIM / LPIPS** — per-sample structural similarity to ground-truth schemas
3. **Qualitative comparison** across quality levels (q0 vs q1 vs q2 inputs) — generation degrades gracefully

**Key downstream experiment**:
- Run inference on 285 Dataset B seals → generate schemas
- Add these 285 (seal, generated-schema) pairs to the gallery of the HIP 2026 retrieval system
- Evaluate R@1, R@5, R@10 — show gallery expansion improves or preserves retrieval performance
- This is the practical payoff: the system now works for seals that have never been manually annotated

### Ablations
- Train on q0+q1 only → test on q2 inputs (effect of training data quality on degraded inputs)
- With mask conditioning vs. without (does the segmentation mask as control signal help?)
- pix2pix baseline vs. diffusion fine-tune

---

## Paper 2: CVPR 2027
**Deadline: ~November 2026 (TBC)**
**Venue: CVPR 2027, Seattle, June 19–26, 2027**

### Title (working)
*"Canonical Form Synthesis and Symbol-Set Recognition for Degraded Historical Artifacts"*

### Core Argument
Many historical artifact categories share a common structure: a *degraded physical impression* (seal photograph, coin rubbing, watermark scan) paired with a *canonical idealized form* (schema, die reconstruction, reference drawing). Current methods handle retrieval between these views but not *generation* — predicting the canonical form from the degraded impression, nor *interpretation* — reading the symbolic content of the artifact. This paper proposes a general framework for both, demonstrated on two independent datasets.

### What is New vs. SUMAC Paper
SUMAC = proof of concept on Byzantine monograms, generation only.
CVPR adds:

1. **A second dataset** for generality — e.g. Chinese seal carvings (CSCD dataset, structurally analogous: characters superimposed in a square frame). To be selected and obtained after SUMAC submission.
2. **Character-set prediction** on generated schemas — multi-label classification predicting *which* Greek letters are present. Framed as **unordered set prediction**, not sequential OCR. This framing is novel and justified by monogram structure: letters are spatially superimposed with no reading order.
3. **Name prediction** from predicted letter sets — probabilistic ranking over Byzantine name corpus given a noisy predicted character set.

### Three-Component Pipeline

```
Seal Image
    ↓  (segmentation mask)
Monogram Crop
    ↓  [Component 1: Schema Synthesis — from SUMAC]
Generated Schema
    ↓  [Component 2: Character-Set Prediction — NEW for CVPR]
Predicted Letter Set  e.g. {Θ, Ε, Ο, Δ, Ω, Ρ, Σ}
    ↓  [Component 3: Name Ranking — NEW for CVPR]
Ranked Name Candidates: ["Theodoros", "Theodosios", ...]
```

### Component 2: Character-Set Prediction
- Input: schema image (real or generated)
- Output: multi-label binary vector over Greek alphabet (~24 classes)
- Architecture: ViT-B or ResNet-50 + multi-label classification head (BCE loss)
- **Do not use noisy human-entered letter annotations for training.** Instead, run a Greek character detector on the clean schema images directly to generate reliable pseudo-labels. Schemas are clean B&W line drawings — far easier to recognize characters on than seal photographs.

### Component 3: Name Prediction
- Input: predicted letter set (possibly noisy/incomplete)
- Retrieve candidate names from Byzantine prosopography corpus where the name's letters are a superset of the predicted set (monograms can omit letters and collapse repetitions)
- Rank by: (a) frequency in historical corpus, (b) onomastic constraints from domain experts
- Evaluation: recall of correct name in top-K candidates

### Data Used (CVPR)
- All of Dataset A and B (same as SUMAC) for Byzantine monogram component
- Second dataset (TBD) for generality claim
- Letter-set annotations from Dataset A schemas — cleaned or pseudo-labeled via character detector
- Byzantine name corpus / prosopography database (existing sigillographic resources)

### Stages After SUMAC Submission

1. Select and obtain second dataset
2. Implement character-set prediction on clean schemas (pseudo-label pipeline first)
3. Semi-supervised extension: leverage Dataset B (285 unpaired seals) more effectively in generation training
4. Second dataset experiments — establish generality claim
5. Full end-to-end pipeline evaluation (seal → name candidates)
6. Ablations and polish
7. Submit

---

## Summary of Data Usage per Paper

| Data | SUMAC | CVPR |
|------|-------|------|
| 332 paired (seal crop + mask + schema) | Train/val/test for generation | Same + character-set prediction training |
| 18 q3 pairs (schema only, no mask) | Excluded | Potentially for schema-only recognition |
| 285 unpaired seals (Dataset B) | Inference only — gallery expansion | Inference + semi-supervised training signal |
| Letter-set annotations | Not used | Pseudo-labeled from clean schemas |
| Byzantine name corpus | Not used | Name ranking (Component 3) |
| Second dataset (TBD) | Not used | Generality claim |

---

## Key Technical Decisions
- **No Blender / no 3D rendering** — all generation is 2D using modern pretrained foundation models with LoRA fine-tuning
- **Segmentation mask as conditioning signal** — provides structural boundary information cheaply
- **Downstream retrieval as evaluation** — connects generation quality to a concrete, measurable practical benefit (directly extends HIP 2026 paper)
- **Character recognition on schemas, not seals** — schemas are clean B&W; avoids reliance on noisy manual letter annotations
- **Set prediction, not sequential OCR** — the correct framing for monogram letter identification; letters have no reading order
