from __future__ import annotations

import csv
import shutil
from pathlib import Path

from build_supervisor_deck import ASSETS as DECK_ASSETS
from build_supervisor_deck import OUT as DECK_OUT
from build_supervisor_deck import build_html, copy_grid_assets, make_triplet, metric_rows, write_results


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentations" / "sumac_experiment_report_latex"
ASSETS = OUT / "assets"


REPRESENTATIVE_RUNS = [
    ("pix2pix", "outputs/2026-05-26/12-30-05_train_pix2pix", "outputs/2026-05-26/12-37-16_evaluate_pix2pix"),
    ("ResNet-U-Net", "outputs/2026-05-26/12-16-57_train_resnet_unet", "outputs/2026-05-26/12-27-20_evaluate_resnet_unet"),
    ("compact DDPM", "outputs/2026-05-26/12-45-24_train_diffusion", "outputs/2026-05-26/12-58-34_evaluate_diffusion"),
    ("pix2pix mask-only", "outputs/2026-05-26/13-39-05_train_pix2pix_mask_only", "outputs/eval_pix2pix_mask_only_baseline"),
    ("pix2pix seal-only", "outputs/2026-05-26/13-57-04_train_pix2pix_seal_only", "outputs/eval_pix2pix_seal_only_baseline"),
    ("pix2pix skeleton losses", "outputs/2026-05-27/12-21-06_train_pix2pix_skeletonlosses", "outputs/eval_pix2pix_skeleton_best_skeleton_metrics"),
    ("SD15 ControlNet seal+mask", "outputs/2026-05-26/14-47-48_train_stable_diffusion_controlnet", "outputs/eval_sd15_controlnet_seal_mask_332"),
    ("SD15 ControlNet longer training", "outputs/2026-05-26/15-16-05_train_stable_diffusion_controlnet_8k", "outputs/eval_sd15_controlnet_seal_mask_332_8k"),
    ("SD15 mask-only control", "outputs/2026-05-26/17-26-40_train_stable_diffusion_controlnet_guidance0_control1.5", "outputs/eval_sd15_controlnet_mask_only_guidance0_control1p5"),
    ("SD15 LoRA + IP-Adapter", "outputs/2026-05-26/19-08-12_train_stable_diffusion_controlnet_ip", "outputs/eval_sd15_controlnet_lora_ip_mask_3k_r16"),
    ("pretrained scribble ControlNet", "outputs/2026-05-27/sd15_pretrained_scribble_controlnet_mask_1p5k_lr1e-5", "outputs/eval_pretrained_scribble_controlnet_mask_1p5k_lr1e-5"),
    ("pretrained lineart ControlNet", "outputs/2026-05-27/09-21-57_train_stable_diffusion_controlnet", "outputs/eval_pretrained_lineart_controlnet_mask_1p5k_lr1e-5"),
    ("DINOv3-B", "outputs/2026-05-27/14-12-28_dinov3_vitb16_frozen_mask_fpn_cropjitter_sharp", "outputs/eval_dinov3_cropjitter_sharp_skeleton_metrics"),
    ("DINOv3-H+", "outputs/2026-05-27/15-07-47_dinov3_H+_lastblock_cropjitter_lowencLR_structure", "outputs/2026-05-27/15-07-47_dinov3_H+_lastblock_cropjitter_lowencLR_structure/eval_best"),
    ("SAM encoder", "outputs/2026-05-27/16-16-33_sam_vitbase_frozen_mask_decoder_cropjitter_structure", "outputs/2026-05-27/16-16-33_sam_vitbase_frozen_mask_decoder_cropjitter_structure/eval_best"),
    ("DPT-large", "outputs/2026-05-27/17-20-57_dpt_large_pretrained_neck_mask_cropjitter_structure", "outputs/2026-05-27/17-20-57_dpt_large_pretrained_neck_mask_cropjitter_structure/eval_best"),
]


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def fmt(value: str | None) -> str:
    if value in (None, ""):
        return "--"
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return str(value)


def read_master_rows() -> list[dict[str, str]]:
    with (DECK_OUT / "master_results.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def result_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{longtable}{p{0.20\textwidth}rrrrrrp{0.22\textwidth}}",
        r"\caption{Representative quantitative results on the held-out paired test split. Lower is better for MAE and Chamfer; higher is better for SSIM, IoU, Recall, and Skeleton F1.}\\",
        r"\toprule",
        r"Run & MAE & SSIM & IoU & Recall & Skel. F1 & Chamfer & Interpretation \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Run & MAE & SSIM & IoU & Recall & Skel. F1 & Chamfer & Interpretation \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['name'])} & {fmt(row['mae'])} & {fmt(row['ssim'])} & {fmt(row['binary_iou'])} & "
            f"{fmt(row['foreground_recall'])} & {fmt(row['skeleton_f1'])} & {fmt(row['skeleton_chamfer'])} & "
            f"{tex_escape(row['note'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def run_mapping_table() -> str:
    lines = [
        r"\begin{longtable}{p{0.20\textwidth}p{0.36\textwidth}p{0.36\textwidth}}",
        r"\caption{Naming map between report labels and repository output folders.}\\",
        r"\toprule",
        r"Report name & Training output folder & Evaluation output folder \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Report name & Training output folder & Evaluation output folder \\",
        r"\midrule",
        r"\endhead",
    ]
    for name, train_dir, eval_dir in REPRESENTATIVE_RUNS:
        lines.append(f"{tex_escape(name)} & \\path{{{train_dir}}} & \\path{{{eval_dir}}} \\\\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def figure(path: str, caption: str, label: str, width: str = "0.95\\textwidth") -> str:
    return (
        r"\begin{figure}[H]" "\n"
        r"\centering" "\n"
        f"\\includegraphics[width={width}]{{assets/{Path(path).name}}}\n"
        f"\\caption{{{tex_escape(caption)}}}\n"
        f"\\label{{fig:{label}}}\n"
        r"\end{figure}" "\n"
    )


def copy_assets() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for asset in DECK_ASSETS.glob("*.png"):
        shutil.copy2(asset, ASSETS / asset.name)
    for extra in [
        ROOT / "outputs" / "soft_skeleton_debug_pix2pix_skeleton_best" / "000_105__4.11__63_soft_skeleton.png",
        ROOT / "outputs" / "soft_skeleton_debug_targets" / "000_105__4.11__63_soft_skeleton.png",
    ]:
        if extra.exists():
            shutil.copy2(extra, ASSETS / extra.name)


def build_report(rows: list[dict[str, str]]) -> str:
    return r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{float}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{array}
\usepackage{url}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\title{Detailed Experiment Report: SUMAC Seal-to-Schema Generation}
\author{Saranga Mahantasingha}
\date{\today}
\begin{document}
\maketitle

\begin{abstract}
This report documents the seal-to-schema generation experiments implemented in the \texttt{recon\_seal2schema} repository. The target task is to generate a clean canonical Byzantine monogram schema from a degraded seal image and its binary monogram mask. The report covers the dataset setup, input/output contract, training losses, evaluation metrics, model families, representative runs, qualitative results, and current conclusions. It is written as a working record for supervisor review and later conversion into a paper-oriented narrative.
\end{abstract}

\section{Research Objective}
The goal is to synthesize expert-style schema drawings for Byzantine monogram seals. A schema is a clean line drawing that represents the canonical structure of the monogram. In the intended SUMAC workflow, the model receives a segmented seal crop and predicts the corresponding schema. If successful, the generated schemas can be appended to the existing HIP cross-modal retrieval gallery, making retrieval possible for seals that do not have manually drawn schemas.

\section{Dataset and Split}
The paired dataset is the Adolfo--Eidelstein collection under \path{/scratch/mahantas/datasets/MonogramSchema_Seal_pairs}. The canonical training set contains 332 q0--q2 paired samples. Each usable sample has a seal image, a binary monogram mask, an annotation file, and a schema image. The 18 q3 samples are excluded from these experiments because the segmentation masks are absent. The split is stratified by quality label and contains 265 training samples, 33 validation samples, and 34 test samples. The 285 segmented seals under \path{/scratch/mahantas/datasets/285_monograms_segmentations} are reserved for future inference and gallery expansion; they are not used for supervised training in the experiments reported here.

""" + figure("task_triplet.png", "Example task triplet used in the report: seal crop, binary mask, and expert schema.", "task-triplet") + r"""

\section{Input and Output Contract}
The standard model input is \texttt{seal\_mask}: an RGB seal crop concatenated channel-wise with a one-channel binary mask, giving four input channels. The output is a one-channel schema image normalized to the training range. The active image size is 224 unless a pretrained model internally resizes its own input; for example, the DPT wrapper feeds the pretrained DPT model at 384 while returning a 224 schema.

\section{Training Losses}
For supervised reconstructors, the base loss is a weighted sum:
\[
L = \lambda_{L1}L_{L1} + \lambda_{BCE}L_{BCE} + \lambda_{Dice}L_{Dice}
    + \lambda_{clDice}L_{clDice} + \lambda_{skel}L_{skel-L1}.
\]
The weighted L1 term measures pixel reconstruction error and upweights schema foreground strokes. BCE treats the schema as a foreground/background probability map. Dice loss encourages thresholded foreground overlap and is useful because the schema foreground is sparse. Skeleton clDice and skeleton L1 are topology-aware terms. They first compute differentiable soft skeletons from the predicted and target schema probabilities and then penalize topological mismatch. The soft skeletonization uses repeated differentiable erosions and openings; the main experiments use 20 iterations.

The adversarial pix2pix and ResNet-U-Net baselines also use a PatchGAN-style discriminator with a GAN loss weight. The compact DDPM uses denoising diffusion noise-prediction loss. Stable Diffusion runs use the diffusion objective inside the SD/ControlNet fine-tuning path, optionally with LoRA and IP-Adapter conditioning.

\section{Evaluation Metrics}
MAE is the average absolute pixel error after mapping images to [0,1]. SSIM measures local structural similarity. Binary IoU thresholds predicted and target schemas and computes foreground intersection over union. Foreground recall measures how much target foreground is recovered; high recall can also indicate overly thick predictions. Skeleton precision, recall, F1, and chamfer are computed after hard skeletonization. Skeleton F1 measures exact skeleton overlap. Skeleton chamfer is the bidirectional average distance between predicted and target skeleton pixels, capped by the configured search radius. These metrics are strict and do not fully replace qualitative inspection.

\section{Model Families and Methodological Steps}
\subsection{Scratch and compact baselines}
The first experiments established reference behavior using pix2pix, ResNet-U-Net, and a compact DDPM. Pix2pix uses a U-Net generator and PatchGAN discriminator. ResNet-U-Net uses a pretrained ResNet-50-style encoder inside an encoder-decoder generator. The compact DDPM is a small conditional diffusion baseline trained from scratch. These runs showed that the task is learnable to some extent, but training a generator from scratch on 332 pairs is not enough for reliable monogram topology.

\subsection{Input conditioning ablations}
The next experiments tested whether the seal image and mask carry complementary information. The mask-only pix2pix run uses only the binary mask as input. The seal-only run removes the mask. These ablations showed that neither source is sufficient alone: the mask provides support and localization, while the seal crop contains internal visual evidence.

\subsection{Stable Diffusion adaptation}
Stable Diffusion experiments attempted to use a much stronger pretrained generative prior. The repository includes SD15 ControlNet-style training, longer training, mask-only conditioning, LoRA, IP-Adapter seal-image conditioning, and pretrained scribble/lineart ControlNet initializations. These outputs often look smoother or more image-like, but they hallucinate schema-like strokes and do not reliably preserve the target monogram structure. This is an important negative result: generic image-generation quality does not equal schema correctness.

\subsection{Topology-aware pix2pix}
The skeleton-loss pix2pix run added soft skeleton clDice and skeleton L1 to the supervised reconstruction loss. This improved the scratch-family result and provided evidence that topology-aware supervision matters. However, the model architecture still limits structural quality.

\subsection{Pretrained visual and dense models}
The final group uses stronger pretrained models. DINOv3-B and DINOv3-H+ provide visual representation priors; the mask enters through a trainable mask pyramid and FPN-style decoder. SAM was tried because it is segmentation-pretrained, but it underperformed because its prior favors filled object regions rather than internal line topology. DPT-large reuses a pretrained dense-prediction encoder and neck, replacing only the final output head with a schema head and injecting the mask into the dense feature map. DINO and DPT are therefore the current strongest directions.

\section{Quantitative Results}
""" + result_table(rows) + r"""

\section{Qualitative Results}
The qualitative grids shown here are standardized to the first target/generated block. In the original evaluator, grids can contain different numbers of row blocks because evaluation batch sizes differed. After standardization, the top row is the ground-truth schema and the bottom row is the generated schema for the same examples.

""" + figure("baseline_grids_pix2pix.png", "pix2pix qualitative result.", "pix2pix") + figure("baseline_grids_resnet_u_net.png", "ResNet-U-Net qualitative result.", "resnet") + figure("baseline_grids_compact_ddpm.png", "Compact DDPM qualitative result.", "ddpm") + r"""

\subsection{Input ablations and skeleton-loss pix2pix}
""" + figure("ablation_grids_pix2pix__mask_only.png", "pix2pix mask-only qualitative result.", "mask-only") + figure("ablation_grids_pix2pix__seal_only.png", "pix2pix seal-only qualitative result.", "seal-only") + figure("ablation_grids_pix2pix___skeleton_losses.png", "pix2pix with skeleton losses qualitative result.", "pix2pix-skeleton") + r"""

\subsection{Stable Diffusion variants}
The Stable Diffusion variants are important to show because they can look visually smoother while being structurally unreliable. This is the clearest example of the gap between image quality and monogram correctness.

""" + figure("sd_grids_sd15_seal___mask.png", "SD15 ControlNet with seal+mask conditioning.", "sd-seal-mask") + figure("sd_grids_sd15_longer_training.png", "SD15 ControlNet with longer training.", "sd-longer") + figure("sd_grids_sd15_mask_only.png", "SD15 mask-only conditioning.", "sd-mask-only") + figure("sd_grids_sd15_lora___ip_adapter.png", "SD15 ControlNet with LoRA and IP-Adapter.", "sd-lora-ip") + figure("sd_grids_pretrained_scribble_controlnet.png", "Pretrained scribble ControlNet.", "sd-scribble") + figure("sd_grids_pretrained_lineart_controlnet.png", "Pretrained lineart ControlNet.", "sd-lineart") + r"""

\subsection{Pretrained encoders and dense decoders}
""" + figure("pretrained_grids_dinov3_b.png", "DINOv3-B qualitative result.", "dino-b") + figure("pretrained_grids_dinov3_h.png", "DINOv3-H+ qualitative result.", "dino-h") + figure("pretrained_grids_sam_encoder.png", "SAM encoder qualitative result.", "sam") + figure("pretrained_grids_dpt_large.png", "DPT-large qualitative result.", "dpt") + r"""

\subsection{Direct qualitative comparison}
""" + figure("comparison_grids_pix2pix.png", "Direct comparison panel: pix2pix.", "cmp-pix2pix") + figure("comparison_grids_sd15_longer_training.png", "Direct comparison panel: Stable Diffusion longer training. Outputs are smoother but hallucinated.", "cmp-sd-longer") + figure("comparison_grids_sd15_lora___ip_adapter.png", "Direct comparison panel: Stable Diffusion with LoRA and IP-Adapter.", "cmp-sd-lora") + figure("comparison_grids_dinov3_h.png", "Direct comparison panel: DINOv3-H+.", "cmp-dino-h") + figure("comparison_grids_dpt_large.png", "Direct comparison panel: DPT-large.", "cmp-dpt") + r"""

\section{Soft Skeleton Debugging}
The soft skeleton visualization checks that the topology loss is operating on sensible maps. The target-only sheet shows the target schema, soft skeleton at 20 iterations, and an iteration sweep. The prediction-vs-target sheet adds the model output, prediction skeleton, and absolute skeleton difference. The absolute skeleton difference is a pixelwise mismatch map between predicted and target soft skeletons; it is not a distance metric.

""" + figure("000_105__4.11__63_soft_skeleton.png", "Soft skeleton debug sheet for a pix2pix skeleton-loss prediction. The final panel shows absolute skeleton mismatch.", "soft-skeleton-debug") + r"""

\section{Current Interpretation}
The main lesson is that the bottleneck is not simply image generation quality. The task requires preserving a sparse internal graph-like structure. Scratch models are too weak for the dataset size. Stable Diffusion has a strong natural-image prior but hallucinates strokes. Skeleton losses help, but architectural priors are still important. DINOv3-H+ and DPT-large are the current contenders: DINO is visually strong, while DPT is attractive because it brings a pretrained dense-prediction decoder.

\section{Next Steps}
The next methodological decision is whether to use DINOv3-H+ or DPT-large as the main SUMAC architecture. Checkpoint selection should use skeleton or structural metrics, not only binary IoU. Once paired-test behavior is stable, the selected model should generate schemas for the 285 unpaired segmented seals. The downstream retrieval-gallery expansion experiment should then evaluate whether generated schemas preserve practical retrieval performance.

\appendix
\section{Run Name and Folder Mapping}
""" + run_mapping_table() + r"""

\end{document}
"""


def main() -> None:
    DECK_ASSETS.mkdir(parents=True, exist_ok=True)
    rows = metric_rows()
    write_results(rows)
    grids = copy_grid_assets()
    make_triplet()
    (DECK_OUT / "index.html").write_text(build_html(rows, make_triplet(), grids), encoding="utf-8")
    copy_assets()
    report_rows = read_master_rows()
    (OUT / "main.tex").write_text(build_report(report_rows), encoding="utf-8")
    shutil.copy2(DECK_OUT / "master_results.csv", OUT / "master_results.csv")
    (OUT / "README.md").write_text(
        "# SUMAC experiment report LaTeX project\n\n"
        "Upload this folder to Overleaf and compile `main.tex` with pdfLaTeX.\n"
        "The report includes the representative metrics table, qualitative figures, soft-skeleton debug figure, and run-folder appendix.\n",
        encoding="utf-8",
    )
    print(OUT / "main.tex")


if __name__ == "__main__":
    main()
