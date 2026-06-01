from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentations" / "supervisor_sumac_experiments"
ASSETS = OUT / "assets"
LATEX_OUT = ROOT / "presentations" / "supervisor_sumac_experiments_latex"


RESULTS = [
    ("Baselines", "pix2pix", "outputs/2026-05-26/12-37-16_evaluate_pix2pix/metrics_summary.csv", "First usable paired image-translation baseline."),
    ("Baselines", "ResNet-U-Net", "outputs/2026-05-26/12-27-20_evaluate_resnet_unet/metrics_summary.csv", "Convolutional encoder-decoder baseline."),
    ("Baselines", "compact DDPM", "outputs/2026-05-26/12-58-34_evaluate_diffusion/metrics_summary.csv", "Small diffusion model trained from scratch."),
    ("Input ablations", "pix2pix: mask only", "outputs/eval_pix2pix_mask_only_baseline/metrics_summary.csv", "Tests whether mask support alone can predict schema."),
    ("Input ablations", "pix2pix: seal only", "outputs/eval_pix2pix_seal_only_baseline/metrics_summary.csv", "Tests whether seal appearance is enough without mask conditioning."),
    ("Topology-aware training", "pix2pix + skeleton losses", "outputs/eval_pix2pix_skeleton_best_skeleton_metrics/metrics_summary.csv", "Adds soft skeleton clDice and skeleton L1 to the supervised loss."),
    ("Stable Diffusion", "SD15 ControlNet: seal + mask", "outputs/eval_sd15_controlnet_seal_mask_332/metrics_summary.csv", "Stable Diffusion with trained ControlNet-style conditioning."),
    ("Stable Diffusion", "SD15 ControlNet: longer training", "outputs/eval_sd15_controlnet_seal_mask_332_8k/metrics_summary.csv", "Same family with more optimization steps."),
    ("Stable Diffusion", "SD15 ControlNet: mask only", "outputs/eval_sd15_controlnet_mask_only_guidance0_control1p5/metrics_summary.csv", "Removes seal image conditioning; emphasizes mask conditioning."),
    ("Stable Diffusion", "SD15 ControlNet + LoRA + IP-Adapter", "outputs/eval_sd15_controlnet_lora_ip_mask_3k_r16/metrics_summary.csv", "Adds LoRA updates and seal-image IP-Adapter conditioning."),
    ("Stable Diffusion", "pretrained scribble ControlNet", "outputs/eval_pretrained_scribble_controlnet_mask_1p5k_lr1e-5/metrics_summary.csv", "Uses an external scribble ControlNet prior."),
    ("Stable Diffusion", "pretrained lineart ControlNet", "outputs/eval_pretrained_lineart_controlnet_mask_1p5k_lr1e-5/metrics_summary.csv", "Uses an external line-art ControlNet prior."),
    ("Pretrained encoders / decoders", "DINOv3-B", "outputs/eval_dinov3_cropjitter_sharp_skeleton_metrics/metrics_summary.csv", "Frozen visual representation encoder plus trainable mask-aware decoder."),
    ("Pretrained encoders / decoders", "DINOv3-H+", "outputs/2026-05-27/15-07-47_dinov3_H+_lastblock_cropjitter_lowencLR_structure/eval_best/metrics_summary.csv", "Larger DINO encoder with gentle last-block adaptation."),
    ("Pretrained encoders / decoders", "SAM encoder", "outputs/2026-05-27/16-16-33_sam_vitbase_frozen_mask_decoder_cropjitter_structure/eval_best/metrics_summary.csv", "Segmentation-pretrained encoder plus trainable decoder."),
    ("Pretrained encoders / decoders", "DPT-large", "outputs/2026-05-27/17-20-57_dpt_large_pretrained_neck_mask_cropjitter_structure/eval_best/metrics_summary.csv", "Pretrained dense-prediction encoder and neck with schema head."),
]


GRID_GROUPS = {
    "baseline_grids": [
        ("pix2pix", "outputs/2026-05-26/12-37-16_evaluate_pix2pix/qualitative/test_target_generated_grid.png"),
        ("ResNet-U-Net", "outputs/2026-05-26/12-27-20_evaluate_resnet_unet/qualitative/test_target_generated_grid.png"),
        ("compact DDPM", "outputs/2026-05-26/12-58-34_evaluate_diffusion/qualitative/test_target_generated_grid.png"),
    ],
    "ablation_grids": [
        ("pix2pix: mask only", "outputs/eval_pix2pix_mask_only_baseline/qualitative/test_target_generated_grid.png"),
        ("pix2pix: seal only", "outputs/eval_pix2pix_seal_only_baseline/qualitative/test_target_generated_grid.png"),
        ("pix2pix + skeleton losses", "outputs/eval_pix2pix_skeleton_best_skeleton_metrics/qualitative/test_target_generated_grid.png"),
    ],
    "sd_grids": [
        ("SD15 seal + mask", "outputs/eval_sd15_controlnet_seal_mask_332/qualitative/test_target_generated_grid.png"),
        ("SD15 longer training", "outputs/eval_sd15_controlnet_seal_mask_332_8k/qualitative/test_target_generated_grid.png"),
        ("SD15 mask only", "outputs/eval_sd15_controlnet_mask_only_guidance0_control1p5/qualitative/test_target_generated_grid.png"),
        ("SD15 LoRA + IP-Adapter", "outputs/eval_sd15_controlnet_lora_ip_mask_3k_r16/qualitative/test_target_generated_grid.png"),
        ("pretrained scribble ControlNet", "outputs/eval_pretrained_scribble_controlnet_mask_1p5k_lr1e-5/qualitative/test_target_generated_grid.png"),
        ("pretrained lineart ControlNet", "outputs/eval_pretrained_lineart_controlnet_mask_1p5k_lr1e-5/qualitative/test_target_generated_grid.png"),
    ],
    "pretrained_grids": [
        ("DINOv3-B", "outputs/eval_dinov3_cropjitter_sharp_skeleton_metrics/qualitative/test_target_generated_grid.png"),
        ("DINOv3-H+", "outputs/2026-05-27/15-07-47_dinov3_H+_lastblock_cropjitter_lowencLR_structure/eval_best/qualitative/test_target_generated_grid.png"),
        ("SAM encoder", "outputs/2026-05-27/16-16-33_sam_vitbase_frozen_mask_decoder_cropjitter_structure/eval_best/qualitative/test_target_generated_grid.png"),
        ("DPT-large", "outputs/2026-05-27/17-20-57_dpt_large_pretrained_neck_mask_cropjitter_structure/eval_best/qualitative/test_target_generated_grid.png"),
    ],
    "comparison_grids": [
        ("pix2pix", "outputs/2026-05-26/12-37-16_evaluate_pix2pix/qualitative/test_target_generated_grid.png"),
        ("SD15 longer training", "outputs/eval_sd15_controlnet_seal_mask_332_8k/qualitative/test_target_generated_grid.png"),
        ("SD15 LoRA + IP-Adapter", "outputs/eval_sd15_controlnet_lora_ip_mask_3k_r16/qualitative/test_target_generated_grid.png"),
        ("DINOv3-H+", "outputs/2026-05-27/15-07-47_dinov3_H+_lastblock_cropjitter_lowencLR_structure/eval_best/qualitative/test_target_generated_grid.png"),
        ("DPT-large", "outputs/2026-05-27/17-20-57_dpt_large_pretrained_neck_mask_cropjitter_structure/eval_best/qualitative/test_target_generated_grid.png"),
    ],
}


def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def fmt(value: str | None, digits: int = 3) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def read_summary(rel_path: str) -> dict[str, str]:
    path = ROOT / rel_path
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return next(csv.DictReader(f))


def metric_rows() -> list[dict[str, str]]:
    rows = []
    for group, name, rel_path, note in RESULTS:
        metrics = read_summary(rel_path)
        rows.append(
            {
                "group": group,
                "name": name,
                "mae": metrics.get("mae", ""),
                "ssim": metrics.get("ssim", ""),
                "binary_iou": metrics.get("binary_iou", ""),
                "foreground_recall": metrics.get("foreground_recall", ""),
                "skeleton_f1": metrics.get("skeleton_f1", ""),
                "skeleton_chamfer": metrics.get("skeleton_chamfer", ""),
                "note": note,
                "path": rel_path,
            }
        )
    return rows


def write_results(rows: list[dict[str, str]]) -> None:
    fields = ["group", "name", "mae", "ssim", "binary_iou", "foreground_recall", "skeleton_f1", "skeleton_chamfer", "note", "path"]
    with (OUT / "master_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "master_results.json").open("w") as f:
        json.dump(rows, f, indent=2)


def copy_grid_assets() -> dict[str, list[tuple[str, str]]]:
    copied: dict[str, list[tuple[str, str]]] = {}
    for group, items in GRID_GROUPS.items():
        copied[group] = []
        for label, rel_path in items:
            src = ROOT / rel_path
            if not src.exists():
                continue
            dst = ASSETS / f"{group}_{safe_name(label)}.png"
            image = Image.open(src).convert("RGB")
            # Evaluation grids concatenate up to three batches depending on eval batch size.
            # Crop every grid to the first target/generated block for fair visual comparison.
            crop_height = min(image.height, 454)
            image.crop((0, 0, image.width, crop_height)).save(dst)
            copied[group].append((label, f"assets/{dst.name}"))
    return copied


def make_triplet() -> str:
    with (ROOT / "splits" / "test.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    panels = [
        ("Seal crop", Image.open(row["seal_path"]).convert("RGB")),
        ("Binary mask", Image.open(row["mask_path"]).convert("L").convert("RGB")),
        ("Expert schema", Image.open(row["schema_path"]).convert("L").convert("RGB")),
    ]
    tile_w, tile_h = 320, 320
    header_h = 42
    gap = 18
    canvas = Image.new("RGB", (tile_w * 3 + gap * 2, tile_h + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(panels):
        img = ImageOps.contain(img, (tile_w, tile_h), method=Image.Resampling.LANCZOS)
        x = idx * (tile_w + gap)
        y = header_h + (tile_h - img.height) // 2
        canvas.paste(img, (x + (tile_w - img.width) // 2, y))
        draw.text((x + 8, 10), label, fill=(30, 36, 44))
    dst = ASSETS / "task_triplet.png"
    canvas.save(dst)
    return f"assets/{dst.name}"


def table(rows: list[dict[str, str]], names: list[str], cols: list[tuple[str, str]], small: bool = False) -> str:
    selected = [row for row in rows if row["name"] in names]
    cls = " class='small-table'" if small else ""
    head = "<tr><th>Experiment</th>" + "".join(f"<th>{html.escape(label)}</th>" for _, label in cols) + "<th>Interpretation</th></tr>"
    body = []
    for row in selected:
        cells = [f"<td>{html.escape(row['name'])}</td>"]
        cells.extend(f"<td>{fmt(row.get(key))}</td>" for key, _ in cols)
        cells.append(f"<td>{html.escape(row['note'])}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table{cls}>" + head + "".join(body) + "</table>"


def all_results_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['group'])}</td>"
            f"<td>{html.escape(row['name'])}</td>"
            f"<td>{fmt(row['mae'])}</td>"
            f"<td>{fmt(row['ssim'])}</td>"
            f"<td>{fmt(row['binary_iou'])}</td>"
            f"<td>{fmt(row['foreground_recall'])}</td>"
            f"<td>{fmt(row['skeleton_f1'])}</td>"
            f"<td>{fmt(row['skeleton_chamfer'])}</td>"
            f"<td>{html.escape(row['path'])}</td>"
            "</tr>"
        )
    return (
        "<table class='tiny-table'><tr><th>Family</th><th>Run</th><th>MAE</th><th>SSIM</th><th>IoU</th>"
        "<th>Recall</th><th>Skel F1</th><th>Chamfer</th><th>Metric file</th></tr>"
        + "".join(body)
        + "</table>"
    )


def grid_gallery(items: list[tuple[str, str]], cols: int = 3) -> str:
    cards = []
    for label, path in items:
        cards.append(
            f"<figure><img src='{html.escape(path)}' alt='{html.escape(label)}'>"
            f"<figcaption>{html.escape(label)}</figcaption></figure>"
        )
    return f"<div class='grid-gallery cols-{cols}'>" + "".join(cards) + "</div>"


def slide(title: str, content: str, cls: str = "") -> str:
    return f"<section class='slide {cls}'><div class='inner'><h2>{title}</h2>{content}</div></section>"


def build_html(rows: list[dict[str, str]], triplet: str, grids: dict[str, list[tuple[str, str]]]) -> str:
    metric_cols = [
        ("mae", "MAE"),
        ("ssim", "SSIM"),
        ("binary_iou", "IoU"),
        ("foreground_recall", "Recall"),
        ("skeleton_f1", "Skel F1"),
        ("skeleton_chamfer", "Chamfer"),
    ]
    slides = [
        slide(
            "SUMAC seal-to-schema experiments",
            "<p class='subtitle'>Generating clean canonical Byzantine monogram schemas from degraded seal photographs.</p>"
            "<p class='tagline'>Detailed working deck: models tried, training losses, metrics, qualitative results, and current conclusions.</p>",
            "title-slide",
        ),
        slide(
            "Research goal",
            "<ul>"
            "<li>Input: a degraded Byzantine seal image containing a monogram, plus its segmentation mask.</li>"
            "<li>Output: a clean expert-style schema, i.e. a canonical line drawing used for reading and retrieval.</li>"
            "<li>Why it matters: generated schemas could expand the HIP retrieval gallery to seals without manual schema annotation.</li>"
            "</ul>",
        ),
        slide(
            "Dataset and split",
            "<div class='stats'><div><strong>332</strong><span>paired q0-q2 seal/schema/mask samples</span></div>"
            "<div><strong>265 / 33 / 34</strong><span>train / val / test, stratified by quality</span></div>"
            "<div><strong>285</strong><span>unpaired segmented seals reserved for later inference</span></div></div>"
            "<p>q3 samples are excluded from generation training because masks are absent.</p>",
        ),
        slide(
            "Task formulation",
            f"<img class='triplet' src='{triplet}' alt='seal mask schema triplet'>"
            "<p>The model sees a crop of the seal and the aligned binary mask. The schema target is one-channel because it is essentially foreground strokes on background.</p>",
        ),
        slide(
            "How to read the qualitative grids",
            "<ul>"
            "<li>Each original evaluation grid is built as target row followed by generated row.</li>"
            "<li>Different runs originally showed different numbers of row blocks because evaluation batch sizes differed.</li>"
            "<li>For this presentation, every grid is standardized to the first target/generated block so comparisons have the same layout.</li>"
            "<li>The top row is ground-truth schema; the second row is the model output for the same examples.</li>"
            "</ul>",
        ),
        slide(
            "Model families tried",
            "<div class='cards'>"
            "<div><b>Scratch image translation</b><span>pix2pix, ResNet-U-Net, compact DDPM.</span></div>"
            "<div><b>Stable Diffusion adaptation</b><span>ControlNet-style training, mask-only, LoRA, IP-Adapter, pretrained scribble/lineart controls.</span></div>"
            "<div><b>Topology-aware supervision</b><span>Added differentiable soft skeleton losses to supervised image translation.</span></div>"
            "<div><b>Pretrained visual/dense models</b><span>DINOv3, SAM, and DPT variants with mask-aware decoding.</span></div>"
            "</div>",
        ),
        slide(
            "Training losses: supervised reconstructor",
            "<p>The supervised models use a weighted combination of image, foreground, mask-overlap, and topology losses.</p>"
            "<div class='formula'>L = λ<sub>L1</sub>L<sub>L1</sub> + λ<sub>BCE</sub>L<sub>BCE</sub> + λ<sub>Dice</sub>L<sub>Dice</sub> + λ<sub>clDice</sub>L<sub>clDice</sub> + λ<sub>skel</sub>L<sub>skel-L1</sub></div>"
            "<ul>"
            "<li><b>Weighted L1:</b> pixel reconstruction loss, with extra weight on schema foreground strokes.</li>"
            "<li><b>BCE:</b> treats the schema as a foreground/background probability map.</li>"
            "<li><b>Dice:</b> encourages overlap while being less sensitive to foreground sparsity than raw BCE.</li>"
            "</ul>",
        ),
        slide(
            "Training losses: skeleton terms",
            "<div class='diagram'><span>prediction</span><b>soft skeletonize</b><span>predicted skeleton</span><b>compare</b><span>target skeleton</span></div>"
            "<ul>"
            "<li><b>Soft skeletonization:</b> differentiable erosion/opening iterations approximate the centerlines of the generated and target strokes.</li>"
            "<li><b>clDice:</b> rewards topology-preserving overlap between predicted skeleton and target foreground, and between target skeleton and predicted foreground.</li>"
            "<li><b>Skeleton L1:</b> directly penalizes distance in the soft skeleton maps.</li>"
            "<li>Purpose: push the model toward correct monogram structure, not merely thick foreground blobs.</li>"
            "</ul>",
        ),
        slide(
            "Stable Diffusion training strategies",
            "<ul>"
            "<li><b>ControlNet-style conditioning:</b> use the mask or seal+mask as structural control for SD15.</li>"
            "<li><b>Guidance/control scale sweeps:</b> reduce text-prompt influence and increase conditioning strength.</li>"
            "<li><b>LoRA:</b> train a small low-rank update instead of full SD weights.</li>"
            "<li><b>IP-Adapter:</b> add image-conditioning from the seal crop.</li>"
            "<li><b>Pretrained scribble/lineart ControlNets:</b> try external line-art priors before training from scratch.</li>"
            "</ul>",
        ),
        slide(
            "Evaluation metrics",
            "<ul>"
            "<li><b>MAE:</b> average absolute pixel error after normalizing images to [0, 1]. Lower is better.</li>"
            "<li><b>SSIM:</b> local structural similarity. Higher is better.</li>"
            "<li><b>Binary IoU:</b> threshold prediction and target, then measure foreground intersection over union. Higher is better.</li>"
            "<li><b>Foreground recall:</b> fraction of target foreground pixels recovered by the prediction. Higher can also mean over-thick outputs.</li>"
            "</ul>",
        ),
        slide(
            "Skeleton metrics",
            "<ul>"
            "<li><b>Skeleton precision:</b> how much of the generated skeleton lands on the target skeleton.</li>"
            "<li><b>Skeleton recall:</b> how much of the target skeleton is recovered.</li>"
            "<li><b>Skeleton F1:</b> harmonic mean of skeleton precision and recall. Higher is better.</li>"
            "<li><b>Skeleton chamfer:</b> bidirectional average distance between generated and target skeleton pixels, capped by search radius. Lower is better.</li>"
            "<li>These are stricter than visual judgment, but they help detect topology errors that pixel metrics miss.</li>"
            "</ul>",
        ),
        slide(
            "Baseline metrics",
            table(rows, ["pix2pix", "ResNet-U-Net", "compact DDPM"], metric_cols)
            + "<p class='takeaway'>The task is hard for scratch models. pix2pix is the first reasonable baseline; compact DDPM is too weak with this data size.</p>",
        ),
        slide("Baseline qualitative results", grid_gallery(grids["baseline_grids"], cols=3), "qual-slide"),
        slide(
            "Input ablations and skeleton-loss metrics",
            table(rows, ["pix2pix: mask only", "pix2pix: seal only", "pix2pix + skeleton losses"], metric_cols)
            + "<p class='takeaway'>Mask-only and seal-only each lose information. Skeleton-aware losses improve the scratch baseline, but do not solve the architecture bottleneck.</p>",
        ),
        slide("Input ablations and skeleton qualitative results", grid_gallery(grids["ablation_grids"], cols=3), "qual-slide"),
        slide(
            "Stable Diffusion metrics",
            table(
                rows,
                [
                    "SD15 ControlNet: seal + mask",
                    "SD15 ControlNet: longer training",
                    "SD15 ControlNet: mask only",
                    "SD15 ControlNet + LoRA + IP-Adapter",
                    "pretrained scribble ControlNet",
                    "pretrained lineart ControlNet",
                ],
                metric_cols,
                small=True,
            )
            + "<p class='takeaway'>Stable Diffusion variants did not learn the sparse schema domain well enough. Longer training helped, but outputs still missed fine monogram structure.</p>",
        ),
        slide("Stable Diffusion qualitative results", grid_gallery(grids["sd_grids"], cols=3), "qual-slide tall"),
        slide(
            "Qualitative comparison with Stable Diffusion",
            grid_gallery(grids["comparison_grids"], cols=5)
            + "<p class='takeaway'>Stable Diffusion outputs often look smoother or more image-like, but they hallucinate schema-like strokes and do not reliably preserve the target monogram structure.</p>",
            "qual-slide comparison",
        ),
        slide(
            "Pretrained visual/dense model metrics",
            table(rows, ["DINOv3-B", "DINOv3-H+", "SAM encoder", "DPT-large"], metric_cols)
            + "<p class='takeaway'>DINO variants are visually strongest; DPT is interesting because it reuses a pretrained dense-prediction decoder; SAM is less suitable because its prior favors filled regions.</p>",
        ),
        slide("Pretrained visual/dense qualitative results", grid_gallery(grids["pretrained_grids"], cols=4), "qual-slide"),
        slide(
            "Current interpretation",
            "<ul>"
            "<li><b>Training from scratch</b> is not enough for 332 paired samples.</li>"
            "<li><b>Stable Diffusion</b> brings a strong image prior, but the prior is poorly aligned with black-background schema strokes.</li>"
            "<li><b>Skeleton losses</b> are useful because the real target is topology, not stroke thickness.</li>"
            "<li><b>DINOv3 and DPT</b> are the strongest current directions because they bring pretrained representation or dense decoding priors.</li>"
            "<li><b>Metrics are imperfect:</b> IoU can favor thick foreground, while visual correctness often cares about monogram graph structure.</li>"
            "</ul>",
        ),
        slide(
            "Next decisions before SUMAC",
            "<ul>"
            "<li>Choose the main architecture for the paper: DINO-H+ or DPT-large.</li>"
            "<li>Select checkpoints using skeleton/structure metrics, not only binary IoU.</li>"
            "<li>Only then generate schemas for the 285 unpaired seals.</li>"
            "<li>Use retrieval-gallery evaluation as the downstream test of practical value.</li>"
            "</ul>",
        ),
        slide(
            "Questions for supervisors",
            "<ul>"
            "<li>Should visual structural correctness be prioritized over binary IoU?</li>"
            "<li>Does DINO-H+ or DPT better match the intended SUMAC story?</li>"
            "<li>Should downstream retrieval performance become the primary success criterion?</li>"
            "<li>Are there domain constraints that should be encoded into loss or post-processing?</li>"
            "</ul>",
        ),
        slide("Appendix: representative result ledger", all_results_table(rows), "appendix"),
    ]
    css = """
    :root { --ink:#1f2933; --muted:#65758b; --accent:#0f766e; --bg:#f7f4ef; --panel:#fff; --line:#d8d2c8; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
    .slide { min-height:100vh; display:none; padding:34px 42px; }
    .slide.active { display:flex; align-items:center; justify-content:center; }
    .inner { width:min(1240px, 100%); }
    h2 { font-size:43px; line-height:1.05; margin:0 0 22px; letter-spacing:0; }
    p, li { font-size:23px; line-height:1.34; }
    ul { margin:0; padding-left:32px; }
    li { margin:11px 0; }
    table { border-collapse:collapse; width:100%; background:var(--panel); border:1px solid var(--line); }
    th, td { padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; font-size:18px; vertical-align:top; }
    th { background:#ece7df; color:#344054; font-weight:750; }
    .small-table th, .small-table td { font-size:14px; padding:8px 9px; }
    .tiny-table th, .tiny-table td { font-size:12px; padding:6px 7px; }
    .title-slide { background:#e9efe9; }
    .title-slide h2 { font-size:62px; max-width:1050px; }
    .subtitle { font-size:33px; max-width:960px; color:#25433f; }
    .tagline, .takeaway { color:var(--muted); font-size:21px; margin-top:18px; }
    .stats, .cards { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:22px; }
    .cards { grid-template-columns:repeat(2,1fr); }
    .stats div, .cards div { background:white; border:1px solid var(--line); padding:20px; min-height:120px; }
    .stats strong { display:block; font-size:40px; color:var(--accent); }
    .stats span, .cards span { display:block; color:var(--muted); font-size:19px; margin-top:8px; }
    .cards b { font-size:24px; color:#25433f; }
    .triplet { width:100%; max-height:500px; object-fit:contain; background:white; border:1px solid var(--line); }
    .formula { font-size:31px; background:white; border:1px solid var(--line); padding:18px 22px; margin:16px 0 18px; color:#25433f; }
    .diagram { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:20px; }
    .diagram span, .diagram b { padding:13px 16px; background:white; border:1px solid var(--line); font-size:20px; }
    .diagram b { color:var(--accent); }
    .grid-gallery { display:grid; gap:12px; align-items:start; }
    .grid-gallery.cols-3 { grid-template-columns:repeat(3, 1fr); }
    .grid-gallery.cols-4 { grid-template-columns:repeat(4, 1fr); }
    .grid-gallery.cols-5 { grid-template-columns:repeat(5, 1fr); }
    figure { margin:0; background:white; border:1px solid var(--line); padding:8px; }
    figure img { width:100%; height:480px; object-fit:contain; display:block; }
    .tall figure img { height:330px; }
    .comparison figure img { height:360px; }
    figcaption { font-size:15px; color:var(--muted); margin-top:7px; text-align:center; }
    .qual-slide h2 { margin-bottom:14px; }
    .appendix h2 { font-size:32px; margin-bottom:16px; }
    .nav { position:fixed; right:18px; bottom:14px; color:var(--muted); font-size:15px; background:rgba(255,255,255,.86); padding:7px 10px; border:1px solid var(--line); }
    @media print { .slide { display:flex !important; break-after:page; } .nav { display:none; } }
    """
    js = """
    const slides = Array.from(document.querySelectorAll('.slide'));
    let idx = 0;
    function show(i) {
      idx = Math.max(0, Math.min(slides.length - 1, i));
      slides.forEach((s, j) => s.classList.toggle('active', j === idx));
      document.querySelector('.nav').textContent = `${idx + 1} / ${slides.length}`;
    }
    window.addEventListener('keydown', (e) => {
      if (['ArrowRight', 'PageDown', ' '].includes(e.key)) show(idx + 1);
      if (['ArrowLeft', 'PageUp', 'Backspace'].includes(e.key)) show(idx - 1);
      if (e.key === 'Home') show(0);
      if (e.key === 'End') show(slides.length - 1);
    });
    show(0);
    """
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>SUMAC Seal-to-Schema Experiments</title><style>"
        + css
        + "</style></head><body>"
        + "".join(slides)
        + "<div class='nav'></div><script>"
        + js
        + "</script></body></html>"
    )


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
    return "".join(replacements.get(ch, ch) for ch in text)


def tex_items(items: list[str]) -> str:
    return "\\begin{itemize}\n" + "\n".join(f"  \\item {item}" for item in items) + "\n\\end{itemize}"


def tex_table(rows: list[dict[str, str]], names: list[str], cols: list[tuple[str, str]], scale: str = "\\scriptsize") -> str:
    selected = [row for row in rows if row["name"] in names]
    header = "Run & " + " & ".join(label for _, label in cols) + r" \\"
    lines = [r"\begin{table}", r"\centering", scale, r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", header, r"\midrule"]
    for row in selected:
        values = " & ".join(fmt(row.get(key)) for key, _ in cols)
        lines.append(f"{tex_escape(row['name'])} & {values} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def tex_figures(items: list[tuple[str, str]], width: str = "0.31\\textwidth") -> str:
    chunks = []
    for label, rel_path in items:
        filename = Path(rel_path).name
        chunks.append(
            "\\begin{minipage}{"
            + width
            + "}\n\\centering\n"
            + f"\\includegraphics[width=\\linewidth]{{assets/{filename}}}\n"
            + f"\\captionof*{{figure}}{{\\scriptsize {tex_escape(label)}}}\n"
            + "\\end{minipage}"
        )
    return "\\begin{center}\n" + "\n\\hfill\n".join(chunks) + "\n\\end{center}"


def tex_frame(title: str, body: str) -> str:
    return f"\\begin{{frame}}{{{tex_escape(title)}}}\n{body}\n\\end{{frame}}\n"


def build_latex(rows: list[dict[str, str]], grids: dict[str, list[tuple[str, str]]]) -> str:
    metric_cols = [
        ("mae", "MAE"),
        ("ssim", "SSIM"),
        ("binary_iou", "IoU"),
        ("foreground_recall", "Recall"),
        ("skeleton_f1", "Skel F1"),
        ("skeleton_chamfer", "Chamfer"),
    ]
    frames = [
        r"""\begin{frame}
\titlepage
\end{frame}
""",
        tex_frame(
            "Research goal",
            tex_items(
                [
                    "Input: degraded Byzantine seal image plus monogram segmentation mask.",
                    "Output: clean expert-style canonical schema.",
                    "Motivation: generated schemas could expand the HIP retrieval gallery to seals without manual schemas.",
                ]
            ),
        ),
        tex_frame(
            "Dataset and split",
            tex_items(
                [
                    "332 paired q0--q2 seal/schema/mask samples.",
                    "Train/validation/test split: 265/33/34, stratified by quality label.",
                    "285 unpaired segmented seals reserved for later inference and gallery expansion.",
                    "q3 samples are excluded because masks are absent.",
                ]
            ),
        ),
        tex_frame(
            "Task formulation",
            r"\begin{center}\includegraphics[width=0.92\textwidth]{assets/task_triplet.png}\end{center}"
            + "\n"
            + tex_items(["The schema target is one-channel: foreground strokes on background.", "The model input is seal crop plus aligned binary mask."]),
        ),
        tex_frame(
            "How to read qualitative grids",
            tex_items(
                [
                    "Original evaluator grids concatenate target rows and generated rows.",
                    "Some runs originally had more rows because their evaluation batch size was smaller.",
                    "In this deck all grids are cropped to the first target/generated block.",
                    "Top row: ground truth schema. Bottom row: generated schema for the same examples.",
                ]
            ),
        ),
        tex_frame(
            "Model families tried",
            tex_items(
                [
                    "Scratch image translation: pix2pix, ResNet-U-Net, compact DDPM.",
                    "Stable Diffusion adaptation: ControlNet-style training, LoRA, IP-Adapter, pretrained scribble/lineart controls.",
                    "Topology-aware supervision: soft skeleton losses.",
                    "Pretrained visual/dense models: DINOv3, SAM, DPT.",
                ]
            ),
        ),
        tex_frame(
            "Training losses",
            r"""\[
L = \lambda_{L1}L_{L1} + \lambda_{BCE}L_{BCE} + \lambda_{Dice}L_{Dice}
    + \lambda_{clDice}L_{clDice} + \lambda_{skel}L_{skel-L1}
\]"""
            + tex_items(
                [
                    "Weighted L1: pixel reconstruction, with extra foreground stroke weight.",
                    "BCE: schema as foreground/background probability map.",
                    "Dice: overlap loss that is less sensitive to sparse foreground than raw BCE.",
                    "Skeleton clDice and skeleton L1: topology-focused supervision on differentiable soft skeletons.",
                ]
            ),
        ),
        tex_frame(
            "Evaluation metrics",
            tex_items(
                [
                    "MAE: average absolute pixel error; lower is better.",
                    "SSIM: local structural similarity; higher is better.",
                    "Binary IoU: thresholded foreground intersection over union; higher is better.",
                    "Foreground recall: fraction of target foreground recovered; can reward thick outputs.",
                    "Skeleton F1: overlap between predicted and target skeletons; higher is better.",
                    "Skeleton chamfer: bidirectional skeleton distance; lower is better.",
                ]
            ),
        ),
        tex_frame("Baseline metrics", tex_table(rows, ["pix2pix", "ResNet-U-Net", "compact DDPM"], metric_cols)),
        tex_frame("Baseline qualitative results", tex_figures(grids["baseline_grids"], "0.31\\textwidth")),
        tex_frame("Ablations and skeleton losses", tex_table(rows, ["pix2pix: mask only", "pix2pix: seal only", "pix2pix + skeleton losses"], metric_cols)),
        tex_frame("Ablation qualitative results", tex_figures(grids["ablation_grids"], "0.31\\textwidth")),
        tex_frame(
            "Stable Diffusion strategies",
            tex_items(
                [
                    "ControlNet-style conditioning with mask or seal+mask.",
                    "Guidance/control-scale changes to reduce prompt influence and strengthen conditioning.",
                    "LoRA for low-rank fine-tuning.",
                    "IP-Adapter for seal-image conditioning.",
                    "Pretrained scribble and line-art ControlNets as external priors.",
                ]
            ),
        ),
        tex_frame(
            "Stable Diffusion metrics",
            tex_table(
                rows,
                [
                    "SD15 ControlNet: seal + mask",
                    "SD15 ControlNet: longer training",
                    "SD15 ControlNet: mask only",
                    "SD15 ControlNet + LoRA + IP-Adapter",
                    "pretrained scribble ControlNet",
                    "pretrained lineart ControlNet",
                ],
                metric_cols,
                "\\tiny",
            ),
        ),
        tex_frame("Stable Diffusion qualitative results", tex_figures(grids["sd_grids"][:3], "0.31\\textwidth")),
        tex_frame("Stable Diffusion qualitative results continued", tex_figures(grids["sd_grids"][3:], "0.31\\textwidth")),
        tex_frame(
            "Stable Diffusion interpretation",
            tex_items(
                [
                    "Outputs can look smoother and more image-like than scratch baselines.",
                    "However, the models hallucinate schema-like strokes and do not reliably preserve monogram structure.",
                    "This explains why qualitative image quality and structural metrics disagree.",
                ]
            ),
        ),
        tex_frame(
            "Pretrained visual/dense metrics",
            tex_table(rows, ["DINOv3-B", "DINOv3-H+", "SAM encoder", "DPT-large"], metric_cols, "\\scriptsize"),
        ),
        tex_frame("Pretrained visual/dense qualitative results", tex_figures(grids["pretrained_grids"], "0.24\\textwidth")),
        tex_frame("Direct qualitative comparison", tex_figures(grids["comparison_grids"], "0.19\\textwidth")),
        tex_frame(
            "Current interpretation",
            tex_items(
                [
                    "Training from scratch is too weak for 332 paired samples.",
                    "Stable Diffusion brings a strong image prior, but it is not aligned with sparse schema topology.",
                    "Skeleton losses help because the target is graph structure, not stroke thickness.",
                    "DINOv3 and DPT are the strongest current directions.",
                    "Metrics are imperfect: IoU can favor thick foreground, while visual correctness depends on topology.",
                ]
            ),
        ),
        tex_frame(
            "Next decisions before SUMAC",
            tex_items(
                [
                    "Choose the main architecture: DINO-H+ or DPT-large.",
                    "Select checkpoints using skeleton/structure metrics, not only binary IoU.",
                    "Generate schemas for the 285 unpaired seals only after paired-test behavior is stable.",
                    "Run retrieval-gallery evaluation as downstream validation.",
                ]
            ),
        ),
        tex_frame(
            "Questions for supervisors",
            tex_items(
                [
                    "Should visual structural correctness be prioritized over binary IoU?",
                    "Does DINO-H+ or DPT better support the SUMAC paper story?",
                    "Should downstream retrieval performance become the main success criterion?",
                    "Are there domain constraints that should be encoded into loss or post-processing?",
                ]
            ),
        ),
    ]
    return r"""\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usecolortheme{seahorse}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{caption}
\title{SUMAC Seal-to-Schema Experiments}
\subtitle{Generating canonical Byzantine monogram schemas from degraded seal photographs}
\author{Saranga Mahantasingha}
\date{\today}
\begin{document}
""" + "\n".join(frames) + "\n\\end{document}\n"


def write_latex_project(rows: list[dict[str, str]], grids: dict[str, list[tuple[str, str]]]) -> None:
    if LATEX_OUT.exists():
        shutil.rmtree(LATEX_OUT)
    (LATEX_OUT / "assets").mkdir(parents=True, exist_ok=True)
    for asset in ASSETS.glob("*.png"):
        shutil.copy2(asset, LATEX_OUT / "assets" / asset.name)
    (LATEX_OUT / "main.tex").write_text(build_latex(rows, grids), encoding="utf-8")
    (LATEX_OUT / "master_results.csv").write_text((OUT / "master_results.csv").read_text(), encoding="utf-8")
    (LATEX_OUT / "README.md").write_text(
        "# Overleaf upload folder\n\n"
        "Upload this folder to Overleaf and compile `main.tex` with pdfLaTeX.\n"
        "The deck uses Beamer with the metropolis theme plus local PNG assets.\n",
        encoding="utf-8",
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for old_png in ASSETS.glob("*.png"):
        old_png.unlink()
    rows = metric_rows()
    write_results(rows)
    grids = copy_grid_assets()
    triplet = make_triplet()
    (OUT / "index.html").write_text(build_html(rows, triplet, grids), encoding="utf-8")
    write_latex_project(rows, grids)
    (OUT / "README.md").write_text(
        "# SUMAC Seal-to-Schema Supervisor Presentation\n\n"
        "Open `index.html` in a browser and use left/right arrows to navigate.\n\n"
        "This detailed working deck includes model families, losses, metrics, qualitative grids, and a reproducibility appendix.\n\n"
        "Generated alongside the deck:\n"
        "- `master_results.csv`\n"
        "- `master_results.json`\n"
        "- `assets/`\n",
        encoding="utf-8",
    )
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
