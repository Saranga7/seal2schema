from __future__ import annotations

import torch
import torch.nn.functional as F


def denorm01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def binary_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_b = denorm01(pred) > threshold
    target_b = denorm01(target) > threshold
    inter = (pred_b & target_b).float().sum(dim=(1, 2, 3))
    union = (pred_b | target_b).float().sum(dim=(1, 2, 3)).clamp_min(1)
    return (inter / union).mean().item()


def simple_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    x = denorm01(pred)
    y = denorm01(target)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, 7, 1, 3)
    mu_y = F.avg_pool2d(y, 7, 1, 3)
    sigma_x = F.avg_pool2d(x * x, 7, 1, 3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 7, 1, 3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 7, 1, 3) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean().item()


def _binary_erode(x: torch.Tensor) -> torch.Tensor:
    return (F.max_pool2d((~x).float(), kernel_size=3, stride=1, padding=1) == 0)


def _binary_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x.float(), kernel_size=3, stride=1, padding=1) > 0


def _binary_open(x: torch.Tensor) -> torch.Tensor:
    return _binary_dilate(_binary_erode(x))


def _binary_skeletonize(x: torch.Tensor, max_iter: int = 64) -> torch.Tensor:
    skeleton = torch.zeros_like(x, dtype=torch.bool)
    current = x.bool()
    for _ in range(max_iter):
        opened = _binary_open(current)
        skeleton = skeleton | (current & ~opened)
        eroded = _binary_erode(current)
        if not eroded.any():
            break
        current = eroded
    return skeleton


def _distance_to_foreground(query: torch.Tensor, reference: torch.Tensor, max_radius: int = 12) -> torch.Tensor:
    if not query.any():
        return torch.zeros(query.shape[0], device=query.device)
    if not reference.any():
        return torch.full((query.shape[0],), float(max_radius), device=query.device)
    distances = torch.full(query.shape, float(max_radius), device=query.device)
    unresolved = query.clone()
    for radius in range(max_radius + 1):
        if radius == 0:
            within = reference
        else:
            within = F.max_pool2d(reference.float(), kernel_size=2 * radius + 1, stride=1, padding=radius) > 0
        hit = unresolved & within
        distances = torch.where(hit, torch.full_like(distances, float(radius)), distances)
        unresolved = unresolved & ~hit
        if not unresolved.any():
            break
    denom = query.float().sum(dim=(1, 2, 3)).clamp_min(1)
    return (distances * query.float()).sum(dim=(1, 2, 3)) / denom


def skeleton_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> list[dict[str, float]]:
    pred_b = denorm01(pred) > threshold
    target_b = denorm01(target) > threshold
    pred_skel = _binary_skeletonize(pred_b)
    target_skel = _binary_skeletonize(target_b)

    dims = (1, 2, 3)
    inter = (pred_skel & target_skel).float().sum(dim=dims)
    pred_count = pred_skel.float().sum(dim=dims)
    target_count = target_skel.float().sum(dim=dims)
    union = (pred_skel | target_skel).float().sum(dim=dims)
    precision = inter / pred_count.clamp_min(1)
    recall = inter / target_count.clamp_min(1)
    f1 = (2.0 * precision * recall) / (precision + recall).clamp_min(1e-6)
    iou = inter / union.clamp_min(1)

    pred_to_target = _distance_to_foreground(pred_skel, target_skel)
    target_to_pred = _distance_to_foreground(target_skel, pred_skel)
    chamfer = 0.5 * (pred_to_target + target_to_pred)
    rows = []
    for idx in range(pred.size(0)):
        rows.append(
            {
                "skeleton_precision": float(precision[idx].item()),
                "skeleton_recall": float(recall[idx].item()),
                "skeleton_f1": float(f1[idx].item()),
                "skeleton_iou": float(iou[idx].item()),
                "skeleton_chamfer": float(chamfer[idx].item()),
            }
        )
    return rows


def reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    rows = reconstruction_metrics_per_item(pred, target, threshold)
    return {key: float(torch.tensor([row[key] for row in rows]).mean()) for key in rows[0]}


def reconstruction_metrics_per_item(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> list[dict[str, float]]:
    pred01 = denorm01(pred)
    target01 = denorm01(target)
    pred_b = denorm01(pred) > threshold
    target_b = denorm01(target) > threshold
    foreground = target_b.float().sum(dim=(1, 2, 3)).clamp_min(1)
    foreground_recall = ((pred_b & target_b).float().sum(dim=(1, 2, 3)) / foreground)
    inter = (pred_b & target_b).float().sum(dim=(1, 2, 3))
    union = (pred_b | target_b).float().sum(dim=(1, 2, 3)).clamp_min(1)
    iou = inter / union
    mae = torch.mean(torch.abs(pred01 - target01), dim=(1, 2, 3))
    rows = []
    skel_rows = skeleton_metrics(pred, target, threshold)
    for idx in range(pred.size(0)):
        rows.append(
            {
                "mae": float(mae[idx].item()),
                "ssim": simple_ssim(pred[idx : idx + 1], target[idx : idx + 1]),
                "binary_iou": float(iou[idx].item()),
                "foreground_recall": float(foreground_recall[idx].item()),
                **skel_rows[idx],
            }
        )
    return rows
