from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


def weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    foreground_weight: float = 1.0,
) -> torch.Tensor:
    pixel_weights = torch.ones_like(target)
    if foreground_weight > 1:
        pixel_weights = pixel_weights + (foreground_weight - 1.0) * (target > 0).float()
    per_item = (torch.abs(pred - target) * pixel_weights).mean(dim=(1, 2, 3))
    return (per_item * weights.to(pred.device)).mean()


def mask_losses(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_prob = ((pred + 1.0) / 2.0).clamp(1e-4, 1.0 - 1e-4)
    target_prob = ((target + 1.0) / 2.0).clamp(0.0, 1.0)
    bce_per_item = F.binary_cross_entropy(pred_prob, target_prob, reduction="none").mean(dim=(1, 2, 3))

    dims = (1, 2, 3)
    intersection = (pred_prob * target_prob).sum(dim=dims)
    denom = pred_prob.sum(dim=dims) + target_prob.sum(dim=dims)
    dice_per_item = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0))

    weights = weights.to(pred.device)
    return (bce_per_item * weights).mean(), (dice_per_item * weights).mean()


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    vertical = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iterations: int) -> torch.Tensor:
    skeleton = F.relu(x - soft_open(x))
    eroded = x
    for _ in range(max(int(iterations), 1)):
        eroded = soft_erode(eroded)
        opened = soft_open(eroded)
        delta = F.relu(eroded - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0, 1)


def skeleton_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_prob = ((pred + 1.0) / 2.0).clamp(0.0, 1.0)
    target_prob = ((target + 1.0) / 2.0).clamp(0.0, 1.0)
    pred_skel = soft_skeletonize(pred_prob, iterations)
    target_skel = soft_skeletonize(target_prob, iterations)

    dims = (1, 2, 3)
    eps = 1.0
    topology_precision = ((pred_skel * target_prob).sum(dim=dims) + eps) / (pred_skel.sum(dim=dims) + eps)
    topology_sensitivity = ((target_skel * pred_prob).sum(dim=dims) + eps) / (target_skel.sum(dim=dims) + eps)
    cldice_per_item = 1.0 - (2.0 * topology_precision * topology_sensitivity) / (
        topology_precision + topology_sensitivity + 1e-6
    )

    skel_l1_per_item = torch.abs(pred_skel - target_skel).mean(dim=dims)
    weights = weights.to(pred.device)
    return (cldice_per_item * weights).mean(), (skel_l1_per_item * weights).mean()


def supervised_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    cfg: DictConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1_loss = weighted_l1(pred, target, weights, float(cfg.model.foreground_weight))
    bce_loss, dice_loss = mask_losses(pred, target, weights)
    skeleton_cldice_loss, skeleton_l1_loss = skeleton_losses(
        pred,
        target,
        weights,
        int(cfg.model.get("skeleton_iterations", 20)),
    )
    loss = (
        float(cfg.model.lambda_l1) * l1_loss
        + float(cfg.model.bce_loss_weight) * bce_loss
        + float(cfg.model.dice_loss_weight) * dice_loss
        + float(cfg.model.get("skeleton_cldice_loss_weight", 0.0)) * skeleton_cldice_loss
        + float(cfg.model.get("skeleton_l1_loss_weight", 0.0)) * skeleton_l1_loss
    )
    return loss, {
        "loss": loss,
        "l1_loss": l1_loss,
        "bce_loss": bce_loss,
        "dice_loss": dice_loss,
        "skeleton_cldice_loss": skeleton_cldice_loss,
        "skeleton_l1_loss": skeleton_l1_loss,
    }


_soft_skeletonize = soft_skeletonize
