from __future__ import annotations

import cv2
import numpy as np
import torch
from omegaconf import DictConfig


def _denorm01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def _otsu_threshold(arr: np.ndarray) -> np.ndarray:
    arr_u8 = (arr.clip(0, 1) * 255).astype(np.uint8)
    _, binary = cv2.threshold(arr_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary > 0


def _fixed_threshold(arr: np.ndarray, threshold: float) -> np.ndarray:
    return arr >= threshold


def _zhang_suen_iteration(image: np.ndarray, odd: bool) -> np.ndarray:
    padded = np.pad(image.astype(np.uint8), 1, mode="constant")
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]

    neighbor_count = sum(neighbors)
    transitions = sum((neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1) for i in range(8))
    candidate = (image == 1) & (neighbor_count >= 2) & (neighbor_count <= 6) & (transitions == 1)
    if odd:
        candidate &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
    else:
        candidate &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
    thinned = image.copy()
    thinned[candidate] = 0
    return thinned


def zhang_suen_skeletonize(image: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    thinned = image.astype(np.uint8).copy()
    for _ in range(max_iterations):
        previous = thinned.copy()
        thinned = _zhang_suen_iteration(thinned, odd=True)
        thinned = _zhang_suen_iteration(thinned, odd=False)
        if np.array_equal(thinned, previous):
            break
    return thinned.astype(bool)


def postprocess_array(arr: np.ndarray, cfg: DictConfig) -> np.ndarray:
    threshold_method = str(cfg.get("threshold_method", "otsu"))
    if threshold_method == "otsu":
        binary = _otsu_threshold(arr)
    elif threshold_method == "fixed":
        binary = _fixed_threshold(arr, float(cfg.get("threshold", 0.5)))
    else:
        raise ValueError(f"Unknown postprocess.threshold_method: {threshold_method}")

    closing_kernel = int(cfg.get("closing_kernel", 3))
    closing_iterations = int(cfg.get("closing_iterations", 1))
    if closing_kernel > 1 and closing_iterations > 0:
        kernel = np.ones((closing_kernel, closing_kernel), dtype=np.uint8)
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=closing_iterations) > 0

    if bool(cfg.get("skeletonize", True)):
        binary = zhang_suen_skeletonize(binary, int(cfg.get("skeleton_max_iterations", 100)))

    dilation_kernel = int(cfg.get("dilation_kernel", 3))
    dilation_iterations = int(cfg.get("dilation_iterations", 1))
    if dilation_kernel > 1 and dilation_iterations > 0:
        kernel = np.ones((dilation_kernel, dilation_kernel), dtype=np.uint8)
        binary = cv2.dilate(binary.astype(np.uint8), kernel, iterations=dilation_iterations) > 0

    return binary.astype(np.float32)


def postprocess_batch(pred: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    processed = []
    pred01 = _denorm01(pred).detach().cpu()
    for item in pred01:
        arr = item.squeeze(0).numpy()
        processed.append(torch.from_numpy(postprocess_array(arr, cfg)).unsqueeze(0))
    return (torch.stack(processed, dim=0).to(pred.device) * 2.0 - 1.0).clamp(-1, 1)
