from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino_decoder import ResidualConvBlock


class DPTMaskDecoder(nn.Module):
    def __init__(
        self,
        model_version: str,
        out_channels: int = 1,
        mask_channels: int = 32,
        freeze_backbone: bool = True,
        freeze_neck: bool = False,
        train_head: bool = True,
        image_size: int = 384,
    ):
        super().__init__()
        from transformers import DPTForDepthEstimation

        pretrained = DPTForDepthEstimation.from_pretrained(model_version)
        self.config = pretrained.config
        self.backbone = pretrained.backbone
        self.encoder = getattr(pretrained, "dpt", None)
        self.neck = pretrained.neck
        self.has_external_backbone = self.backbone is not None
        self.image_size = int(image_size)

        if freeze_backbone:
            if self.backbone is not None:
                self.backbone.requires_grad_(False)
            if self.encoder is not None:
                self.encoder.requires_grad_(False)
        if freeze_neck:
            self.neck.requires_grad_(False)

        fusion_channels = int(self.config.fusion_hidden_size)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(1, mask_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mask_channels),
            nn.GELU(),
            ResidualConvBlock(mask_channels, mask_channels),
        )
        self.schema_head = nn.Sequential(
            ResidualConvBlock(fusion_channels + mask_channels, fusion_channels // 2),
            nn.Conv2d(fusion_channels // 2, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh(),
        )
        if not train_head:
            self.mask_proj.requires_grad_(False)
            self.schema_head.requires_grad_(False)

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _dpt_input(self, seal_rgb: torch.Tensor) -> torch.Tensor:
        seal01 = ((seal_rgb.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
        seal01 = F.interpolate(seal01, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return (seal01 - self.imagenet_mean) / self.imagenet_std

    def _feature_maps(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        if self.has_external_backbone:
            outputs = self.backbone(pixel_values, output_hidden_states=True)
            return list(outputs.feature_maps)

        outputs = self.encoder(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if not self.config.is_hybrid:
            return [
                feature
                for idx, feature in enumerate(hidden_states[1:])
                if idx in self.config.backbone_out_indices
            ]

        backbone_hidden_states = list(outputs.intermediate_activations)
        backbone_hidden_states.extend(
            feature
            for idx, feature in enumerate(hidden_states[1:])
            if idx in self.config.backbone_out_indices[2:]
        )
        return backbone_hidden_states

    def _neck_features(self, seal_rgb: torch.Tensor) -> list[torch.Tensor]:
        pixel_values = self._dpt_input(seal_rgb)
        frozen_backbone = self.encoder is None or not any(param.requires_grad for param in self.encoder.parameters())
        if self.has_external_backbone and self.backbone is not None:
            frozen_backbone = frozen_backbone and not any(param.requires_grad for param in self.backbone.parameters())

        if frozen_backbone:
            if self.encoder is not None:
                self.encoder.eval()
            if self.backbone is not None:
                self.backbone.eval()
            with torch.no_grad():
                hidden_states = self._feature_maps(pixel_values)
        else:
            hidden_states = self._feature_maps(pixel_values)

        patch_height, patch_width = None, None
        if self.config.backbone_config is not None and self.config.is_hybrid is False:
            _, _, height, width = pixel_values.shape
            patch_size = self.config.backbone_config.patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size
        return self.neck(hidden_states, patch_height, patch_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < 4:
            raise ValueError("DPTMaskDecoder expects data.input_mode=seal_mask with RGB seal plus mask channels")
        seal_rgb = x[:, :3]
        mask = ((x[:, 3:4].clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)

        hidden_states = self._neck_features(seal_rgb)
        features = hidden_states[int(self.config.head_in_index)]
        mask_features = self.mask_proj(F.interpolate(mask, size=features.shape[-2:], mode="bilinear", align_corners=False))
        pred = self.schema_head(torch.cat([features, mask_features], dim=1))
        return F.interpolate(pred, size=x.shape[-2:], mode="bilinear", align_corners=False)


def make_dpt_decoder(cfg):
    return DPTMaskDecoder(
        model_version=str(cfg.model.model_version),
        out_channels=int(cfg.model.out_channels),
        mask_channels=int(cfg.model.mask_channels),
        freeze_backbone=bool(cfg.model.freeze_backbone),
        freeze_neck=bool(cfg.model.freeze_neck),
        train_head=bool(cfg.model.train_head),
        image_size=int(cfg.model.image_size),
    )
