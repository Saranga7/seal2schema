from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino_decoder import MaskPyramid, ResidualConvBlock


class SAMMaskDecoder(nn.Module):
    def __init__(
        self,
        model_version: str,
        out_channels: int = 1,
        decoder_channels: int = 256,
        mask_channels: int = 32,
        freeze_encoder: bool = True,
        image_size: int = 1024,
    ):
        super().__init__()
        from transformers import SamVisionModel

        self.encoder = SamVisionModel.from_pretrained(model_version)
        self.freeze_encoder = bool(freeze_encoder)
        self.image_size = int(image_size)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)

        config = self.encoder.config
        hidden_size = int(getattr(config, "output_channels", getattr(config, "hidden_size", 256)))

        self.register_buffer("sam_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1) / 255.0, persistent=False)
        self.register_buffer("sam_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1) / 255.0, persistent=False)

        self.mask_pyramid = MaskPyramid(mask_channels)
        self.token_proj = nn.Sequential(
            nn.Conv2d(hidden_size, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.fuse_features = ResidualConvBlock(decoder_channels + mask_channels, decoder_channels)
        self.up112 = ResidualConvBlock(decoder_channels + mask_channels, decoder_channels // 2)
        self.up224 = ResidualConvBlock(decoder_channels // 2 + mask_channels, decoder_channels // 4)
        self.refine = ResidualConvBlock(decoder_channels // 4, decoder_channels // 8)
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels // 8, decoder_channels // 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 8, out_channels, 1),
            nn.Tanh(),
        )

    def _sam_input(self, seal_rgb: torch.Tensor) -> torch.Tensor:
        seal01 = ((seal_rgb.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
        seal01 = F.interpolate(seal01, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return (seal01 - self.sam_mean) / self.sam_std

    def _image_features(self, seal_rgb: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                out = self.encoder(pixel_values=self._sam_input(seal_rgb))
        else:
            out = self.encoder(pixel_values=self._sam_input(seal_rgb))
        features = out.image_embeds if out.image_embeds is not None else out.last_hidden_state
        if features is None:
            raise ValueError("SAM vision encoder returned neither image_embeds nor last_hidden_state")
        expected_channels = int(getattr(self.encoder.config, "output_channels", features.shape[1]))
        if features.ndim != 4:
            raise ValueError(f"Expected 4D SAM image features, got shape {tuple(features.shape)}")
        if features.shape[1] != expected_channels and features.shape[-1] == expected_channels:
            features = features.permute(0, 3, 1, 2).contiguous()
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < 4:
            raise ValueError("SAMMaskDecoder expects data.input_mode=seal_mask with RGB seal plus mask channels")
        seal_rgb = x[:, :3]
        mask = ((x[:, 3:4].clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)

        mask_features = self.mask_pyramid(mask)
        features = self.token_proj(self._image_features(seal_rgb))
        mask_at_features = F.interpolate(mask_features[56], size=features.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse_features(torch.cat([features, mask_at_features], dim=1))

        y = F.interpolate(y, size=mask_features[112].shape[-2:], mode="bilinear", align_corners=False)
        y = self.up112(torch.cat([y, mask_features[112]], dim=1))
        y = F.interpolate(y, size=mask_features[224].shape[-2:], mode="bilinear", align_corners=False)
        y = self.up224(torch.cat([y, mask_features[224]], dim=1))
        y = self.refine(y)
        return self.head(y)


def make_sam_decoder(cfg):
    return SAMMaskDecoder(
        model_version=str(cfg.model.model_version),
        out_channels=int(cfg.model.out_channels),
        decoder_channels=int(cfg.model.decoder_channels),
        mask_channels=int(cfg.model.mask_channels),
        freeze_encoder=bool(cfg.model.freeze_encoder),
        image_size=int(cfg.model.image_size),
    )
