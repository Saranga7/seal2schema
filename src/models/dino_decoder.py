from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.proj(x))


class MaskPyramid(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.stem = ResidualConvBlock(1, channels)
        self.down112 = ResidualConvBlock(channels, channels)
        self.down56 = ResidualConvBlock(channels, channels)
        self.down28 = ResidualConvBlock(channels, channels)
        self.down14 = ResidualConvBlock(channels, channels)

    def forward(self, mask: torch.Tensor) -> dict[int, torch.Tensor]:
        m224 = self.stem(mask)
        m112 = self.down112(F.interpolate(m224, scale_factor=0.5, mode="bilinear", align_corners=False))
        m56 = self.down56(F.interpolate(m112, scale_factor=0.5, mode="bilinear", align_corners=False))
        m28 = self.down28(F.interpolate(m56, scale_factor=0.5, mode="bilinear", align_corners=False))
        m14 = self.down14(F.interpolate(m28, scale_factor=0.5, mode="bilinear", align_corners=False))
        return {224: m224, 112: m112, 56: m56, 28: m28, 14: m14}


class FPNUpBlock(nn.Module):
    def __init__(self, in_channels: int, mask_channels: int, out_channels: int):
        super().__init__()
        self.block = ResidualConvBlock(in_channels + mask_channels, out_channels)

    def forward(self, x: torch.Tensor, mask_skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=mask_skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, mask_skip], dim=1))


class DINOv3MaskDecoder(nn.Module):
    def __init__(
        self,
        model_version: str,
        out_channels: int = 1,
        decoder_channels: int = 256,
        mask_channels: int = 32,
        freeze_encoder: bool = True,
        unfreeze_last_layer: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_version)
        self.freeze_encoder = bool(freeze_encoder)
        hidden_size = int(self.encoder.config.hidden_size)
        self.patch_size = int(getattr(self.encoder.config, "patch_size", 16))

        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
        if unfreeze_last_layer:
            layers = getattr(self.encoder, "layer", None)
            if layers is None:
                layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
            if layers is None:
                raise ValueError("Could not find DINOv3 transformer layers for unfreeze_last_layer=true")
            for param in layers[-1].parameters():
                param.requires_grad = True
        self.encoder_has_trainable_params = any(param.requires_grad for param in self.encoder.parameters())

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.mask_pyramid = MaskPyramid(mask_channels)
        self.token_proj = nn.Sequential(
            nn.Conv2d(hidden_size, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.fuse14 = ResidualConvBlock(decoder_channels + mask_channels, decoder_channels)
        self.up28 = FPNUpBlock(decoder_channels, mask_channels, decoder_channels // 2)
        self.up56 = FPNUpBlock(decoder_channels // 2, mask_channels, decoder_channels // 4)
        self.up112 = FPNUpBlock(decoder_channels // 4, mask_channels, decoder_channels // 8)
        self.up224 = FPNUpBlock(decoder_channels // 8, mask_channels, decoder_channels // 16)
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels // 16, decoder_channels // 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 16, out_channels, 1),
            nn.Tanh(),
        )

    def _dino_input(self, seal_rgb: torch.Tensor) -> torch.Tensor:
        seal01 = ((seal_rgb.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
        return (seal01 - self.imagenet_mean) / self.imagenet_std

    def _patch_grid(self, seal_rgb: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder and not self.encoder_has_trainable_params:
            self.encoder.eval()
            with torch.no_grad():
                out = self.encoder(self._dino_input(seal_rgb))
        else:
            out = self.encoder(self._dino_input(seal_rgb))
        h = out.last_hidden_state
        num_register_tokens = int(getattr(self.encoder.config, "num_register_tokens", 0))
        patch_tokens = h[:, 1 + num_register_tokens :, :]
        grid_h = seal_rgb.shape[-2] // self.patch_size
        grid_w = seal_rgb.shape[-1] // self.patch_size
        expected = grid_h * grid_w
        patch_tokens = patch_tokens[:, :expected, :]
        return patch_tokens.transpose(1, 2).reshape(seal_rgb.size(0), -1, grid_h, grid_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < 4:
            raise ValueError("DINOv3MaskDecoder expects data.input_mode=seal_mask with RGB seal plus mask channels")
        seal_rgb = x[:, :3]
        mask = ((x[:, 3:4].clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)

        mask_features = self.mask_pyramid(mask)
        tokens = self.token_proj(self._patch_grid(seal_rgb))
        y = self.fuse14(torch.cat([tokens, mask_features[14]], dim=1))
        y = self.up28(y, mask_features[28])
        y = self.up56(y, mask_features[56])
        y = self.up112(y, mask_features[112])
        y = self.up224(y, mask_features[224])
        return self.head(y)


def make_dino_decoder(cfg):
    return DINOv3MaskDecoder(
        model_version=str(cfg.model.model_version),
        out_channels=int(cfg.model.out_channels),
        decoder_channels=int(cfg.model.decoder_channels),
        mask_channels=int(cfg.model.mask_channels),
        freeze_encoder=bool(cfg.model.freeze_encoder),
        unfreeze_last_layer=bool(cfg.model.unfreeze_last_layer),
    )
