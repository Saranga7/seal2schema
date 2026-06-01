from __future__ import annotations

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.d1 = UNetDown(in_channels, c, normalize=False)
        self.d2 = UNetDown(c, c * 2)
        self.d3 = UNetDown(c * 2, c * 4)
        self.d4 = UNetDown(c * 4, c * 8)
        self.d5 = UNetDown(c * 8, c * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 8, c * 8, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.u1 = UNetUp(c * 8, c * 8)
        self.u2 = UNetUp(c * 16, c * 4)
        self.u3 = UNetUp(c * 8, c * 2)
        self.u4 = UNetUp(c * 4, c)
        self.final = nn.Sequential(nn.ConvTranspose2d(c * 2, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        b = self.bottleneck(d5)
        u1 = self.u1(b, d4)
        u2 = self.u2(u1, d3)
        u3 = self.u3(u2, d2)
        u4 = self.u4(u3, d1)
        return self.final(u4)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        c = base_channels

        def block(in_ch: int, out_ch: int, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, c, normalize=False),
            *block(c, c * 2),
            *block(c * 2, c * 4),
            *block(c * 4, c * 8),
            nn.Conv2d(c * 8, 1, 4, padding=1),
        )

    def forward(self, seal: torch.Tensor, schema: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([seal, schema], dim=1))


def make_pix2pix(cfg):
    generator = Pix2PixGenerator(cfg.model.in_channels, cfg.model.out_channels, cfg.model.base_channels)
    discriminator = PatchDiscriminator(cfg.model.in_channels + cfg.model.out_channels, cfg.model.base_channels)
    return generator, discriminator
