from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        c = base_channels
        time_dim = c * 4
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(c), nn.Linear(c, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.in_conv = nn.Conv2d(in_channels, c, 3, padding=1)
        self.down1 = ResBlock(c, c, time_dim)
        self.down2 = ResBlock(c, c * 2, time_dim)
        self.down3 = ResBlock(c * 2, c * 4, time_dim)
        self.mid = ResBlock(c * 4, c * 4, time_dim)
        self.up3 = ResBlock(c * 8, c * 2, time_dim)
        self.up2 = ResBlock(c * 4, c, time_dim)
        self.up1 = ResBlock(c * 2, c, time_dim)
        self.out = nn.Conv2d(c, out_channels, 1)

    def forward(self, noisy_schema: torch.Tensor, seal: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noisy_schema, seal], dim=1)
        emb = self.time_mlp(t)
        x = self.in_conv(x)
        d1 = self.down1(x, emb)
        d2 = self.down2(F.avg_pool2d(d1, 2), emb)
        d3 = self.down3(F.avg_pool2d(d2, 2), emb)
        mid = self.mid(F.avg_pool2d(d3, 2), emb)
        u3 = F.interpolate(mid, size=d3.shape[-2:], mode="nearest")
        u3 = self.up3(torch.cat([u3, d3], dim=1), emb)
        u2 = F.interpolate(u3, size=d2.shape[-2:], mode="nearest")
        u2 = self.up2(torch.cat([u2, d2], dim=1), emb)
        u1 = F.interpolate(u2, size=d1.shape[-2:], mode="nearest")
        u1 = self.up1(torch.cat([u1, d1], dim=1), emb)
        return self.out(u1)


class DDPM:
    def __init__(self, cfg, device: torch.device):
        self.timesteps = int(cfg.model.timesteps)
        betas = torch.linspace(float(cfg.model.beta_start), float(cfg.model.beta_end), self.timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.sqrt_alphas_cumprod[t][:, None, None, None] * x0 + self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None, None
        ] * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, seal: torch.Tensor) -> torch.Tensor:
        x = torch.randn(seal.size(0), 1, seal.size(2), seal.size(3), device=seal.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((seal.size(0),), i, device=seal.device, dtype=torch.long)
            pred_noise = model(x, seal, t)
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]
            beta = self.betas[i]
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise)
            if i > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x.clamp(-1, 1)


def make_diffusion(cfg, device: torch.device) -> tuple[ConditionalUNet, DDPM]:
    return ConditionalUNet(cfg.model.in_channels + cfg.model.out_channels, cfg.model.out_channels, cfg.model.base_channels), DDPM(
        cfg, device
    )
