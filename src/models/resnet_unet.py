from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


def _make_encoder(name: str, pretrained: bool):
    if name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
        channels = [64, 64, 128, 256, 512]
    elif name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        channels = [64, 256, 512, 1024, 2048]
    else:
        raise ValueError(f"Unsupported encoder: {name}")
    return model, channels


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if in_channels == conv.in_channels:
        return conv
    adapted = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        adapted.weight[:, : conv.in_channels] = conv.weight
        if in_channels > conv.in_channels:
            extra = conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels - conv.in_channels, 1, 1)
            adapted.weight[:, conv.in_channels :] = extra
        else:
            adapted.weight = nn.Parameter(conv.weight[:, :in_channels].clone())
        if conv.bias is not None:
            adapted.bias.copy_(conv.bias)
    return adapted


class ResNetUNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder: str = "resnet34",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        decoder_channels: int = 256,
    ):
        super().__init__()
        backbone, ch = _make_encoder(encoder, pretrained)
        backbone.conv1 = _adapt_first_conv(backbone.conv1, in_channels)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if freeze_encoder:
            for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in module.parameters():
                    param.requires_grad = False

        d = decoder_channels
        self.center = ConvBlock(ch[4], d)
        self.up4 = UpBlock(d, ch[3], d // 2)
        self.up3 = UpBlock(d // 2, ch[2], d // 4)
        self.up2 = UpBlock(d // 4, ch[1], d // 8)
        self.up1 = UpBlock(d // 8, ch[0], d // 16)
        self.final = nn.Sequential(
            nn.Conv2d(d // 16, d // 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 16, out_channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)      # 112x112 for 224 input
        x1 = self.layer1(self.maxpool(x0))  # 56x56
        x2 = self.layer2(x1)   # 28x28
        x3 = self.layer3(x2)   # 14x14
        x4 = self.layer4(x3)   # 7x7
        y = self.center(x4)
        y = self.up4(y, x3)
        y = self.up3(y, x2)
        y = self.up2(y, x1)
        y = self.up1(y, x0)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.final(y)


def make_resnet_unet(cfg):
    return ResNetUNetGenerator(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        encoder=cfg.model.encoder,
        pretrained=bool(cfg.model.pretrained),
        freeze_encoder=bool(cfg.model.freeze_encoder),
        decoder_channels=int(cfg.model.base_channels) * 4,
    )
