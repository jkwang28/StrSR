import os

import torch
from torch import nn
import open_clip
from open_clip.factory import CLIP


def _visual_forward(
    model: CLIP,
    image: torch.Tensor,
    return_feats: bool = False,
    return_pooled_feats: bool = False,
):
    x, intermediates = model.visual.trunk.forward_intermediates(
        image,
        indices=None,
        norm=False,
        stop_early=False,
        intermediates_only=False,
    )
    if return_feats:
        return intermediates[1:]
    x = model.visual.trunk.forward_head(x)
    x = model.visual.head(x)
    if return_pooled_feats:
        intermediates[-1] = x
        return intermediates[1:]
    return x


class ImageOpenCLIPConvNext(nn.Module):

    def __init__(self, precision="fp32", clip_model_path=None):
        super().__init__()

        if clip_model_path is not None:
            clip_model_path = os.path.expanduser(os.fspath(clip_model_path))
            if not os.path.isfile(clip_model_path):
                raise FileNotFoundError(
                    "Local CLIP checkpoint was not found: "
                    f"{clip_model_path}. Set clip_model_path to the downloaded "
                    "open_clip_model.safetensors or open_clip_pytorch_model.bin file."
                )
            pretrained = clip_model_path
        else:
            pretrained = "laion2b_s34b_b82k_augreg_soup"

        self.model, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge",
            pretrained=pretrained,
            precision=precision,
        )

    def encode_image(self, image, return_feats=False, return_pooled_feats=False):
        return _visual_forward(
            self.model,
            image,
            return_feats,
            return_pooled_feats,
        )

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual

class CNNRefiner(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, num_blocks=4):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.body = nn.Sequential(*[
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])

        self.tail = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        residual = self.tail(feat)
        
        refined = torch.clamp(x + residual, -1.0, 1.0)
        return refined
