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
    # stem, stages, norm_pre
    x, intermediates = model.visual.trunk.forward_intermediates(
        image,
        indices=None,
        norm=False,  # useless
        stop_early=False,
        intermediates_only=False,
    )
    if return_feats:
        return intermediates[1:]
    # trunk.head
    x = model.visual.trunk.forward_head(x)
    # visual.head
    x = model.visual.head(x)
    if return_pooled_feats:
        intermediates[-1] = x
        return intermediates[1:]
        # return intermediates[:]
    return x


class ImageOpenCLIPConvNext(nn.Module):

    def __init__(self, precision="fp32", clip_model_path=None):
        super().__init__()

        # OpenCLIP accepts either a registered pretrained tag or a local
        # checkpoint file.  Prefer the explicit local path used by the
        # training configs so offline training does not silently contact HF.
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
            # Backward-compatible fallback for legacy trainers that do not
            # expose a local CLIP path yet.
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
        
        # 1. 初始特征提取 (保持分辨率)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 2. 残差主体 (用于去噪和平滑伪影)
        self.body = nn.Sequential(*[
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # 3. 输出层 (映射回 RGB 空间)
        self.tail = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh() # 假设输入图像归一化在 [-1, 1]
        )

    def forward(self, x):
        # 输入 x 预期范围 [-1, 1]
        feat = self.head(x)
        feat = self.body(feat)
        residual = self.tail(feat)
        
        # Refiner 学习的是残差 (Artifacts)，所以输出是 原图 + 修正量
        # 这种 Residual Learning 结构更容易训练且能保留原图结构
        refined = torch.clamp(x + residual, -1.0, 1.0)
        return refined
