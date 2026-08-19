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
        norm=False, # useless
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
    return x


class ImageOpenCLIPConvNext(nn.Module):

    def __init__(self, precision="fp32", clip_model_path=None):
        super().__init__()

        # OpenCLIP accepts either a registered pretrained tag or a local
        # checkpoint file. Prefer the explicit local path used by the
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
