import math
from typing import Any, Tuple


INFERENCE_RESIZE_MODES = ("none", "area", "short-edge")


def inference_resize_shape(
    height: int,
    width: int,
    mode: str,
    reference_size: int,
) -> Tuple[int, int]:
    """Return an aspect-ratio-preserving inference shape for the selected mode.

    ``area`` enlarges images whose pixel count is below ``reference_size ** 2``.
    ``short-edge`` enlarges images whose shorter edge is below ``reference_size``.
    Neither mode downsizes images that already meet its threshold.
    """
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    if mode not in INFERENCE_RESIZE_MODES:
        raise ValueError(f"Unsupported inference resize mode: {mode!r}")
    if mode == "none":
        return height, width
    if reference_size <= 0:
        raise ValueError("Inference reference size must be positive")

    if mode == "area":
        reference_area = reference_size * reference_size
        image_area = height * width
        if image_area >= reference_area:
            return height, width
        scale = math.sqrt(reference_area / image_area)
    else:
        if min(height, width) >= reference_size:
            return height, width
        scale = reference_size / min(height, width)

    return round(height * scale), round(width * scale)


def resize_to_shape(tensor: Any, size: Tuple[int, int]) -> Any:
    """Resize a BCHW tensor with antialiased bicubic interpolation and clamp it to [0, 1]."""
    from torch.nn import functional as F

    output = F.interpolate(
        tensor,
        size=size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return output.clamp(0, 1)
