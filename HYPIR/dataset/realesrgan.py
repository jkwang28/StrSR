from typing import Dict, Optional
import math
import random
import time
import io
import os

import numpy as np
import torch
from torch.utils import data
from PIL import Image

from HYPIR.dataset.utils import augment, random_crop_arr, center_crop_arr, load_file_meta
from HYPIR.utils.degradation import circular_lowpass_kernel, random_mixed_kernels
from HYPIR.utils.common import instantiate_from_config


class RealESRGANDataset(data.Dataset):

    def __init__(
        self,
        file_meta,
        file_backend_cfg,
        out_size,
        crop_type,
        use_hflip,
        use_rot,
        blur_kernel_size,
        kernel_list,
        kernel_prob,
        blur_sigma,
        betag_range,
        betap_range,
        sinc_prob,
        blur_kernel_size2,
        kernel_list2,
        kernel_prob2,
        blur_sigma2,
        betag_range2,
        betap_range2,
        sinc_prob2,
        final_sinc_prob,
        p_empty_prompt,
        return_file_name=False,
    ):
        super(RealESRGANDataset, self).__init__()
        self.file_meta = file_meta
        self.image_files = load_file_meta(file_meta)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]

        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.sinc_prob = sinc_prob

        self.blur_kernel_size2 = blur_kernel_size2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2

        self.final_sinc_prob = final_sinc_prob

        self.use_hflip = use_hflip
        self.use_rot = use_rot

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

        self.p_empty_prompt = p_empty_prompt
        self.return_file_name = return_file_name

        Image.MAX_IMAGE_PIXELS = 268435456

    def load_gt_image(self, image_path: str, max_retry: int = 5) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            try:
                image_bytes = self.file_backend.get(image_path)
            except:
                return None
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
            return None

        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            if not (image.height == self.out_size and image.width == self.out_size):
                print(f"Warning: image size {image.width}x{image.height}, with no crop.")
            image = np.array(image)
        return image

    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_gt = None
        while img_gt is None:
            image_file = self.image_files[index]
            gt_path = image_file["image_path"]
            prompt = image_file["prompt"]
            if "lq_path" in image_file:
                lq_path = image_file["lq_path"]
                img_lq = self.load_gt_image(lq_path)
                if img_lq is None:
                    print(f"failed to load {lq_path}")
            else:
                img_lq = None
            img_gt = self.load_gt_image(gt_path)
            if img_gt is None:
                print(f"failed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)

        img_hq = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        if img_lq is not None:
            img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        if np.random.uniform() < self.p_empty_prompt:
            prompt = ""

        img_hq = augment(img_hq, self.use_hflip, self.use_rot)

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        img_hq = torch.from_numpy(img_hq[..., ::-1].transpose(2, 0, 1).copy()).float()
        if img_lq is not None:
            img_lq = torch.from_numpy(img_lq[..., ::-1].transpose(2, 0, 1).copy()).float()
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        data = {
            "hq": img_hq,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "txt": prompt,
        }
        if img_lq is not None:
            data["lq"] = img_lq
        if self.return_file_name:
            prefix = None
            try:
                prefix = self.file_meta.get("image_path_prefix", None)
            except Exception:
                prefix = None
            if prefix:
                try:
                    rel = os.path.relpath(gt_path, start=prefix)
                    data["filename"] = rel
                except Exception:
                    data["filename"] = os.path.basename(gt_path)
            else:
                data["filename"] = os.path.basename(gt_path)
        return data

    def __len__(self) -> int:
        return len(self.image_files)
