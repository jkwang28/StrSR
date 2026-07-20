import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Tuple, Optional
import loguru

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel
from peft import LoraConfig, get_peft_model

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from HYPIR.utils.inference import load_trainable_weights
from HYPIR.utils.captioner import IMAGE_DESCRIPTION_PROMPT


def manifest_path(output: str, is_single_file: bool) -> str:
    if is_single_file and not os.path.isdir(output):
        return f"{os.path.splitext(output)[0]}.manifest.json"
    return os.path.join(output, "inference_manifest.json")


def write_manifest(
    path: str,
    args: argparse.Namespace,
    summary: dict,
    started_at: str,
    status: str,
) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": vars(args),
        "summary": summary,
    }
    with open(path, "w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")


# Simple inference helper reusing trainer logic (zimage_val) instead of enhancer.
class ZImageValInfer:
    def __init__(
        self,
        base_model_path: str,
        weight_path: Optional[str],
        lora_modules: Optional[List[str]],
        lora_rank: int,
        model_t: int,
        coeff_t: int,
        conditioning: str = "qwen",
        qwen_model_path = None,
        device: Optional[torch.device] = None,
        weight_dtype: torch.dtype = torch.bfloat16,
    ):
        self.base_model_path = base_model_path
        self.weight_path = weight_path
        self.qwen_model_path = qwen_model_path
        self.lora_modules = lora_modules or []
        self.lora_rank = lora_rank
        self.model_t = model_t
        self.coeff_t = coeff_t
        if conditioning not in {"qwen", "txt"}:
            raise ValueError(f"Unsupported conditioning mode: {conditioning!r}")
        self.conditioning = conditioning
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = weight_dtype

        self._init_models()

    def _init_models(self):
        self._init_vae()
        self._init_generator()
        self._load_lora_weights()
        if self.conditioning == "qwen":
            self._init_qwen()
            self._init_qwen_projector()
        else:
            self._init_text_models()

    def _init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path,
            subfolder="vae",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.vae.eval().requires_grad_(False)   

    def _init_generator(self):
        self.G = ZImageTransformer2DModel.from_pretrained(
            self.base_model_path,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.G.eval()

        target_patterns = list(self.lora_modules or [])
        if target_patterns:
            lora_cfg = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_patterns,
            )
            self.G = get_peft_model(self.G, lora_cfg)
            self.G.to(self.device)
        # keep eval mode for inference
        self.G.eval()

    def _init_qwen(self):
        model_path = self.qwen_model_path
        
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=self.weight_dtype,
            device_map=self.device,
            attn_implementation="flash_attention_2"
        ).eval()
        self.qwen_model.requires_grad_(False)
        
        self.qwen_processor = AutoProcessor.from_pretrained(model_path)
        
        self.image_token_id = self.qwen_model.config.image_token_id
        self.qwen_hidden_size = self.qwen_model.config.text_config.hidden_size

    def _init_qwen_projector(self):
        base_model = self.G
        if hasattr(base_model, "get_base_model"):
            base_model = base_model.get_base_model()
        if hasattr(base_model, "module"):
            base_model = base_model.module

        qwen_dim = getattr(self, "qwen_hidden_size", 2560)
        target_dim = base_model.cap_embedder[0].weight.shape[0]
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(qwen_dim, target_dim // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(target_dim // 2, target_dim),
        ).to(self.device, dtype=self.weight_dtype)
        
        projector_file = os.path.join(self.weight_path, "projector.pth")
        state_dict = torch.load(projector_file, map_location="cpu")

        self.projector.load_state_dict(state_dict)
        self.projector.eval()

    def _init_text_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            subfolder="tokenizer",
        )
        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.text_encoder.eval().requires_grad_(False)

    def _encode_prompt(self, prompts: List[str], max_sequence_length: int = 512):
        if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("Z-Image text conditioning requires a non-empty prompt.")

        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )

        text_inputs = self.tokenizer(
            formatted_prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device).bool()

        with torch.no_grad():
            hidden_states = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[-2]

        return [hidden_states[i][attention_mask[i]].to(self.weight_dtype) for i in range(len(prompts))]

    def _load_lora_weights(self):
        if not self.weight_path:
            if self.lora_modules:
                raise ValueError(
                    "--weight_path is required when LoRA modules are configured; "
                    "refusing to run with randomly initialized LoRA weights."
                )
            return
        if os.path.isdir(self.weight_path):
            state_file = os.path.join(self.weight_path, "state_dict.pth")
        elif os.path.isfile(self.weight_path):
            state_file = self.weight_path
        else:
            raise FileNotFoundError(f"LoRA checkpoint path not found: '{self.weight_path}'")
        if not os.path.isfile(state_file):
            raise FileNotFoundError(f"LoRA checkpoint file not found: '{state_file}'")

        state_dict = torch.load(state_file, map_location="cpu")
        load_trainable_weights(self.G, state_dict, state_file)

    def _prepare_prompt_embeds(self, lq: torch.Tensor = None, prompts: List[str] = None):
        if self.conditioning == "qwen":
            if lq is None:
                raise ValueError("Z-Image Qwen conditioning requires the original low-resolution LQ image.")
            prompt_embeds = self.extract_qwen_feature(lq, prompts)
        elif self.conditioning == "txt":
            prompt_embeds = self._encode_prompt(prompts)
        else:
            raise RuntimeError(f"Unsupported conditioning mode: {self.conditioning!r}")
        # store as list to match trainer format
        self.c_txt = {"prompt_embeds": [embeds.to(self.device) for embeds in prompt_embeds]}

    def _step(self, latents, noise_pred, sigmas):
        return latents.float() - (sigmas[0] - sigmas[1]) * noise_pred.float()

    def _denoise_step(self, latents, timesteps, prompt_embeds):
        latent_model_input = latents.to(device=self.device, dtype=self.weight_dtype)
        timestep_model_input = timesteps.to(device=self.device)
        prompt_embeds_model_input = [embeds.to(device=self.device, dtype=self.weight_dtype) for embeds in prompt_embeds]

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.weight_dtype == torch.bfloat16 and self.device.type == "cuda")):
            noise_pred = self.G(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds_model_input,
            )[0]
        return noise_pred

    def extract_qwen_feature(self, lq, prompts):
        batch_size = len(prompts)
        lq_images_denorm = (lq * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        
        from PIL import Image
        
        messages_batch = []
        for i in range(batch_size):
            img_pil = Image.fromarray(lq_images_denorm[i])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_pil},
                        {"type": "text", "text": prompts[i]}, 
                    ],
                }
            ]
            messages_batch.append(messages)

        texts = [
            self.qwen_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        
        inputs = self.qwen_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.qwen_model.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state # [B, Seq_Len, Hidden]
        
        text_embeds_list = []

        for i in range(batch_size):
            input_ids = inputs.input_ids[i]
            attn_mask = inputs.attention_mask[i]

            text_mask = (attn_mask == 1) & (input_ids != self.image_token_id)

            text_tokens = last_hidden_state[i][text_mask]              # [N_txt, H_qwen]
            text_tokens = text_tokens.to(dtype=self.weight_dtype)
            proj_text_tokens = self.projector(text_tokens)
            text_embeds_list.append(proj_text_tokens)

        return text_embeds_list
    
    def forward_generator(self, z_lq: torch.Tensor):
        t_expand = torch.full((z_lq.shape[0],), self.model_t, dtype=torch.long, device=self.device)
        t_expand = (1000 - t_expand) / 1000
        sigmas = torch.tensor([self.coeff_t / 1000.0, 0], dtype=torch.float32, device=self.device)

        model_out_list = self._denoise_step(z_lq, t_expand, self.c_txt["prompt_embeds"])
        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)
        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred
        latents = self._step(z_lq, noise_pred, sigmas)
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            latents = latents / self.vae.config.scaling_factor
        latents = latents.to(dtype=self.weight_dtype).contiguous()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.weight_dtype == torch.bfloat16 and self.device.type == "cuda")):
            image = self.vae.decode(latents, return_dict=False)[0]
        return image
        

def load_image_as_tensor(path: str) -> torch.Tensor:
    """Load an RGB image and return tensor of shape [1, 3, H, W] in [0,1]."""
    img = Image.open(path).convert("RGB")
    to_tensor = transforms.ToTensor()
    t = to_tensor(img).unsqueeze(0)
    return t


def save_tensor_image(t: torch.Tensor, path: str) -> None:
    """Save tensor [1,3,H,W] in [0,1] to file."""
    t = t.clamp(0, 1)
    img = transforms.ToPILImage()(t.squeeze(0).cpu())
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    img.save(path)


def bicubic_upscale(t: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
    """Upscale tensor [1,3,H,W] with bicubic interpolation."""
    out = F.interpolate(t, scale_factor=scale_factor, mode="bicubic", align_corners=False)
    return out.clamp(0, 1)


def make_feather_mask(tile_h: int, tile_w: int, overlap: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 2D feathering mask to blend overlapping tiles smoothly."""
    if overlap <= 0:
        return torch.ones((1, 1, tile_h, tile_w), device=device, dtype=dtype)

    def _hann(n):
        if n <= 1:
            return torch.ones((n,), device=device, dtype=dtype)
        return torch.hann_window(n, periodic=False, dtype=dtype, device=device)

    wy = _hann(tile_h)
    wx = _hann(tile_w)
    mask = torch.outer(wy, wx)  # [H, W]
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return mask


def tile_coords(H: int, W: int, tile: int, overlap: int) -> Tuple[Tuple[int, int], ...]:
    stride = max(1, tile - overlap)
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    # ensure coverage of right/bottom edges
    last_y = max(0, H - tile)
    last_x = max(0, W - tile)
    if ys[-1] != last_y:
        ys.append(last_y)
    if xs[-1] != last_x:
        xs.append(last_x)
    return tuple((y, x) for y in ys for x in xs)


def infer_tiled(
    image_tensor: torch.Tensor,
    model: ZImageValInfer,
    image_tensor_lq: torch.Tensor = None,
    conditioning: str = "qwen",
    prompt: Optional[str] = None,
    tile: int = 1024,
    overlap: int = 64,
):
    """Run tiled inference on a single image tensor [1,3,H,W] in [0,1]."""
    orig_H, orig_W = image_tensor.shape[-2], image_tensor.shape[-1]
    if orig_H <= 0 or orig_W <= 0:
        raise ValueError("Invalid image size")

    # Pad once so all patches respect DiT/VAE alignment (16 = VAE 8x * patch 2x)
    pad_h = (-orig_H) % 16
    pad_w = (-orig_W) % 16
    if pad_h or pad_w:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    H, W = image_tensor.shape[-2], image_tensor.shape[-1]

    device = model.device
    weight_dtype = model.weight_dtype

    out_acc = torch.zeros((1, 3, H, W), device=device, dtype=torch.float32)
    w_acc = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)

    coords = tile_coords(H, W, tile, overlap)

    if conditioning == "qwen":
        if image_tensor_lq is None:
            raise ValueError("image_tensor_lq is required for Z-Image Qwen conditioning")
        model._prepare_prompt_embeds(lq=image_tensor_lq, prompts=[IMAGE_DESCRIPTION_PROMPT])
    elif conditioning == "txt":
        if not prompt or not prompt.strip():
            raise ValueError("Z-Image text conditioning requires a non-empty prompt.")
        model._prepare_prompt_embeds(prompts=[prompt])
    else:
        raise ValueError(f"Unsupported conditioning mode: {conditioning!r}")
    for (y, x) in coords:
        patch = image_tensor[:, :, y:y + tile, x:x + tile].to(device)
        patch_h, patch_w = patch.shape[-2], patch.shape[-1]
        mask = torch.ones((1, 1, patch_h, patch_w), device=device, dtype=torch.float32)

        if y > 0:
            mask[:, :, :overlap, :] *= torch.linspace(0, 1, overlap, device=device).view(1, 1, -1, 1)

        if y + tile < H:
            mask[:, :, -overlap:, :] *= torch.linspace(1, 0, overlap, device=device).view(1, 1, -1, 1)

        if x > 0:
            mask[:, :, :, :overlap] *= torch.linspace(0, 1, overlap, device=device).view(1, 1, 1, -1)

        if x + tile < W:
            mask[:, :, :, -overlap:] *= torch.linspace(1, 0, overlap, device=device).view(1, 1, 1, -1)

        lq = (patch * 2.0 - 1.0).to(dtype=weight_dtype, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(weight_dtype == torch.bfloat16 and device.type == "cuda")):
            z_lq = model.vae.encode(lq).latent_dist.sample()
            latents = model.forward_generator(z_lq)
            img_patch = model.decode_latents(latents).float()

        out_acc[:, :, y:y + patch_h, x:x + patch_w] += img_patch * mask
        w_acc[:, :, y:y + patch_h, x:x + patch_w] += mask

    w_acc = w_acc.clamp_min(1e-6)
    out = out_acc / w_acc
    out = out.clamp(-1, 1)
    out = (out + 1.0) / 2.0

    # Crop back to original size if we padded
    out = out[:, :, :orig_H, :orig_W]
    return out


def main():
    parser = argparse.ArgumentParser(description="Multi-resolution tiled inference for Z-Image models")
    parser.add_argument("--output", type=str, required=True, help="Output image or directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even when they already exist (default: reuse existing outputs).",
    )
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model (diffusers folder)")
    parser.add_argument("--input_lq", type=str, required=True, help="Path to LR images (will be bicubic-upscaled internally)")
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default="pretrained/Qwen3-VL-4B-Instruct",
        help="Path to the Qwen3-VL model",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="pretrained/StrSR-zimage",
        help="Checkpoint directory containing state_dict.pth and projector.pth",
    )
    parser.add_argument(
        "--lora_modules",
        type=str,
        nargs="*",
        default=["to_q", "to_k", "to_v", "to_out.0", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"],
        help="LoRA target module patterns",
    )
    parser.add_argument("--lora_rank", type=int, default=256, help="LoRA rank")
    parser.add_argument("--tile", type=int, default=1024, help="Tile size")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap in pixels for smooth blending")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp32"], default="bf16", help="Computation precision")
    parser.add_argument("--model_t", type=int, default=800, help="Model timestep T")
    parser.add_argument("--coeff_t", type=int, default=800, help="Noise schedule coefficient T")
    parser.add_argument(
        "--conditioning",
        choices=["qwen", "txt"],
        default="qwen",
        help="Conditioning source: Qwen3-VL with the input image, or the user-provided text prompt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt; required when --conditioning txt",
    )
    parser.add_argument("--bicubic-scale", type=int, default=4)

    args = parser.parse_args()

    if args.conditioning == "txt" and (args.prompt is None or not args.prompt.strip()):
        parser.error("--prompt is required and must be non-empty when --conditioning txt")
    if args.conditioning == "qwen" and args.prompt is not None:
        parser.error("--prompt can only be used when --conditioning txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if args.precision == "bf16" and device.type == "cuda" else torch.float32

    if not args.output:
        raise ValueError("output path is required")

    source_root = args.input_lq
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    is_single_file = os.path.isfile(source_root)
    if is_single_file:
        if os.path.splitext(source_root)[1].lower() not in exts:
            raise ValueError(
                f"Input file '{source_root}' does not have a supported image extension."
            )
        rel_base = os.path.dirname(source_root)
        targets = [source_root]
    else:
        if not os.path.isdir(source_root):
            raise FileNotFoundError(f"input_lq path not found: '{source_root}'")
        rel_base = source_root
        targets = []
        for root, _, files in os.walk(source_root):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    targets.append(os.path.join(root, name))

    if not targets:
        raise FileNotFoundError(
            f"No input images found under '{source_root}' "
            f"(supported extensions: {', '.join(sorted(exts))})."
        )

    started_at = datetime.now(timezone.utc).isoformat()
    run_manifest_path = manifest_path(args.output, is_single_file)
    if os.path.exists(args.output):
        if args.overwrite:
            loguru.logger.warning(
                f"Output path already exists: {args.output}. "
                "--overwrite is enabled; existing outputs will be regenerated."
            )
        else:
            loguru.logger.warning(
                f"Output path already exists: {args.output}. "
                "Existing outputs will be reused by default; use --overwrite to regenerate them."
            )
    write_manifest(
        run_manifest_path,
        args,
        {"total_inputs": len(targets), "generated": 0, "skipped": 0, "failed": 0},
        started_at,
        "running",
    )

    model = ZImageValInfer(
        base_model_path=args.base_model_path,
        weight_path=args.weight_path,
        lora_modules=args.lora_modules,
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        qwen_model_path=args.qwen_model_path,
        conditioning=args.conditioning,
        device=device,
        weight_dtype=weight_dtype,
    )

    folder_failures = defaultdict(int)
    total_generated = 0
    total_skipped = 0
    total_failures = 0

    for lq_path in sorted(targets):
        rel_path = os.path.relpath(lq_path, rel_base)
        rel_stem, input_ext = os.path.splitext(rel_path)
        input_ext = input_ext.lower()
        if is_single_file and os.path.isdir(args.output):
            out_path = os.path.join(args.output, os.path.basename(rel_stem) + input_ext)
        elif is_single_file:
            output_stem = os.path.splitext(args.output)[0]
            out_path = output_stem + input_ext
        else:
            out_path = os.path.join(args.output, rel_stem + input_ext)

        if os.path.exists(out_path) and not args.overwrite:
            total_skipped += 1
            loguru.logger.info(
                f"Reusing existing output; skipped {rel_path}. "
                "Use --overwrite to regenerate it."
            )
            continue

        try:
            img_lq = load_image_as_tensor(lq_path)
            img_hr = bicubic_upscale(img_lq, scale_factor=args.bicubic_scale)
            out_t = infer_tiled(
                img_hr,
                model,
                image_tensor_lq=img_lq if args.conditioning == "qwen" else None,
                conditioning=args.conditioning,
                prompt=args.prompt,
                tile=args.tile,
                overlap=args.overlap,
            )
            save_tensor_image(out_t, out_path)
            total_generated += 1
            loguru.logger.info(f"Saved: {out_path}")
        except torch.cuda.OutOfMemoryError as exc:
            folder = os.path.dirname(rel_path) or "."
            folder_failures[folder] += 1
            total_failures += 1
            loguru.logger.error(f"OOM on {rel_path}: {exc}")
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            folder = os.path.dirname(rel_path) or "."
            folder_failures[folder] += 1
            total_failures += 1
            loguru.logger.exception(f"Error on {rel_path}: {exc}")
            continue

    summary = {
        "total_inputs": len(targets),
        "generated": total_generated,
        "skipped": total_skipped,
        "failed": total_failures,
    }
    loguru.logger.info(
        f"Inference summary: generated={total_generated}, "
        f"skipped={total_skipped}, failed={total_failures}, total={len(targets)}"
    )
    if folder_failures:
        loguru.logger.info("Failure summary per folder:")
        for folder, cnt in sorted(folder_failures.items()):
            loguru.logger.info(f"  {folder}: {cnt} failed")
    write_manifest(
        run_manifest_path,
        args,
        summary,
        started_at,
        "failed" if total_failures else "completed",
    )
    if total_failures:
        raise RuntimeError(
            f"Inference failed for {total_failures} image(s); "
            "see the failure summary above."
        )


if __name__ == "__main__":
    main()
