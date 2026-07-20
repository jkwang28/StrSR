import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Union
import loguru

from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2TokenizerFast, Qwen3ForCausalLM, Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
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


# Simple inference helper reusing Flux trainer logic instead of enhancer.
class FluxValInfer:
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
        if self.conditioning == "txt":
            self._init_text_models()
        else:
            self._init_qwen()
            self._init_qwen_projector()

    def _init_vae(self):
        self.vae = AutoencoderKLFlux2.from_pretrained(
            self.base_model_path,
            subfolder="vae",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.vae.eval().requires_grad_(False)   

    def _init_generator(self):
        self.G = Flux2Transformer2DModel.from_pretrained(
            self.base_model_path,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.G.eval()

        target_patterns = list(self.lora_modules or [])
        if target_patterns:
            resolved_targets = self._resolve_lora_targets(self.G, target_patterns)
            if not resolved_targets:
                raise ValueError(f"No valid LoRA target modules resolved from patterns: {target_patterns}")
            lora_cfg = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_rank,
                init_lora_weights="gaussian",
                target_modules=resolved_targets,
            )
            self.G = get_peft_model(self.G, lora_cfg)
            self.G.to(self.device)
        # keep eval mode for inference
        self.G.eval()

    def _resolve_lora_targets(self, model: torch.nn.Module, target_patterns: List[str]) -> List[str]:
        """Resolve LoRA targets to concrete supported module names (Linear/Conv), avoiding ModuleList wrappers."""
        candidate_types = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
        matched = []
        for module_name, module in model.named_modules():
            if not isinstance(module, candidate_types):
                continue
            if any(module_name.endswith(pattern) or pattern in module_name for pattern in target_patterns):
                matched.append(module_name)
        return sorted(set(matched))

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

    def _init_text_models(self):
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            self.base_model_path,
            subfolder="tokenizer",
        )
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            self.base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.text_encoder.eval().requires_grad_(False)

    def _init_qwen_projector(self):
        base_model = self.G
        if hasattr(base_model, "get_base_model"):
            base_model = base_model.get_base_model()
        target_dim = getattr(base_model.config, "joint_attention_dim", None)
        if target_dim is None:
            raise ValueError("Cannot infer Flux.2 text embedding dimension from transformer config.")
        qwen_dim = getattr(self, "qwen_hidden_size", 3584)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(qwen_dim, target_dim // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(target_dim // 2, target_dim),
        ).to(self.device, dtype=self.weight_dtype)
        
        projector_file = os.path.join(self.weight_path, "projector.pth")
        state_dict = torch.load(projector_file, map_location="cpu")

        self.projector.load_state_dict(state_dict)
        self.projector.eval()

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
                raise ValueError("FLUX Qwen conditioning requires the original low-resolution LQ image.")
            # Qwen must see the original LR image.  The caller separately sends the
            # bicubic-upscaled image through the VAE/DiT path.
            raw_lq = lq.float().to(self.device).clamp(0, 1)
            text_embeds, text_ids = self.extract_qwen_feature(raw_lq, prompts)
        elif self.conditioning == "txt":
            if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
                raise ValueError("FLUX text conditioning requires a non-empty prompt.")
            layer_ids = tuple(getattr(self, "flux_text_encoder_layers", (9, 18, 27)))
            text_embeds = self._get_qwen3_prompt_embeds(prompts, hidden_states_layers=layer_ids)
            text_ids = self._prepare_text_ids(text_embeds)
        else:
            raise RuntimeError(f"Unsupported conditioning mode: {self.conditioning!r}")
        self.c_txt = {"text_embeds": text_embeds, "text_ids": text_ids}

    def _step(self, latents, noise_pred, sigmas):
        return latents.float() - (sigmas[0] - sigmas[1]) * noise_pred.float()

    def _get_qwen3_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 512,
        hidden_states_layers: Tuple[int, ...] = (9, 18, 27),
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        all_input_ids, all_attention_masks = [], []
        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.device)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=self.weight_dtype, device=self.device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        return out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor):
        batch, length, _ = x.shape
        ids = torch.zeros(batch, length, 4, dtype=x.dtype, device=x.device)
        ids[:, :, 3] = torch.arange(length, dtype=x.dtype, device=x.device)
        return ids

    @staticmethod
    def _patchify_latents(latents):
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify_latents(latents):
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    @staticmethod
    def _pack_flux2_latents(latents):
        b, c, h, w = latents.shape
        return latents.reshape(b, c, h * w).permute(0, 2, 1)

    @staticmethod
    def _unpack_flux2_latents(latents, height, width):
        b, _, c = latents.shape
        return latents.permute(0, 2, 1).reshape(b, c, height, width)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor):
        b, _, h, w = latents.shape
        hs = torch.arange(h, device=latents.device, dtype=latents.dtype)
        ws = torch.arange(w, device=latents.device, dtype=latents.dtype)
        grid_h, grid_w = torch.meshgrid(hs, ws, indexing="ij")
        out = torch.zeros(b, h * w, 4, device=latents.device, dtype=latents.dtype)
        out[:, :, 1] = grid_h.reshape(-1)
        out[:, :, 2] = grid_w.reshape(-1)
        return out

    def _normalize_flux2_latents(self, latents: torch.Tensor):
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        return (latents - bn_mean) / bn_std

    def _denormalize_flux2_latents(self, latents: torch.Tensor):
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        return latents * bn_std + bn_mean
    
    def extract_qwen_feature(self, lq, prompts):
        batch_size = len(prompts)
        lq_images = (lq * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        from PIL import Image

        messages_batch = []
        for i in range(batch_size):
            messages_batch.append(
                [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(lq_images[i])}, {"type": "text", "text": prompts[i]}]}]
            )

        texts = [self.qwen_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.qwen_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            last_hidden_state = self.qwen_model.model(**inputs, output_hidden_states=True).last_hidden_state

        max_tokens = int(getattr(self, "flux_max_text_tokens", 512))
        text_embeds_list = []
        for i in range(batch_size):
            input_ids = inputs.input_ids[i]
            attn_mask = inputs.attention_mask[i]
            text_mask = (attn_mask == 1) & (input_ids != self.image_token_id)
            text_tokens = last_hidden_state[i][text_mask].to(dtype=self.weight_dtype)
            if text_tokens.shape[0] > max_tokens:
                text_tokens = text_tokens[:max_tokens]
            text_embeds_list.append(self.projector(text_tokens))

        text_embeds = torch.nn.utils.rnn.pad_sequence(text_embeds_list, batch_first=True)
        return text_embeds, self._prepare_text_ids(text_embeds)
    
    def _denoise_step(self, packed_latents, timesteps, height, width):
        text_embeds = self.c_txt["text_embeds"].to(device=self.device, dtype=self.weight_dtype)
        txt_ids = self.c_txt["text_ids"].to(device=self.device, dtype=self.weight_dtype)
        latent_model_input = packed_latents.to(device=self.device, dtype=self.weight_dtype)
        timestep_input = timesteps.to(device=self.device, dtype=self.weight_dtype) / 1000
        img_ids = self._prepare_latent_ids(self._unpack_flux2_latents(latent_model_input, height, width)).to(
            device=self.device, dtype=self.weight_dtype
        )
        return self.G(
            hidden_states=latent_model_input,
            timestep=timestep_input,
            guidance=None,
            encoder_hidden_states=text_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    
    def forward_generator(self, z_lq: torch.Tensor):
        timesteps = torch.full((z_lq.shape[0],), self.model_t, dtype=torch.long, device=self.device)
        sigmas = torch.tensor([self.coeff_t / 1000.0, 0], dtype=torch.float32, device=self.device)

        latent_model_input = z_lq
        _, _, h, w = latent_model_input.shape
        packed_latents = self._pack_flux2_latents(latent_model_input)
        noise_pred = self._denoise_step(packed_latents, timesteps, h, w)
        packed_denoised = self._step(packed_latents, noise_pred, sigmas)
        latents = self._unpack_flux2_latents(packed_denoised, h, w)

        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self._denormalize_flux2_latents(latents)
        latents = self._unpatchify_latents(latents)
        return self.vae.decode(latents.to(dtype=self.weight_dtype), return_dict=False)[0]
        

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
    model: FluxValInfer,
    image_tensor_lq: torch.Tensor = None,
    conditioning: str = "qwen",
    prompt: Optional[str] = None,
    tile: int = 1024,
    overlap: int = 64,
):
    """Run tiled inference with HR-sized VAE input and the selected conditioning."""
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
            raise ValueError("image_tensor_lq is required for FLUX Qwen conditioning")
        model._prepare_prompt_embeds(lq=image_tensor_lq, prompts=[IMAGE_DESCRIPTION_PROMPT])
    elif conditioning == "txt":
        if not prompt or not prompt.strip():
            raise ValueError("FLUX text conditioning requires a non-empty prompt.")
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
            z_lq = model._normalize_flux2_latents(model._patchify_latents(z_lq))
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
    parser = argparse.ArgumentParser(description="Multi-resolution tiled inference for Flux.2 models")
    parser.add_argument("--output", type=str, required=True, help="Output image or directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even when they already exist (default: reuse existing outputs).",
    )
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model (diffusers folder)")
    parser.add_argument(
        "--input_lq",
        type=str,
        required=True,
        help="Path to original LR images (upscaled only for VAE/DiT; raw LR is used for Qwen)",
    )
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default="pretrained/Qwen3-VL-4B-Instruct",
        help="Path to the Qwen3-VL model",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="pretrained/StrSR-flux",
        help="Checkpoint directory containing state_dict.pth and projector.pth",
    )
    parser.add_argument(
        "--lora_modules",
        type=str,
        nargs="*",
        default=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_k_proj",
            "add_q_proj",
            "add_v_proj",
            "to_add_out",
            "to_out",
            "to_qkv_mlp_proj",
        ],
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

    model = FluxValInfer(
        base_model_path=args.base_model_path,
        weight_path=args.weight_path,
        lora_modules=args.lora_modules,
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        conditioning=args.conditioning,
        qwen_model_path=args.qwen_model_path,
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
            # VAE/DiT receives the 4x input, while Qwen receives the raw LR image.
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
