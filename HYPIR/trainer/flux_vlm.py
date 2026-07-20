import logging
import os
from contextlib import nullcontext
from typing import List, Tuple, Union

import torch
from accelerate.logging import get_logger
from diffusers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2TokenizerFast, Qwen3ForCausalLM, Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from HYPIR.model.D import ImageConvNextDiscriminator
from HYPIR.trainer.base import BaseTrainer, BatchInput
from HYPIR.utils.common import SuppressLogging, human_bytes, module_param_memory, print_vram_state
from HYPIR.utils.captioner import IMAGE_DESCRIPTION_PROMPT

try:
    from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
except Exception:
    AutoencoderKLFlux2 = None
    Flux2Transformer2DModel = None

logger = get_logger(__name__, log_level="INFO")


class FluxVLMTrainer(BaseTrainer):
    def step(self, latents, noise_pred, sigmas, step_i):
        return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()

    def init_scheduler(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.base_model_path,
            subfolder="scheduler",
        )

    def init_models(self):
        if Flux2Transformer2DModel is None or AutoencoderKLFlux2 is None:
            raise ImportError(
                "Flux.2 classes are unavailable. Please upgrade diffusers/transformers to versions that provide "
                "Flux2Transformer2DModel and AutoencoderKLFlux2."
            )

        logger.info(
            "Use VAE: %s, Use D: %s, Use EMA: %s",
            getattr(self.config, "use_vae", True),
            getattr(self.config, "use_D", True),
            getattr(self.config, "use_ema", True),
        )

        self.init_scheduler()
        if getattr(self.config, "use_vae", True):
            self.init_vae()
        self.init_generator()
        if getattr(self.config, "use_D", True):
            self.init_discriminator()
        if getattr(self.config, "use_vae", True):
            self.init_lpips()
        if getattr(self.config, "use_txt", False):
            self.init_text_models()
        if getattr(self.config, "use_qwen", False):
            self.init_qwen()
            self.init_qwen_projector()

        try:
            vae_bytes = module_param_memory(self.vae)
            g_bytes_trainable = module_param_memory(self.G, only_trainable=True)
            g_bytes_all = module_param_memory(self.G, only_trainable=False)
            d_bytes = module_param_memory(self.D)
            lpips_bytes = module_param_memory(self.net_lpips)
            logger.info(
                "[Param memory] VAE=%s, G(trainable)=%s, G(all)=%s, D=%s, LPIPS=%s",
                human_bytes(vae_bytes),
                human_bytes(g_bytes_trainable),
                human_bytes(g_bytes_all),
                human_bytes(d_bytes),
                human_bytes(lpips_bytes),
            )
        except Exception as exc:
            logger.warning(f"Param memory report failed: {exc}")

    def init_vae(self):
        self.vae = AutoencoderKLFlux2.from_pretrained(
            self.config.base_model_path,
            subfolder="vae",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.vae.eval().requires_grad_(False)
        logger.info("Flux.2 VAE loaded")
        print_vram_state("After VAE to(device)", logger=logger)

    def init_generator(self):
        self.G = Flux2Transformer2DModel.from_pretrained(
            self.config.base_model_path,
            subfolder="transformer",
            low_cpu_mem_usage=False,
            torch_dtype=self.weight_dtype,
        ).to(self.device)

        logger.info("Flux.2 transformer loaded")
        print_vram_state("After Flux2 transformer to(device)", logger=logger)

        if getattr(self.config, "gradient_checkpointing", False) and hasattr(self.G, "enable_gradient_checkpointing"):
            self.G.enable_gradient_checkpointing()

        target_patterns = list(getattr(self.config, "lora_modules", []) or [])
        if target_patterns:
            resolved_targets = self._resolve_lora_targets(self.G, target_patterns)
            if not resolved_targets:
                raise ValueError(f"Failed to match LoRA target modules: {target_patterns}")
            logger.info(f"Apply LoRA to: {resolved_targets}")
            lora_cfg = LoraConfig(
                r=getattr(self.config, "lora_rank", 16),
                lora_alpha=getattr(self.config, "lora_alpha", getattr(self.config, "lora_rank", 16)),
                init_lora_weights="gaussian",
                target_modules=resolved_targets,
            )
            self.G = get_peft_model(self.G, lora_cfg)

            lora_params = [p for p in self.G.parameters() if p.requires_grad]
            if not lora_params:
                raise ValueError("No trainable LoRA parameters found on Flux.2 transformer.")
            for p in lora_params:
                p.data = p.data.to(device=self.device, dtype=torch.float32)
            self.G.to(self.device)
            self.lora_target_modules = resolved_targets
            print_vram_state("After enabling Flux2 LoRA", logger=logger)
        else:
            logger.warning("LoRA modules list is empty; generator remains frozen.")

        self.G.train()

    def init_discriminator(self):
        ctx = nullcontext() if self.accelerator.is_local_main_process else SuppressLogging(logging.WARNING)
        with ctx:
            self.D = ImageConvNextDiscriminator(
                precision="fp32",
                clip_model_path=getattr(self.config, "clip_model_path", None),
            ).to(device=self.device)
            self.D.train().requires_grad_(True)

    def _resolve_lora_targets(self, model, target_patterns):
        candidate_types = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
        matched = []
        for module_name, module in model.named_modules():
            if isinstance(module, candidate_types) and any(module_name.endswith(p) for p in target_patterns):
                matched.append(module_name)
        return sorted(set(matched))

    def init_text_models(self):
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(self.config.base_model_path, subfolder="tokenizer")
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            self.config.base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        logger.info("Flux.2 tokenizer + Qwen3 text encoder loaded")

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

    def init_qwen(self):
        logger.info("Loading Qwen3-VL for visual conditioning...")
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.qwen_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        self.qwen_model.requires_grad_(False)
        self.qwen_processor = AutoProcessor.from_pretrained(self.config.qwen_model_path)
        self.image_token_id = self.qwen_model.config.image_token_id
        self.qwen_hidden_size = self.qwen_model.config.text_config.hidden_size

    def init_qwen_projector(self):
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
        self.projector.train().requires_grad_(True)

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

        max_tokens = int(getattr(self.config, "flux_max_text_tokens", 512))
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

    def prepare_batch_inputs(self, batch, transform=None):
        transform = transform or self.batch_transform
        batch = transform(batch)

        gt = (batch["GT"] * 2 - 1).float().to(self.device)
        lq = (batch["LQ"] * 2 - 1).float().to(self.device)
        bs = lq.shape[0]

        z_lq = self.vae.encode(lq.to(device=self.device, dtype=self.weight_dtype)).latent_dist.sample()
        z_gt = self.vae.encode(gt.to(device=self.device, dtype=self.weight_dtype)).latent_dist.sample()
        z_lq = self._normalize_flux2_latents(self._patchify_latents(z_lq))
        z_gt = self._normalize_flux2_latents(self._patchify_latents(z_gt))
        timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)

        if getattr(self.config, "use_qwen", False):
            raw_lq = batch["low_LQ"].float().to(self.device).clamp(0, 1)
            qwen_prompts = [IMAGE_DESCRIPTION_PROMPT] * bs
            text_embeds, text_ids = self.extract_qwen_feature(raw_lq, qwen_prompts)
        elif getattr(self.config, "use_txt", False):
            prompts = batch.get("txt", ["Describe this image in detail"] * bs)
            layer_ids = tuple(getattr(self.config, "flux_text_encoder_layers", (9, 18, 27)))
            text_embeds = self._get_qwen3_prompt_embeds(prompts, hidden_states_layers=layer_ids)
            text_ids = self._prepare_text_ids(text_embeds)
        else:
            text_embeds = torch.load("debug_inputs/default_prompt_embeds.pt", map_location="cpu").to(self.device)
            text_embeds = torch.zeros_like(text_embeds).to(self.device)
            text_ids = self._prepare_text_ids(text_embeds)

        self.c_txt = {"text_embeds": text_embeds, "text_ids": text_ids}
        self.batch_inputs = BatchInput(gt=gt, lq=lq, z_lq=z_lq, z_gt=z_gt, timesteps=timesteps)

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

    def forward_generator(self):
        sigmas = torch.tensor([self.config.coeff_t / 1000.0, 0.0], dtype=torch.float32, device=self.device)

        latent_model_input = self.batch_inputs.z_lq
        _, _, h, w = latent_model_input.shape
        packed_latents = self._pack_flux2_latents(latent_model_input)
        noise_pred = self._denoise_step(packed_latents, self.batch_inputs.timesteps, h, w)
        packed_denoised = self.step(packed_latents, noise_pred, sigmas, 0)
        latents = self._unpack_flux2_latents(packed_denoised, h, w)

        if not getattr(self.config, "use_vae", True):
            return latents

        x = self._decode_latents(latents.to(self.weight_dtype)).float()
        return x, latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self._denormalize_flux2_latents(latents)
        latents = self._unpatchify_latents(latents)
        return self.vae.decode(latents.to(dtype=self.weight_dtype), return_dict=False)[0]

    def attach_accelerator_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if not self.accelerator.is_main_process:
                return
            model = self.unwrap_model(models[0])
            weights.pop(0)
            state_dict = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            torch.save(state_dict, os.path.join(output_dir, "state_dict.pth"))
            if hasattr(self, "projector"):
                torch.save(self.unwrap_model(self.projector).state_dict(), os.path.join(output_dir, "projector.pth"))

        def load_model_hook(models, input_dir):
            model = self.unwrap_model(models.pop(0))
            state_dict = torch.load(os.path.join(input_dir, "state_dict.pth"), map_location="cpu")
            load_result = model.load_state_dict(state_dict, strict=False)
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
            real_missing = [k for k in missing if k in trainable_keys]
            if real_missing:
                logger.info(f"LoRA missing keys: {real_missing}")
            if unexpected:
                logger.info(f"LoRA unexpected keys: {unexpected}")
            if hasattr(self, "projector"):
                proj_path = os.path.join(input_dir, "projector.pth")
                if os.path.exists(proj_path):
                    self.unwrap_model(self.projector).load_state_dict(torch.load(proj_path, map_location="cpu"))

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)
