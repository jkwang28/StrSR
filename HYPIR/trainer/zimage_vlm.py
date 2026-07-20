import logging
from typing import Any, Callable, Dict, List, Optional, Union
import os
import torch
import random
from tqdm.auto import tqdm
from accelerate.logging import get_logger
from contextlib import nullcontext
from peft import LoraConfig, get_peft_model
from HYPIR.utils.ema import EMAModel
try:
    from peft import mark_only_lora_as_trainable
except ImportError:
    def mark_only_lora_as_trainable(model):
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
from HYPIR.utils.common import (
    instantiate_from_config,
    log_txt_as_img,
    print_vram_state,
    SuppressLogging,
    module_param_memory,
    human_bytes,
)
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from HYPIR.utils.common import instantiate_from_config, log_txt_as_img, print_vram_state, SuppressLogging
from HYPIR.model.backbone import CNNRefiner
from HYPIR.model.D import ImageConvNextDiscriminator
from diffusers.models.transformers import ZImageTransformer2DModel
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from HYPIR.trainer.base import BaseTrainer, BatchInput
from HYPIR.utils.others import NoOpContext, EdgeDetectionModel, total_variation_loss
from HYPIR.utils.captioner import IMAGE_DESCRIPTION_PROMPT
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
logger = get_logger(__name__, log_level="INFO")

# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "Qwen3ForCausalLM":
        from transformers import Qwen3ForCausalLM

        return Qwen3ForCausalLM
    else:
        raise "Invalid Text Encoder"


# Copied from dreambooth sd3 example
def load_text_encoder(class_text_encoder, args):
    text_encoder = class_text_encoder.from_pretrained(
        args.base_model_path, subfolder="text_encoder"
    )
    return text_encoder

class ZImageVLMTrainer(BaseTrainer):
    def step(self, latents, noise_pred, sigmas, step_i):
        return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()    
    
    def init_scheduler(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.base_model_path, subfolder="scheduler"
        )

    def init_dataset(self):
        super().init_dataset()
        # load from saved debug inputs
        # if not self.config.use_txt:
            # pos_promt_emb = torch.load(f"debug_inputs/prompt_embeds.pt")
            # self.c_txt = {"prompt_embeds": [pos_promt_emb.to(self.device)]}
        
    def prepare_batch_inputs(self, batch, transform=None):
        if transform == None:
            transform = self.batch_transform
        batch = transform(batch)
        gt = (batch["GT"] * 2 - 1).float().to(self.device)
        lq = (batch["LQ"] * 2 - 1).float().to(self.device)
        origin_lq = batch["low_LQ"].float().to(self.device)
        bs = lq.shape[0]
        z_lq = self.vae.encode(lq.to(device=self.device, dtype=self.weight_dtype)).latent_dist.sample()
        z_gt = self.vae.encode(gt.to(device=self.device, dtype=self.weight_dtype)).latent_dist.sample()
        timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)
        prompt = batch["txt"]
        self.c_txt = {}
        if getattr(self.config, "use_qwen", False):
            qwen_visual_embeds, qwen_text_embeds = self.extract_qwen_feature(
                origin_lq, [IMAGE_DESCRIPTION_PROMPT] * bs
            )
            self.c_txt["vision_embeds"] = qwen_visual_embeds
            self.c_txt["text_embeds"] = qwen_text_embeds
        else:
            with torch.no_grad():
                if self.config.use_txt:
                    prompt_embeds, _ = self.encode_prompt(prompt=prompt)
                    self.c_txt["prompt_embeds"] = prompt_embeds,
                else:
                    pos_promt_emb = torch.load(f"debug_inputs/prompt_embeds.pt")
                    self.c_txt["prompt_embeds"] = [pos_promt_emb.to(self.device)] * bs

        self.batch_inputs = BatchInput(
            gt=gt, lq=lq,
            z_lq=z_lq, z_gt=z_gt,
            timesteps=timesteps,
        )
        
    def init_models(self):
        print(f"Use VAE: {self.config.use_vae}, Use D: {self.config.use_D}, Use EMA: {self.config.use_ema}")
        self.init_scheduler()
        if getattr(self.config, "use_vae", True):
            self.init_vae()
        self.init_generator()
        if getattr(self.config, "use_refiner", False):
            self.init_refiner()
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
            G_bytes_trainable = module_param_memory(self.G, only_trainable=True)
            G_bytes_all = module_param_memory(self.G, only_trainable=False)
            D_bytes = module_param_memory(self.D)
            lpips_bytes = module_param_memory(self.net_lpips)
            logger.info(
                "[Param memory] VAE=%s, G(trainable)=%s, G(all)=%s, D=%s, LPIPS=%s",
                human_bytes(vae_bytes),
                human_bytes(G_bytes_trainable),
                human_bytes(G_bytes_all),
                human_bytes(D_bytes),
                human_bytes(lpips_bytes),
            )
        except Exception as exc:
            logger.warning(f"Param memory report failed: {exc}")

    def init_qwen_projector(self):
        base_model = self.G.module if hasattr(self.G, "module") else self.G
        target_dim = base_model.cap_embedder[0].weight.shape[0]
        qwen_dim = getattr(self, "qwen_hidden_size", 3584)

        logger.info(f"Initializing qwen_projector: {qwen_dim} -> {target_dim}")
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(qwen_dim, target_dim // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(target_dim // 2, target_dim),
        ).to(self.device, dtype=self.weight_dtype)
        self.projector.train().requires_grad_(True)

    def init_qwen(self):
        logger.info("Loading Qwen3-VL for visual conditioning...")
        model_path = self.config.qwen_model_path
        
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device,
            attn_implementation="flash_attention_2"
        ).eval()
        self.qwen_model.requires_grad_(False)
        
        self.qwen_processor = AutoProcessor.from_pretrained(model_path)
        
        self.image_token_id = self.qwen_model.config.image_token_id
        self.qwen_hidden_size = self.qwen_model.config.text_config.hidden_size
        logger.info(f"✓ Qwen3-VL loaded. Hidden size: {self.qwen_hidden_size}")

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype).to(self.device)

        self.vae = self.vae.to(self.device, dtype=self.weight_dtype)
        logger.info("✓ VAE loaded")
        self.vae.eval().requires_grad_(False)
        print_vram_state("After VAE to(device)", logger=logger)

    def init_generator(self):
        self.G = ZImageTransformer2DModel.from_pretrained(
            self.config.base_model_path, 
            low_cpu_mem_usage=False,
            subfolder="transformer", 
            torch_dtype=self.weight_dtype
        ).to(self.device)

        logger.info("✓ DiT model loaded")
        print_vram_state("After DiT to(device)", logger=logger)

        if getattr(self.config, "gradient_checkpointing", False):
            if hasattr(self.G, "enable_gradient_checkpointing"):
                self.G.enable_gradient_checkpointing()

        target_patterns = list(getattr(self.config, "lora_modules", []) or [])
        if target_patterns:
            resolved_targets = self._resolve_lora_targets(self.G, target_patterns)
            if not resolved_targets:
                raise ValueError(
                    f"Failed to match any LoRA target modules. Requested patterns: {target_patterns}"
                )
            logger.info(f"Add LoRA parameters to {resolved_targets}")
            G_lora_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_patterns,
            )
            self.G = get_peft_model(self.G, G_lora_cfg)
            # mark_only_lora_as_trainable(self.G)
            lora_params = [p for p in self.G.parameters() if p.requires_grad]
            assert lora_params, "Failed to find LoRA parameters"
            for p in lora_params:
                p.data = p.data.to(device=self.device, dtype=torch.float32)
            self.G.to(self.device)
            self.lora_target_modules = resolved_targets
            print_vram_state("After enabling LoRA", logger=logger)
        else:
            logger.warning("LoRA modules list is empty; generator will remain frozen.")

        # if getattr(self.config, "use_qwen", False):
        #     self.init_qwen_projector()
        self._set_byt5_precision(self.weight_dtype)
        # Ensure module is in training mode so gradient checkpointing can take effect,
        # while keeping only LoRA params trainable
        self.G.train()

    def init_discriminator(self):
        # Suppress logs from open-clip
        ctx = (
            nullcontext()
            if self.accelerator.is_local_main_process
            else SuppressLogging(logging.WARNING)
        )
        with ctx:
            self.D = ImageConvNextDiscriminator(
                precision="fp32",
                clip_model_path=getattr(self.config, "clip_model_path", None),
            ).to(device=self.device)
            self.D.train().requires_grad_(True)

    def _resolve_lora_targets(self, model, target_patterns):
        candidate_types = (
            torch.nn.Linear,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
        )
        matched = []
        for module_name, module in model.named_modules():
            if not isinstance(module, candidate_types):
                continue
            if any(module_name.endswith(pattern) for pattern in target_patterns):
                matched.append(module_name)
        return sorted(set(matched))
        
    def init_text_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_path,
            subfolder="tokenizer",
        )
        
        self.text_encoder_cls = import_model_class_from_model_name_or_path(
            self.config.base_model_path, subfolder = "text_encoder"
        )
        
        self.text_encoder = load_text_encoder(
            self.text_encoder_cls, self.config
        )

        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        logger.info("Load Text Tokenizer and Encoder")

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = False,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=self.device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt)
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=self.device,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds

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
                        {"type": "text", "text": prompts[i]}, # 使用 prompt 引导 Qwen 关注特定内容
                    ],
                }
            ]
            messages_batch.append(messages)

        # 注意：process_vision_info 需要逐个处理
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

        # base_model = self.G.module if hasattr(self.G, "module") else self.G
        # projector = base_model.qwen_projector
        # proj_dtype = next(projector.parameters()).dtype
        
        visual_embeds_list, text_embeds_list = [], []

        for i in range(batch_size):
            input_ids = inputs.input_ids[i]
            attn_mask = inputs.attention_mask[i]

            image_mask = input_ids == self.image_token_id
            text_mask = (attn_mask == 1) & (input_ids != self.image_token_id)


            vis_tokens = last_hidden_state[i][image_mask]              # [N_vis, H_qwen]
            vis_tokens = vis_tokens.to(dtype=self.weight_dtype)
            visual_embeds_list.append(vis_tokens)

            text_tokens = last_hidden_state[i][text_mask]              # [N_txt, H_qwen]
            text_tokens = text_tokens.to(dtype=self.weight_dtype)
            proj_text_tokens = self.projector(text_tokens)
            text_embeds_list.append(proj_text_tokens)

        return visual_embeds_list, text_embeds_list
    
    def attach_accelerator_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                model = models[0]
                weights.pop(0)
                model = self.unwrap_model(model)
                assert isinstance(model, ZImageTransformer2DModel) or hasattr(model, "base_model")
                state_dict = {
                    name: param.detach().cpu()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                torch.save(state_dict, os.path.join(output_dir, "state_dict.pth"))
                for i, model in enumerate(models):
                    unwrapped = self.unwrap_model(model)
                    
                    # 保存 Projector
                    if hasattr(self, "projector") and unwrapped is self.unwrap_model(self.projector):
                        torch.save(unwrapped.state_dict(), os.path.join(output_dir, "projector.pth"))
                        weights.pop(i-1)

        def load_model_hook(models, input_dir):
            model = models.pop(0)
            model = self.unwrap_model(model)
            assert isinstance(model, ZImageTransformer2DModel) or hasattr(model, "base_model")
            state_dict = torch.load(os.path.join(input_dir, "state_dict.pth"), map_location="cpu")
            load_result = model.load_state_dict(state_dict, strict=False)
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            trainable_keys = set(n for n, p in model.named_parameters() if p.requires_grad)
            real_missing = [k for k in missing if k in trainable_keys]

            if real_missing:
                logger.info(f"LoRA missing keys (trainable parameters that failed to load): {real_missing}")
            elif missing:
                logger.info(f"Successfully loaded LoRA weights. Ignored {len(missing)} missing keys for frozen base model parameters.")
            if unexpected:
                logger.info(f"LoRA unexpected keys: {unexpected}")
                
            for i, model in enumerate(models):
                unwrapped = self.unwrap_model(model)
                if hasattr(self, "projector") and unwrapped is self.unwrap_model(self.projector):
                    models.pop(i)
                    proj_path = os.path.join(input_dir, "projector.pth")
                    if os.path.exists(proj_path):
                        logger.info(f"Loading projector from {proj_path}")
                        state_dict = torch.load(proj_path, map_location="cpu")
                        unwrapped.load_state_dict(state_dict)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def _set_byt5_precision(self, dtype: torch.dtype):
        base_model = self.G
        # unwrap Peft or DDP wrappers to reach underlying module
        if hasattr(base_model, "get_base_model"):
            base_model = base_model.get_base_model()
        if hasattr(base_model, "module"):
            base_model = base_model.module
        byt5_module = getattr(base_model, "byt5_in", None)
        if byt5_module is None:
            logger.warning("ByT5 module not found; skip precision adjustment")
            return
        byt5_module.to(device=self.device, dtype=dtype)
        # ensure LayerNorm params stay in desired dtype
        if hasattr(byt5_module, "layernorm"):
            byt5_module.layernorm.to(dtype=dtype)
        for param in byt5_module.parameters():
            param.requires_grad = False

    def _denoise_step(self, latents, timesteps, prompt_embeds, timesteps_r=None):
        """
        Perform one denoising step.

        Args:
            latents: Latent tensor
            timesteps: Timesteps tensor
            text_emb: Text embedding
            text_mask: Text mask
            byt5_emb: byT5 embedding
            byt5_mask: byT5 mask
            guidance_scale: Guidance scale
            timesteps_r: Optional next timestep

        Returns:
            Noise prediction tensor
        """
        latent_model_input = latents.to(device=self.device, dtype=self.weight_dtype)
        timestep_model_input = timesteps.to(device=self.device)
        prompt_embeds_model_input = [embeds.to(device=self.device, dtype=self.weight_dtype) for embeds in prompt_embeds]

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))
        if getattr(self, "accelerator", None) is None or self.accelerator.is_local_main_process:
            base_model = self.G.module if hasattr(self.G, "module") else self.G

        guidance_expand = None
        
        noise_pred= self.G(
            latent_model_input_list,
            timestep_model_input,
            prompt_embeds_model_input,
        )[0]
        
        return noise_pred
    
    def forward_generator(self):
        t_expand = (1000 - self.batch_inputs.timesteps) / 1000
        sigmas = torch.tensor([self.config.coeff_t / 1000.0, 0]).to(dtype=torch.float32, device=self.device)
        latent_model_input = self.batch_inputs.z_lq
        model_out_list = self._denoise_step(
            latent_model_input, t_expand, 
            self.c_txt["text_embeds"],
            timesteps_r=None
        )
        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)
        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred
        # noise_pred = -noise_pred
        # print_vram_state("After _denoise_step (G forward)", logger=logger)
        latents = self.step(latent_model_input, noise_pred, sigmas, 0)

        # If we are training in latent space, return latents directly
        if not getattr(self.config, "use_vae", True):
            return latents
        
        x = self._decode_latents(latents.to(self.weight_dtype)).float()
        return x, latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            latents = latents / self.vae.config.scaling_factor

        latents = latents.to(dtype=self.weight_dtype).contiguous()
        image = self.vae.decode(latents, return_dict=False)[0]
        
        if getattr(self.config, "use_refiner", False):
            image = self.refiner(image)
        return image
