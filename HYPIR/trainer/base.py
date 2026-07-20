import logging
import os
import shutil
from pathlib import Path
from typing import overload, List, Dict
import importlib
import warnings
from contextlib import nullcontext
import json
import time
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
import pyiqa
# Torch 2.6 introduces safe serialization helpers; provide backward-compatible fallbacks for 2.5.x
try:
    from torch.serialization import get_unsafe_globals_in_checkpoint, add_safe_globals  # type: ignore
except Exception:  # torch < 2.6.0
    def get_unsafe_globals_in_checkpoint(path):  # type: ignore
        # In torch<=2.5, there is no safe-unpickling gate, so nothing to add.
        return []

    def add_safe_globals(globals_list):  # type: ignore
        # No-op on older torch versions.
        return None
from torchvision.utils import make_grid
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedType
from tqdm.auto import tqdm
import transformers
import lpips
import diffusers
from diffusers import AutoencoderKL
from PIL import Image
from omegaconf import OmegaConf

# from REPA.loss import SILoss
from HYPIR.model.D import ImageConvNextDiscriminator
from HYPIR.utils.common import instantiate_from_config, log_txt_as_img, print_vram_state, SuppressLogging
from HYPIR.utils.ema import EMAModel
from HYPIR.utils.tabulate import tabulate
from HYPIR.utils.others import NoOpContext, EdgeDetectionModel, total_variation_loss
from HYPIR.trainer.checkpoint_utils import (
    load_qwen_projectors,
    load_trainable_state_dict,
    save_qwen_projectors,
    save_trainable_state_dict,
)

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

class BatchInput:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ValueError(f"Duplicated key in BatchInput: {name}")
        self.__dict__[name] = value

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self.__dict__[name] = value


class BaseTrainer:

    def __init__(self, config):
        self.config = config
        set_seed(config.seed)
        self.init_environment()
        if self.config.use_repa:
            self.init_repa()
        self.init_models()
        self.init_fdl()
        self.summary_models()
        self.init_optimizers()
        self.init_lr_schedulers()
        self.init_dataset()
        self.prepare_all()

    def init_environment(self):
        logging_dir = Path(self.config.output_dir, self.config.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.config.output_dir, logging_dir=logging_dir)

        deepspeed_plugin = None
        ds_plugin_args_log = None
        ds_cfg = getattr(self.config, "deepspeed", None)
        if ds_cfg:
            ds_kwargs = {}
            zero_stage = getattr(ds_cfg, "zero_stage", None)
            if zero_stage is not None:
                ds_kwargs["zero_stage"] = int(zero_stage)
            offload_optimizer = getattr(ds_cfg, "offload_optimizer", None)
            if offload_optimizer is not None:
                ds_kwargs["offload_optimizer"] = bool(offload_optimizer)
            offload_param = getattr(ds_cfg, "offload_param", None)
            if offload_param is not None:
                ds_kwargs["offload_param"] = bool(offload_param)
            gradient_clipping = getattr(ds_cfg, "gradient_clipping", None)
            if gradient_clipping is not None:
                ds_kwargs["gradient_clipping"] = float(gradient_clipping)
            grad_accum = getattr(ds_cfg, "gradient_accumulation_steps", None)
            if grad_accum is not None:
                ds_kwargs["gradient_accumulation_steps"] = int(grad_accum)

            config_file = getattr(ds_cfg, "config_file", None)
            if config_file:
                ds_kwargs["hf_ds_config"] = config_file

            deepspeed_plugin = DeepSpeedPlugin(**ds_kwargs)
            ds_plugin_args_log = ds_kwargs

        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with=self.config.report_to,
            project_config=accelerator_project_config,
            deepspeed_plugin=deepspeed_plugin,
        )
        if ds_plugin_args_log:
            logger.info(f"Using DeepSpeed plugin with args: {ds_plugin_args_log}")
        logger.info(accelerator.state, main_process_only=True)
        if accelerator.is_main_process:
            accelerator.init_trackers("train")
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_warning()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        if accelerator.is_main_process:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.device = accelerator.device

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        return model

    def init_models(self):
        print(f"Use VAE: {self.config.use_vae}, Use D: {self.config.use_D}, Use EMA: {self.config.use_ema}")
        self.init_scheduler()
        self.init_text_models()
        if self.config.use_vae:
            self.init_vae()
        self.init_generator()
        if self.config.use_D:
            self.init_discriminator()
        self.init_lpips()
        self.init_dists()

    @overload
    def init_scheduler(self):
        ...
    
    def init_repa(self):
        ...

    @overload
    def init_text_models(self):
        ...

    @overload
    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        ...

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype).to(self.device)
        self.vae.eval().requires_grad_(False)

    def init_lpips(self):
        with warnings.catch_warnings():
            # Suppress warnings from lpips
            warnings.simplefilter("ignore")
            self.net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
        self.net_lpips.eval().requires_grad_(False)

        if getattr(self.config, "lambda_edge_detect", False):
            self.edge_detection_model = EdgeDetectionModel().to(self.device)
            self.edge_detection_model.eval().requires_grad_(False)

    def init_fdl(self):
        self.fdl_loss = None
        if getattr(self.config, "use_fdl", False):
            # FDL-pytorch 1.0 executes CUDA code while its module is imported.
            # Import it only after Accelerator has selected this process's device.
            from FDL_pytorch import FDL_loss

            self.fdl_loss = FDL_loss().to(self.device)
            self.fdl_loss.eval().requires_grad_(False)

    def init_dists(self):
        self.metric_dists = pyiqa.create_metric('dists', device=self.device, as_loss=True)

    @overload
    def init_generator(self):
        ...

    def init_discriminator(self):
        # Suppress logs from open-clip
        ctx = (
            nullcontext()
            if self.accelerator.is_local_main_process
            else SuppressLogging(logging.WARNING)
        )
        with ctx:
            self.D = ImageConvNextDiscriminator(precision="fp32").to(device=self.device)
            self.D.train().requires_grad_(True)

    def summary_models(self):
        table_data = []
        for attr, value in self.__dict__.items():
            if not isinstance(value, torch.nn.Module):
                continue
            model = value
            model_type = type(model).__name__
            total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
            learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
            table_data.append([attr, model_type, f"{total_params:.2f}", f"{learnable_params:.2f}"])
        headers = ["Model Name", "Model Type", "Total Parameters (M)", "Learnable Parameters (M)"]
        table = tabulate(table_data, headers=headers, tablefmt="pretty")
        logger.info(f"Model Summary:\n{table}")

    def init_lr_schedulers(self):
        from diffusers.optimization import get_scheduler
        lr_scheduler = getattr(self.config, "lr_scheduler_type", "constant")
        logger.info(f"Creating {lr_scheduler} schedulers")
        lr_warmup_steps = getattr(self.config, "lr_warmup_steps", 0)

        self.G_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=self.G_opt,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=self.config.max_train_steps,
        )

        if hasattr(self, "D_opt"):
            self.D_scheduler = get_scheduler(
                lr_scheduler,
                optimizer=self.D_opt,
                num_warmup_steps=lr_warmup_steps,
                num_training_steps=self.config.max_train_steps,
            )

    def init_optimizers(self):
        logger.info(f"Creating {self.config.optimizer_type} optimizers")
        if self.config.optimizer_type == "adamw":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            optimizer_cls = None

        self.G_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
        
        if self.config.use_vae and hasattr(self, 'vae') and getattr(self.config, "train_encoder", True):
            vae_encoder_params = list(filter(lambda p: p.requires_grad, self.vae.encoder.parameters()))
            if vae_encoder_params:
                logger.info(f"Adding trainable parameters from VAE encoder to G_opt.")
                self.G_params.extend(vae_encoder_params)

        if self.config.use_vae and hasattr(self, 'vae') and self.config.train_decoder:
            vae_decoder_params = list(filter(lambda p: p.requires_grad, self.vae.decoder.parameters()))
            if vae_decoder_params:
                logger.info(f"Adding {len(vae_decoder_params)} trainable parameters from VAE decoder to G_opt.")
                self.G_params.extend(vae_decoder_params)

        if hasattr(self, 'projector') and getattr(self.config, "use_qwen", False):
            logger.info("Add proejctor to optimizer")
            projector_params = self.projector.parameters()
            self.G_params.extend(projector_params)

        self.G_opt = optimizer_cls(
            self.G_params,
            lr=self.config.lr_G,
            **self.config.opt_kwargs,
        )

        if self.config.use_D:
            self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
            self.D_opt = optimizer_cls(
                self.D_params,
                lr=self.config.lr_D,
                **self.config.opt_kwargs,
            )

    def init_dataset(self):
        data_cfg = self.config.data_config
        dataset = instantiate_from_config(data_cfg.train.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=data_cfg.train.batch_size,
            num_workers=data_cfg.train.dataloader_num_workers,
        )

        if hasattr(data_cfg, "validation") and hasattr(data_cfg.validation, "dataset"):
            val_dataset = instantiate_from_config(data_cfg.validation.dataset)
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=getattr(data_cfg.validation, "batch_size", 1),
                num_workers=getattr(data_cfg.validation, "dataloader_num_workers", 1),
            )
            if hasattr(data_cfg.validation, "batch_transform"):
                self.val_batch_transform = instantiate_from_config(data_cfg.validation.batch_transform)
            else:
                logger.warning("No validation batch_transform found, falling back to training transform.")
                self.val_batch_transform = self.batch_transform
            logger.info(f"Validation dataset initialized with {len(val_dataset)} samples.")
        else:
            self.val_dataloader = None

        self.batch_transform = instantiate_from_config(data_cfg.train.batch_transform)

    def prepare_all(self):
        logger.info("Wrapping models, optimizers and dataloaders")
        if self.accelerator.state.deepspeed_plugin is not None:
            logger.info("DeepSpeed detected: preparing generator only with accelerator")
            self.G, self.G_opt = self.accelerator.prepare(self.G, self.G_opt)
            self.dataloader = self.accelerator.prepare_data_loader(self.dataloader)
            # Prepare discriminator separately only when enabled
            if getattr(self.config, "use_D", False) and hasattr(self, "D") and self.D is not None:
                self.D = self.D.to(self.device)
                if self.accelerator.distributed_type != DistributedType.NO:
                    ddp_kwargs = {}
                    if self.device.type == "cuda":
                        ddp_kwargs["device_ids"] = [self.accelerator.device.index]
                    self.D = torch.nn.parallel.DistributedDataParallel(self.D, **ddp_kwargs)
        else:
            # Prepare only existing/required components when not using DeepSpeed
            attrs = ["G", "G_opt", "G_scheduler", "dataloader"]
            if getattr(self.config, "use_D", False) and hasattr(self, "D") and hasattr(self, "D_opt"):
                attrs.extend(["D", "D_opt", "D_scheduler"])
            if getattr(self.config, "use_refiner", False) and hasattr(self, "refiner") and hasattr(self, "refiner_opt"):
                attrs.extend(["refiner", "refiner_opt"])
            if getattr(self.config, "use_qwen", False) and hasattr(self, "projector"):
                attrs.extend(["projector"])
            prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
            for attr, obj in zip(attrs, prepared_objs):
                setattr(self, attr, obj)
        print_vram_state("After accelerator.prepare", logger=logger)

    def force_optimizer_ckpt_safe(self, checkpoint_dir):
        def get_symbol(s):
            module_name, symbol_name = s.rsplit('.', 1)
            module = importlib.import_module(module_name)
            symbol = getattr(module, symbol_name)
            return symbol

        for file_name in os.listdir(checkpoint_dir):
            if "optimizer" in file_name and not file_name.endswith("safetensors"):
                path = os.path.join(checkpoint_dir, file_name)
                unsafe_globals = get_unsafe_globals_in_checkpoint(path)
                logger.info(f"Unsafe globals in {path}: {unsafe_globals}")
                unsafe_globals = list(map(get_symbol, unsafe_globals))
                add_safe_globals(unsafe_globals)

    def attach_accelerator_hooks(self):
        ...

    def on_training_start(self):
        self._save_config()

        # Build ema state dict
        logger.info(f"Creating EMA handler, Use EMA = {self.config.use_ema}, EMA decay = {self.config.ema_decay}")
        if self.config.resume_from_checkpoint is not None and self.config.resume_ema:
            ema_resume_pth = os.path.join(self.config.resume_from_checkpoint, "ema_state_dict.pth")
        else:
            ema_resume_pth = None
        self.ema_handler = EMAModel(
            self.unwrap_model(self.G),
            decay=self.config.ema_decay,
            use_ema=self.config.use_ema,
            ema_resume_pth=ema_resume_pth,
            verbose=self.accelerator.is_local_main_process,
        )

        global_step = 0
        if self.config.resume_from_checkpoint:
            path = self.config.resume_from_checkpoint
            ckpt_name = os.path.basename(path)
            logger.info(f"Resuming from checkpoint {path}")
            accel_state_path = os.path.join(path, "model.safetensors")
            if os.path.exists(accel_state_path) and not getattr(self.config, "load_minimal_checkpoint", False):
                # Full state exists (non-minimal): resume via accelerator
                self.force_optimizer_ckpt_safe(path)
                self.accelerator.load_state(path)
            else:
                # Minimal checkpoint: load LoRA/trainable weights only
                logger.info("Load LoRA/trainable weights only")
                self._load_minimal_checkpoint(path)
            global_step = int(ckpt_name.split("-")[1])
            init_global_step = global_step
        else:
            init_global_step = 0

        self.global_step = global_step
        self.pbar = tqdm(
            range(0, self.config.max_train_steps),
            initial=init_global_step,
            desc="Steps",
            disable=not self.accelerator.is_main_process,
        )

    def _save_config(self):
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and not accelerator.is_main_process:
            return

        out_dir = getattr(self.config, "output_dir", None)
        if out_dir is None:
            raise ValueError("Cannot save config without an output_dir")

        os.makedirs(out_dir, exist_ok=True)
        cfg_obj = OmegaConf.to_container(self.config, resolve=True)

        import yaml
        yaml_path = os.path.join(out_dir, "config.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_obj, f, sort_keys=False)
        logger.info(f"Saved config to {yaml_path}")

    def prepare_batch_inputs(self, batch, transform=None):
        if transform == None:
            transform = self.batch_transform
        batch = transform(batch)
        gt = (batch["GT"] * 2 - 1).float()
        lq = (batch["LQ"] * 2 - 1).float()
        prompt = batch["txt"]
        bs = len(prompt)
        c_txt = self.encode_prompt(prompt)
        z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
        timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)
        self.batch_inputs = BatchInput(
            gt=gt, lq=lq,
            z_lq=z_lq,
            c_txt=c_txt,
            timesteps=timesteps,
            prompt=prompt,
        )

    @overload
    def forward_generator(self) -> torch.Tensor:
        ...

    def repa_loss(self, zs, zs_pred):
        latents_scale = torch.tensor(
            [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(self.device)
        latents_bias = torch.tensor(
            [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(self.device)

        def mean_flat(x):
            """
            Take the mean over all non-batch dimensions.
            """
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_pred) in enumerate(zip(zs, zs_pred )):
            for j, (z_j, z_pred_j) in enumerate(zip(z, z_pred)):
                z_pred_j = torch.nn.functional.normalize(z_pred_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1)[:z_pred_j.shape[0], :]
                proj_loss += mean_flat(-(z_j * z_pred_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)
        
        return proj_loss

    def relativistic_discriminator_loss(self, real_logits, fake_logits):
        r_real = real_logits - torch.mean(fake_logits, dim=0, keepdim=True)
        r_fake = fake_logits - torch.mean(real_logits, dim=0, keepdim=True)
        
        # 判别器目标：
        # 1. 真实样本比假样本更真实 (r_real -> 1)
        loss_real = F.binary_cross_entropy_with_logits(r_real, torch.ones_like(r_real))
        
        # 2. 假样本比真实样本更不真实 (r_fake -> 0)
        loss_fake = F.binary_cross_entropy_with_logits(r_fake, torch.zeros_like(r_fake))
        
        return (loss_real + loss_fake) / 2
    
    def relativistic_generator_loss(self, real_logits, fake_logits):
        r_real = real_logits - torch.mean(fake_logits, dim=0, keepdim=True)
        r_fake = fake_logits - torch.mean(real_logits, dim=0, keepdim=True)
        
        # 生成器目标（欺骗判别器，反转目标）：
        # 1. 假样本看起来比真实样本更真实 (r_fake -> 1)
        loss_fake = F.binary_cross_entropy_with_logits(r_fake, torch.ones_like(r_fake))
        
        # 2. 真实样本看起来比假样本更不真实 (r_real -> 0)
        loss_real = F.binary_cross_entropy_with_logits(r_real, torch.zeros_like(r_real))
        
        return (loss_real + loss_fake) / 2

    def validate(self):
        if self.val_dataloader is None:
            return

        logger.info("Running validation...")
        self.G.eval()
        if self.config.use_ema:
            self.ema_handler.activate_ema_weights()

        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0
        
        # Directory to save validation images
        val_save_dir = os.path.join(self.config.output_dir, self.config.logging_dir, "val_images", f"{self.global_step:07}")
        if self.accelerator.is_main_process:
            os.makedirs(val_save_dir, exist_ok=True)

        pbar_val = tqdm(self.val_dataloader, desc="Validation", disable=not self.accelerator.is_main_process)

        with torch.no_grad():
            for i, batch in enumerate(pbar_val):
                self.prepare_batch_inputs(batch, transform=self.val_batch_transform)
                
                if self.config.use_repa:
                    x, _, _ = self.forward_generator()
                else:
                    x, _ = self.forward_generator()
                
                # Metrics
                pred_img = (x + 1) / 2
                gt_img = (self.batch_inputs.gt + 1) / 2
                loss = F.mse_loss(x, self.batch_inputs.gt)
                total_loss += loss.item()
                mse = F.mse_loss(pred_img, gt_img)
                psnr = -10 * torch.log10(mse)
                total_psnr += psnr.item()
                num_batches += 1

                # Save images for the first few batches or specific interval
                # Only save on main process to avoid write conflicts
                if self.accelerator.is_main_process and i < 204: # Save first 5 batches
                    # Concatenate GT and Pred for side-by-side comparison
                    # x shape: [B, C, H, W]
                    vis_imgs = torch.cat([gt_img, pred_img], dim=3) # Concat horizontally
                    
                    image_arrs = (vis_imgs * 255.0).clamp(0, 255).to(torch.uint8) \
                        .permute(0, 2, 3, 1).contiguous().cpu().numpy()
                    
                    for j, img in enumerate(image_arrs):
                        file_name = f"batch{i}_sample{j}_gt_vs_pred.png"
                        Image.fromarray(img).save(os.path.join(val_save_dir, file_name))

        if self.config.use_ema:
            self.ema_handler.deactivate_ema_weights()
        self.G.train()

        # Aggregate metrics
        avg_loss = torch.tensor(total_loss / num_batches, device=self.device)
        avg_psnr = torch.tensor(total_psnr / num_batches, device=self.device)
        
        if self.accelerator.num_processes > 1:
            avg_loss = self.accelerator.gather(avg_loss).mean()
            avg_psnr = self.accelerator.gather(avg_psnr).mean()

        logger.info(f"Validation Step {self.global_step}: Loss={avg_loss.item():.4f}, PSNR={avg_psnr.item():.4f}")
        
        self.accelerator.log({
            "val/loss": avg_loss.item(),
            "val/psnr": avg_psnr.item()
        }, step=self.global_step)

    def optimize_generator_latent(self):
        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        # Avoid accelerate.accumulate (which uses no_sync) when ZeRO stage >= 2
        use_null_ctx = bool(ds_plugin and getattr(ds_plugin, "zero_stage", 0) >= 2)
        ctx = nullcontext() if use_null_ctx else self.accelerator.accumulate(self.G)
        with ctx:
            # Discriminator is not used in latent-only optimization; guard access if exists
            if hasattr(self, "D") and self.D is not None:
                self.unwrap_model(self.D).eval().requires_grad_(False)
            start_time = time.perf_counter()
            if self.config.use_repa:
                _, latent, zs_pred = self.forward_generator()
                zs = self.batch_inputs.z_s
            else:
                latent = self.forward_generator()
            end_time = time.perf_counter()
            # self.G_pred = x
            # Compute MSE loss in float32 to avoid dtype mismatch with bf16 params/backward
            loss_l2 = F.mse_loss(latent.float(), self.batch_inputs.z_gt.float(), reduction="mean") * self.config.lambda_l2
            loss_G = loss_l2.float()
            if self.config.use_repa:
                proj_loss = self.repa_loss(zs, zs_pred)
                loss_G += (1 - proj_loss) * self.config.proj_coef

            self.accelerator.backward(loss_G)
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(self.G_params, self.config.max_grad_norm)
            self.G_opt.step()
            self.G_opt.zero_grad()
        # Log something
        loss_dict = {"G_mse": loss_l2}
        if self.config.use_repa:
            # 计算 REPA 损失项本身
            repa_loss_term = (1 - proj_loss) * self.config.proj_coef
            loss_dict['G_repa'] = repa_loss_term
            # G_total 是两者的和
            loss_dict['G_total'] = loss_l2 + repa_loss_term
        else:
            # 如果没有 repa，G_total 就是 G_mse
            loss_dict['G_total'] = loss_l2

        for k, v in loss_dict.items():
            print(f"--------------loss key: {k}, loss value: {v}--------------")
        return loss_dict, grad_norm

    def optimize_generator_image(self):
        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        # Avoid accelerate.accumulate (which uses no_sync) when ZeRO stage >= 2
        use_null_ctx = bool(ds_plugin and getattr(ds_plugin, "zero_stage", 0) >= 2)
        ctx = nullcontext() if use_null_ctx else self.accelerator.accumulate(self.G)
        with ctx:
            if getattr(self.config, "use_D", False) and hasattr(self, "D") and self.D is not None:
                D_unwrapped = self.unwrap_model(self.D)
                if not getattr(self, "_discriminator_sn_calibrated", False):
                    with torch.no_grad():
                        D_unwrapped.train()(self.batch_inputs.gt, for_real=True, verbose=False)
                    self._discriminator_sn_calibrated = True
                D_unwrapped.eval().requires_grad_(False)
            if self.config.use_repa:
                x, latent, zs_pred = self.forward_generator()
                zs = self.batch_inputs.z_s
            else:
                x, latents = self.forward_generator()

            self.G_pred = x
            loss_l2 = F.mse_loss(x, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
            # loss_l2 = torch.zeros((), device=self.device, dtype=self.weight_dtype)
            loss_l1 = F.l1_loss(x, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l1
            # loss_l1 = torch.zeros((), device=self.device, dtype=self.weight_dtype)
            if not getattr(self.config, "use_dists", False):
                loss_lpips = self.net_lpips(x, self.batch_inputs.gt).mean() * self.config.lambda_lpips
                loss_dists = torch.zeros((), device=self.device, dtype=loss_l2.dtype)
            else:
                loss_lpips = torch.zeros((), device=self.device, dtype=loss_l2.dtype)
                loss_dists = self.metric_dists(((1 + x)/2).clamp(0, 1), ((1 + self.batch_inputs.gt)/2).clamp(0, 1)).mean() * self.config.lambda_dists

            if getattr(self.config, "lambda_edge_detect", False):
                edge_x = self.edge_detection_model(x)
                edge_gt = self.edge_detection_model(self.batch_inputs.gt)
                loss_edge = self.net_lpips(edge_x, edge_gt).mean() * self.config.lambda_edge_detect
                # print(f"------------loss edge: {loss_edge}------------")
            else:
                loss_edge = torch.zeros((), device=self.device, dtype=loss_l2.dtype)

            if getattr(self.config, "lambda_tv", False):
                tv_x = total_variation_loss(x)
                tv_gt = total_variation_loss(self.batch_inputs.gt)
                loss_tv = self.net_lpips(tv_x, tv_gt).mean() * self.config.lambda_tv
                # print(f"------------loss tv: {loss_tv}------------")
            else:
                loss_tv = torch.zeros((), device=self.device, dtype=loss_l2.dtype)

            if self.config.use_fdl:
                loss_fdl = self.fdl_loss(x, self.batch_inputs.gt) * self.config.lambda_fdl
            else:
                loss_fdl = torch.zeros((), device=self.device, dtype=loss_l2.dtype)

            if self.config.use_repa:
                proj_loss = (1 - self.repa_loss(zs, zs_pred)) * self.config.proj_coef
                # print(f"------------loss repa: {proj_loss}------------")
            else:
                proj_loss = torch.zeros((), device=self.device, dtype=loss_l2.dtype)

            if getattr(self.config, "use_D", False) and hasattr(self, "D") and self.D is not None and not self.is_warmup:
                # loss_disc = self.D(x, for_G=True, verbose=False).mean() * self.config.lambda_gan

                _, fake_logits = D_unwrapped(x, for_G=True, verbose=False, return_logits=True)
                with torch.no_grad():
                    _, real_logits = D_unwrapped(self.batch_inputs.gt, for_real=True, return_logits=True)
                
                loss_disc = 0.0
                for r, f in zip(real_logits, fake_logits):
                    # RaGAN loss for Generator
                    loss_disc = loss_disc + self.relativistic_generator_loss(r, f)
                loss_disc = loss_disc * self.config.lambda_gan
            else:
                loss_disc = torch.zeros((), device=self.device, dtype=loss_l2.dtype)

            loss_G = loss_l2 + loss_lpips + loss_dists +loss_disc + proj_loss + loss_edge + loss_tv + loss_l1 + loss_fdl
            self.accelerator.backward(loss_G)
            if self.accelerator.sync_gradients:
                norm_before_clip = self.accelerator.clip_grad_norm_(self.G_params, self.config.max_grad_norm)
                # norm_after_clip = self.accelerator.get_grad_norm(self.G_params, norm_type=2)
            self.G_opt.step()
            self.G_scheduler.step()
            self.G_opt.zero_grad()

            if hasattr(self, "refiner"):
                self.refiner_opt.step()
                self.refiner_opt.zero_grad()
        # Log something
        loss_dict = dict(G_total=loss_G, G_mse=loss_l2, G_l1=loss_l1, G_lpips=loss_lpips , G_dists=loss_dists, G_disc=loss_disc, G_fdl=loss_fdl)
        # logger.info(f"loss dict: {loss_dict}")
        return loss_dict, norm_before_clip

    def optimize_discriminator(self):
        gt = self.batch_inputs.gt
        with torch.no_grad():
            if self.config.use_repa:
                x = self.forward_generator()[0]
            else:
                x, latents = self.forward_generator()
        self.G_pred = x
        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        # Avoid accelerate.accumulate (which uses no_sync) when ZeRO stage >= 2
        use_null_ctx = bool(ds_plugin and getattr(ds_plugin, "zero_stage", 0) >= 2)
        ctx = nullcontext() if use_null_ctx else self.accelerator.accumulate(self.D)
        with ctx:
            self.unwrap_model(self.D).train().requires_grad_(True)
            loss_D_real, real_logits = self.D(gt, for_real=True, return_logits=True)
            loss_D_fake, fake_logits = self.D(x, for_real=False, return_logits=True)

            # _, real_logits = self.D(normalized_gt, for_real=True, return_logits=True)
            # _, fake_logits = self.D(normalized_x, for_real=False, return_logits=True)

            loss_D = 0.0
            for r, f in zip(real_logits, fake_logits):
                loss_D = loss_D + self.relativistic_discriminator_loss(r, f)
            
            # loss_D_real, real_logits = self.D(gt, for_real=True, return_logits=True)
            # loss_D_fake, fake_logits = self.D(x, for_real=False, return_logits=True)
            # loss_D = loss_D_real.mean() + loss_D_fake.mean()

            # calculate R1 loss
            if self.config.use_r1:    
                lambda_r1 = getattr(self.config, "lambda_r1", 1000.0)
                r1_sigma = getattr(self.config, "r1_sigma", 0.01)
                # noise = torch.rand_like(normalized_gt, device=gt.device, dtype=gt.dtype) * r1_sigma
                # noised_gt = normalized_gt + noise
                noise = torch.rand_like(gt, device=gt.device, dtype=gt.dtype) * r1_sigma
                noised_gt = gt + noise
                _, real_logits_noisy = self.D(noised_gt, for_real=True, return_logits=True)

                r1_per_sample = None

                for real, real_noised in zip(real_logits, real_logits_noisy):
                    diff = real - real_noised
                    sq = (diff ** 2).mean(dim=list(range(1, diff.dim())))
                    sq = sq.view(-1, 1)
                    if r1_per_sample is None:
                        r1_per_sample = sq
                    else:
                        r1_per_sample = r1_per_sample + sq
                approx_r1 = lambda_r1 * r1_per_sample.mean()
            else:
                approx_r1 = torch.tensor(0.0, device=self.device, dtype=self.weight_dtype)
            loss_D = loss_D + approx_r1
            self.accelerator.backward(loss_D)
            if self.accelerator.sync_gradients:
                norm_before_clip = self.accelerator.clip_grad_norm_(self.D_params, self.config.max_grad_norm)
                # norm_after_clip = self.accelerator.get_grad_norm(self.D_params, norm_type=2)
            self.D_opt.step()
            self.D_scheduler.step()
            self.D_opt.zero_grad()

        if hasattr(self, "refiner"):
            self.refiner_opt.step()
            self.refiner_opt.zero_grad()
        loss_dict = dict(D=loss_D, D_r1=approx_r1)
        # logits = D(x) w/o sigmoid = log(p_real(x) / p_fake(x))
        with torch.no_grad():
            real_logits = torch.tensor([logit_map.mean() for logit_map in real_logits], device=self.device).mean()
            fake_logits = torch.tensor([logit_map.mean() for logit_map in fake_logits], device=self.device).mean()
        loss_dict.update(dict(D_logits_real=real_logits, D_logits_fake=fake_logits))
        return loss_dict, norm_before_clip

    def run(self):
        self.attach_accelerator_hooks()
        self.on_training_start()
        self.batch_count = 0
        val_interval = getattr(self.config, "validation_steps", 10000)
        
        while self.global_step < self.config.max_train_steps:
            train_loss = {}
            for batch in self.dataloader:
                start_time = time.perf_counter()
                self.prepare_batch_inputs(batch)
                end_time = time.perf_counter()
                prepare_time = end_time - start_time
                # print(f"Take {prepare_time} to prepare inputs(most likely encode image)")
                try:
                    bs = len(self.batch_inputs.lq)
                except AttributeError:
                    bs = len(self.batch_inputs.z_lq)
                if self.config.use_warmup:
                    warmup_steps = getattr(self.config, "lr_warmup_steps", 3000)
                    self.is_warmup = self.global_step < warmup_steps
                else:
                    self.is_warmup = False
                # self.is_warmup = False
                if self.config.use_D and not self.is_warmup:
                    generator_step = ((self.batch_count // self.config.gradient_accumulation_steps) % 2) == 0
                else:
                    generator_step = True

                if generator_step:
                    if self.config.use_vae:
                        loss_dict, norm_before_clip = self.optimize_generator_image()
                    else:
                        loss_dict, norm_before_clip = self.optimize_generator_latent()
                else:
                    loss_dict, norm_before_clip = self.optimize_discriminator()

                for k, v in loss_dict.items():
                    avg_loss = self.accelerator.gather(v.repeat(bs)).mean()
                    if k not in train_loss:
                        train_loss[k] = 0
                    train_loss[k] += avg_loss.item() / self.config.gradient_accumulation_steps

                self.batch_count += 1
                if self.accelerator.sync_gradients:
                    if generator_step:
                        # update EMA
                        self.ema_handler.update()
                    state = "Generator     Step" if generator_step else "Discriminator Step"
                    # state = "Generator     Step" if not generator_step else "Discriminator Step"
                    _, _, peak = print_vram_state(None)
                    self.pbar.set_description(f"{state}, VRAM peak: {peak:.2f} GB")

                # Advance global step and log
                if self.accelerator.sync_gradients:
                    should_advance = (not self.config.use_D and generator_step) or (self.config.use_D and not generator_step) or (self.config.use_D and generator_step and self.is_warmup)
                    if should_advance:
                        self.global_step += 1
                        self.pbar.update(1)
                        log_dict = {}
                        for k in train_loss.keys():
                            log_dict[f"loss/{k}"] = train_loss[k]
                        train_loss = {}
                        if self.config.use_vae and (self.global_step % self.config.log_image_steps == 0 or self.global_step == 1):
                            self.log_images()
                        if self.config.log_grads and (self.global_step % self.config.log_grad_steps == 0 or self.global_step == 1):
                            if generator_step:
                                log_dict[f"norm_before_clip/G"] = norm_before_clip.item()
                            else:
                                log_dict[f"norm_before_clip/D"] = norm_before_clip.item()
                        self.accelerator.log(log_dict, step=self.global_step)
                        if self.global_step % self.config.checkpointing_steps == 0 or self.global_step == 1:
                            self.save_checkpoint()
                        # if (self.global_step % val_interval == 0 and self.global_step > 0) or self.global_step == 20000:
                        #     self.validate()
                if self.global_step >= self.config.max_train_steps:
                    break
        self.accelerator.end_training()

    def log_images(self):
        N = 4
        try:
            image_logs = dict(
                # lq=(self.batch_inputs.lq[:N] + 1) / 2,
                gt=(self.batch_inputs.gt[:N] + 1) / 2,
                G=(self.G_pred[:N] + 1) / 2,
                # prompt=(log_txt_as_img((256, 256), self.batch_inputs.prompt[:N]) + 1) / 2,
            )
        except AttributeError:
            image_logs = dict(
                G=(self.G_pred[:N] + 1) / 2,
                # prompt=(log_txt_as_img((256, 256), self.batch_inputs.prompt[:N]) + 1) / 2,
            )
        if self.config.use_ema:
            # recompute for EMA results
            self.ema_handler.activate_ema_weights()
            with torch.no_grad():
                if self.config.use_repa:
                    ema_x = self.forward_generator()[0]
                else:
                    ema_x = self.forward_generator()
                image_logs["G_ema"] = (ema_x[:N] + 1) / 2
            self.ema_handler.deactivate_ema_weights()

        if not self.accelerator.is_main_process:
            return

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                for tag, images in image_logs.items():
                    tracker.writer.add_image(
                        f"image/{tag}",
                        make_grid(images.float(), nrow=4),
                        self.global_step,
                    )

        for key, images in image_logs.items():
            image_arrs = (images * 255.0).clamp(0, 255).to(torch.uint8) \
                .permute(0, 2, 3, 1).contiguous().cpu().numpy()
            save_dir = os.path.join(
                self.config.output_dir, self.config.logging_dir, "log_images", f"{self.global_step:07}", key)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i, img in enumerate(image_arrs):
                Image.fromarray(img).save(os.path.join(save_dir, f"sample{i}.png"))

    def log_grads(self):
    # 判别器可能未启用，安全处理
        if hasattr(self, "D") and self.D is not None:
            self.unwrap_model(self.D).eval().requires_grad_(False)

        # --- 前向：与当前计算图一致地拿到输出 ---
        if self.config.use_vae:
            if self.config.use_repa:
                x, latent, zs_pred = self.forward_generator()
                zs = self.batch_inputs.z_s
            else:
                x, latents = self.forward_generator()
            x_float = x.float()
            gt_float = self.batch_inputs.gt.float()
            loss_l2 = F.mse_loss(x_float, gt_float, reduction="mean") * self.config.lambda_l2
            loss_lpips = self.net_lpips(x_float, gt_float).mean() * self.config.lambda_lpips
            
            # Relativistic GAN Loss Calculation
            if hasattr(self, "D") and self.D is not None:
                # Get Real Logits (No grad needed for G step)
                with torch.no_grad():
                    _, real_logits = self.D(gt_float, for_real=True, return_logits=True)
                
                # Get Fake Logits
                _, fake_logits = self.D(x_float, for_real=False, return_logits=True)

                loss_disc = 0.0
                if isinstance(real_logits, list):
                    for r, f in zip(real_logits, fake_logits):
                        loss_disc = loss_disc + self.relativistic_generator_loss(r, f)
                else:
                    loss_disc = self.relativistic_generator_loss(real_logits, fake_logits)
                
                loss_disc = loss_disc * self.config.lambda_gan
            else:
                loss_disc = torch.zeros((), device=self.device, dtype=x_float.dtype)

            loss_repa = (1 - self.repa_loss(zs, zs_pred)) * self.config.proj_coef if self.config.use_repa else  torch.zeros((), device=self.device, dtype=x_float.dtype)
            losses = [("l2", loss_l2), ("lpips", loss_lpips), ("disc", loss_disc)]
            if self.config.use_repa:
                losses.append(("lrepa", loss_repa))

        # --- 逐项反向，记录 LoRA/目标模块梯度范数 ---
        grad_dict = {}
        self.G_opt.zero_grad(set_to_none=True)
        for idx, (name, loss) in enumerate(losses):
            retain_graph = idx != len(losses) - 1
            try:
                # 调试当前 loss 名称
                print(f"Current gradient's loss: {name}")
                loss.backward(retain_graph=retain_graph)
            except RuntimeError as e:
                logger.error(f"[log_grads] backward failed for {name}: {e}")
                continue

            lora_module_grads = {}
            for module_name, module in self.unwrap_model(self.G).named_modules():
                for suffix in getattr(self.config, "log_grad_modules", []):
                    # if module_name.endswith(suffix):
                    if suffix in module_name:
                        grads = []
                        for p in module.parameters():
                            if p.requires_grad and p.grad is not None:
                                g = p.grad
                                if g.numel() > 0:
                                    grads.append(g.reshape(-1))
                        if grads:
                            flat_grad = torch.cat(grads, dim=0)
                            lora_module_grads.setdefault(suffix, []).append(flat_grad)
                        break
            for k, v in lora_module_grads.items():
                # 跳过空列表，避免 cat 空张量
                if not v:
                    continue
                grad_dict[f"grad_norm/{k}_{name}"] = torch.norm(torch.cat(v, dim=0)).item()

            self.G_opt.zero_grad(set_to_none=True)

        if grad_dict:
            self.accelerator.log(grad_dict, step=self.global_step)

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            if self.config.checkpoints_total_limit is not None:
                checkpoints = os.listdir(self.config.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                if len(checkpoints) >= self.config.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]
                    logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(self.config.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)
            save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
            os.makedirs(save_path, exist_ok=True)

            use_minimal = (
                getattr(self.accelerator.state, "deepspeed_plugin", None) is not None and
                getattr(self.config, "save_minimal_checkpoint", True)
            )
            if use_minimal:
                self._save_minimal_checkpoint(save_path)
                logger.info(f"Saved minimal checkpoint to {save_path}")
            else:
                self.accelerator.save_state(save_path)
                logger.info(f"Saved state via accelerator to {save_path}")

            if self.config.use_vae and hasattr(self, 'vae') and getattr(self.config, "train_encoder", True):
                vae_encoder_state_dict = self.unwrap_model(self.vae).encoder.state_dict()
                
                torch.save(vae_encoder_state_dict, os.path.join(save_path, "vae_encoder.pth"))
                logger.info(f"Saved fine-tuned VAE encoder weights to {save_path}/vae_encoder.pth")
            if self.config.use_vae and hasattr(self, 'vae') and self.config.train_decoder:
                vae_decoder_state_dict = self.unwrap_model(self.vae).decoder.state_dict()
                
                torch.save(vae_decoder_state_dict, os.path.join(save_path, "vae_decoder.pth"))
                logger.info(f"Saved fine-tuned VAE decoder weights to {save_path}/vae_decoder.pth")
            if self.config.use_refiner and hasattr(self, "refiner"):
                refiner_state_dict = self.unwrap_model(self.refiner).state_dict()

                torch.save(refiner_state_dict, os.path.join(save_path, "refiner.pth"))
                logger.info(f"Saved refiner weights to {save_path}/refiner.pth")
            # Save ema weights (works for both modes)
            self.ema_handler.save_ema_weights(save_path)
            logger.info(f"Saved ema weights to {save_path}")

    def _save_minimal_checkpoint(self, save_path: str):
        logger.info("Saving minimal checkpoint (trainable parameters only)")
        model = self.unwrap_model(self.G)
        save_trainable_state_dict(model, os.path.join(save_path, "state_dict.pth"))
        if getattr(self.config, "use_qwen", False):
            save_qwen_projectors(self, save_path)
        trainer_state = {"global_step": self.global_step}
        with open(os.path.join(save_path, "trainer_state.json"), "w") as f:
            json.dump(trainer_state, f)

    def _load_minimal_checkpoint(self, load_path: str):
        model = self.unwrap_model(self.G)
        load_trainable_state_dict(model, os.path.join(load_path, "state_dict.pth"))
        if getattr(self.config, "use_qwen", False):
            load_qwen_projectors(self, load_path)
