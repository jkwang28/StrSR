from argparse import ArgumentParser
from importlib import import_module

from omegaconf import OmegaConf


TRAINER_REGISTRY = {
    "sd2": ("HYPIR.trainer.sd2", "SD2Trainer"),
    "hy": ("HYPIR.trainer.hy", "HunyuanImage21Trainer"),
    "sd35": ("HYPIR.trainer.sd35", "SD35Trainer"),
    "sd35_repa": ("HYPIR.trainer.sd35_repa", "SD35REPATrainer"),
    "zimage": ("HYPIR.trainer.zimage", "ZImageTrainer"),
    "zimage_val": ("HYPIR.trainer.zimage_val", "ZImageTrainerVal"),
    "zimage_irepa": ("HYPIR.trainer.zimage_irepa", "ZImageREPATrainer"),
    "zimage_turbo": ("HYPIR.trainer.zimage_turbo", "ZImageTurboTrainer"),
    "zimage_vlm": ("HYPIR.trainer.zimage_vlm", "ZImageVLMTrainer"),
    "zimage_denoise": ("HYPIR.trainer.zimage_denoise", "ZImageDenoiseTrainer"),
    "zimage_ablation": ("HYPIR.trainer.zimage_ablation", "ZImageAblationTrainer"),
    "flux_vlm": ("HYPIR.trainer.flux_vlm", "FluxVLMTrainer"),
    "objectclear": ("HYPIR.trainer.objectclear", "ObjectClearTrainer"),
    "objectclear_new": ("HYPIR.trainer.objectclear_new", "ObjectClearNewTrainer"),
}


def load_trainer_class(base_model_type: str):
    try:
        module_name, class_name = TRAINER_REGISTRY[base_model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(TRAINER_REGISTRY))
        raise ValueError(f"Unsupported model type: {base_model_type}. Supported: {supported}") from exc

    module = import_module(module_name)
    return getattr(module, class_name)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    trainer_cls = load_trainer_class(config.base_model_type)
    trainer = trainer_cls(config)
    trainer.run()


if __name__ == "__main__":
    main()
