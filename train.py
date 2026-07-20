from argparse import ArgumentParser
from importlib import import_module

from omegaconf import OmegaConf


TRAINER_REGISTRY = {
    "zimage_vlm": ("HYPIR.trainer.zimage_vlm", "ZImageVLMTrainer"),
    "flux_vlm": ("HYPIR.trainer.flux_vlm", "FluxVLMTrainer"),
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
