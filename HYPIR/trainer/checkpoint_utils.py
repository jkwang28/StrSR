"""Helpers for validating the trainable part of minimal checkpoints."""

import os
from collections.abc import Mapping

import torch


_QWEN_PROJECTOR_FILES = (
    ("projector", "projector.pth"),
    ("qwen_projector_text", "qwen_projector_text.pth"),
    ("qwen_projector_pooled", "qwen_projector_pooled.pth"),
)


def _load_state_dict(path):
    """Load a tensor-only checkpoint on CPU across supported torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location="cpu")


def save_trainable_state_dict(model, path):
    """Save the parameters that a minimal checkpoint is expected to restore."""
    state_dict = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(state_dict, path)


def load_trainable_state_dict(model, path):
    """Restore trainable parameters and reject LoRA key drift.

    Minimal checkpoints intentionally omit frozen base-model parameters. Those
    keys are therefore allowed to be missing, while every trainable key must be
    present and no checkpoint key may be unknown.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Minimal checkpoint missing trainable weights: {path}")

    state_dict = _load_state_dict(path)
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(f"Expected a state-dict mapping in minimal checkpoint: {path}")

    trainable_keys = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    checkpoint_keys = set(state_dict)
    missing = sorted(trainable_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - trainable_keys)

    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing LoRA/trainable keys: {missing}")
        if unexpected:
            problems.append(f"unexpected keys: {unexpected}")
        raise RuntimeError(
            f"Minimal checkpoint key mismatch in {path}; " + "; ".join(problems)
        )

    # Frozen base-model keys are intentionally absent; strict=False is safe
    # here because the exact trainable-key set was checked above.
    return model.load_state_dict(state_dict, strict=False)


def _qwen_projectors(trainer):
    projectors = []
    for attribute, filename in _QWEN_PROJECTOR_FILES:
        module = getattr(trainer, attribute, None)
        if module is not None:
            projectors.append((filename, trainer.unwrap_model(module)))
    return projectors


def save_qwen_projectors(trainer, output_dir):
    """Save every Qwen projector owned by a trainer."""
    projectors = _qwen_projectors(trainer)
    if not projectors:
        raise RuntimeError("Qwen is enabled, but no Qwen projector module is available to save")
    for filename, projector in projectors:
        torch.save(projector.state_dict(), os.path.join(output_dir, filename))


def load_qwen_projectors(trainer, input_dir):
    """Load every Qwen projector and require an exact state-dict match."""
    projectors = _qwen_projectors(trainer)
    if not projectors:
        raise RuntimeError("Qwen is enabled, but no Qwen projector module is available to load")

    for filename, projector in projectors:
        path = os.path.join(input_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Qwen projector checkpoint missing: {path}")
        state_dict = _load_state_dict(path)
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(f"Expected a state-dict mapping in Qwen projector checkpoint: {path}")
        try:
            projector.load_state_dict(state_dict, strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Qwen projector checkpoint key/shape mismatch in {path}: {exc}") from exc
