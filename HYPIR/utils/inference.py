"""Helpers shared by the standalone inference entry points."""

from collections.abc import Mapping
from typing import Any

import torch


def _preview(keys: list[str], limit: int = 8) -> str:
    """Keep checkpoint error messages useful without dumping huge key lists."""
    if len(keys) <= limit:
        return ", ".join(keys) or "<none>"
    return ", ".join(keys[:limit]) + f", ... ({len(keys)} total)"


def load_trainable_weights(
    model: torch.nn.Module,
    state_dict: Any,
    checkpoint_path: str,
) -> None:
    """Load a minimal/LoRA checkpoint without silently dropping bad weights.

    Training saves only parameters with ``requires_grad=True``. The frozen
    base-model parameters are therefore intentionally absent, but every
    trainable parameter must be present exactly once and have the expected
    shape. The already-loaded frozen base weights are merged back before the
    final ``strict=True`` load.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Checkpoint '{checkpoint_path}' must contain a state-dict mapping, "
            f"got {type(state_dict).__name__}."
        )

    state_dict = dict(state_dict)
    non_string_keys = sorted(str(key) for key in state_dict if not isinstance(key, str))
    if non_string_keys:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' contains non-string parameter keys: "
            f"{_preview(non_string_keys)}"
        )

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise RuntimeError(
            f"Model has no trainable parameters while loading '{checkpoint_path}'; "
            "the LoRA configuration may not have been attached."
        )

    checkpoint_keys = set(state_dict)
    expected_keys = set(trainable)
    missing = sorted(expected_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    invalid = sorted(
        key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)
    )
    shape_mismatches = sorted(
        f"{key}: checkpoint {tuple(state_dict[key].shape)} != model {tuple(trainable[key].shape)}"
        for key in expected_keys & checkpoint_keys
        if isinstance(state_dict[key], torch.Tensor)
        and state_dict[key].shape != trainable[key].shape
    )

    problems = []
    if missing:
        problems.append(f"missing trainable keys [{_preview(missing)}]")
    if unexpected:
        problems.append(f"unexpected keys [{_preview(unexpected)}]")
    if invalid:
        problems.append(f"non-tensor values [{_preview(invalid)}]")
    if shape_mismatches:
        problems.append(f"shape mismatches [{_preview(shape_mismatches)}]")
    if problems:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' does not match the inference model: "
            + "; ".join(problems)
        )

    # Frozen base-model keys are absent from the minimal checkpoint by design.
    # Merge them from the already-loaded base model so the actual load can use
    # PyTorch's strict=True validation as a final guard.
    complete_state_dict = model.state_dict()
    complete_state_dict.update(state_dict)
    try:
        model.load_state_dict(complete_state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' could not be loaded strictly."
        ) from exc
