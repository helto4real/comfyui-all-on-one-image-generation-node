"""Krea 2 conditioning rebalance helpers.

Adapted from nova452/ComfyUI-ConditioningKrea2Rebalance under Apache-2.0.
Imports torch lazily so this node pack remains lightweight at import time.
"""

from __future__ import annotations

from typing import Any


DEFAULT_KREA2_REBALANCE_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"


def parse_per_layer_weights(value: str | None) -> list[float] | None:
    """Parse comma/semicolon separated layer weights; invalid input disables layer weighting."""
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        weights = [float(part) for part in value.replace(";", ",").split(",") if part.strip() != ""]
    except ValueError:
        return None
    if len(weights) < 2:
        return None
    return weights


def _scale_cond_tensor(tensor: Any, multiplier: float, per_layer_weights: list[float] | None, torch_module: Any) -> Any:
    if per_layer_weights is None:
        return tensor * multiplier

    flat = tensor.shape[-1]
    layer_count = len(per_layer_weights)
    if layer_count > 1 and flat % layer_count == 0:
        layer_dim = flat // layer_count
        original_dtype = tensor.dtype
        tensor = tensor.float()
        tensor = tensor.view(*tensor.shape[:-1], layer_count, layer_dim)
        gains = torch_module.tensor(per_layer_weights, dtype=tensor.dtype, device=tensor.device)
        tensor = tensor * gains.view(*([1] * (tensor.dim() - 2)), layer_count, 1)
        tensor = tensor.view(*tensor.shape[:-2], flat)
        return tensor.to(original_dtype) * multiplier
    return tensor * multiplier


def rebalance_conditioning(
    structure: Any,
    *,
    multiplier: float,
    per_layer_weights: str | list[float] | None = None,
) -> Any:
    import torch  # type: ignore

    weights = (
        parse_per_layer_weights(per_layer_weights)
        if isinstance(per_layer_weights, str) or per_layer_weights is None
        else per_layer_weights
    )
    return _rebalance_conditioning(
        structure,
        multiplier=float(multiplier),
        per_layer_weights=weights,
        torch_module=torch,
    )


def _rebalance_conditioning(
    structure: Any,
    *,
    multiplier: float,
    per_layer_weights: list[float] | None,
    torch_module: Any,
) -> Any:
    if isinstance(structure, list):
        out = []
        for item in structure:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], torch_module.Tensor)
                and isinstance(item[1], dict)
            ):
                cond_tensor, extras = item
                out.append([
                    _scale_cond_tensor(cond_tensor, multiplier, per_layer_weights, torch_module),
                    dict(extras),
                ])
            else:
                out.append(
                    _rebalance_conditioning(
                        item,
                        multiplier=multiplier,
                        per_layer_weights=per_layer_weights,
                        torch_module=torch_module,
                    )
                )
        return out
    if isinstance(structure, torch_module.Tensor):
        return _scale_cond_tensor(structure, multiplier, per_layer_weights, torch_module)
    if isinstance(structure, dict):
        return {
            key: _rebalance_conditioning(
                value,
                multiplier=multiplier,
                per_layer_weights=per_layer_weights,
                torch_module=torch_module,
            )
            for key, value in structure.items()
        }
    return structure
