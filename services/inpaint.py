"""Inpaint configuration and tensor helpers for AIO generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from .dimensions import ResolvedDimensions, infer_nearest_aspect_ratio, round_to_multiple
except ImportError:  # pragma: no cover - direct test imports
    from services.dimensions import ResolvedDimensions, infer_nearest_aspect_ratio, round_to_multiple


INPAINT_CONFIG_VERSION = 1
INPAINT_SIZE_MODE = "use inpaint image size"


def normalize_inpaint_config(
    config: Mapping[str, Any] | None = None,
    *,
    image: Any = None,
    mask: Any = None,
    mask_invert: bool | None = None,
    mask_grow: int | None = None,
    mask_feather: int | None = None,
    denoise: float | None = None,
    final_blend: bool | None = None,
) -> dict[str, Any]:
    source = dict(config or {})
    resolved_image = image if image is not None else source.get("image")
    resolved_mask = mask if mask is not None else source.get("mask")
    if resolved_image is None:
        raise ValueError("inpaint image is required.")
    if resolved_mask is None:
        raise ValueError("inpaint mask is required.")

    return {
        "version": INPAINT_CONFIG_VERSION,
        "image": resolved_image,
        "mask": resolved_mask,
        "mask_invert": bool(_get_value(source, "mask_invert", mask_invert, False)),
        "mask_grow": _validate_int(_get_value(source, "mask_grow", mask_grow, 6), "mask_grow", 0, 64),
        "mask_feather": _validate_int(_get_value(source, "mask_feather", mask_feather, 16), "mask_feather", 0, 256),
        "denoise": _validate_float(_get_value(source, "denoise", denoise, 1.0), "denoise", 0.0, 1.0),
        "final_blend": bool(_get_value(source, "final_blend", final_blend, True)),
    }


def normalize_optional_inpaint_config(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return normalize_inpaint_config(config)


def inpaint_image_dimensions(config: Mapping[str, Any]) -> tuple[int, int]:
    image = config["image"]
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        raise ValueError("inpaint image must be an IMAGE tensor with shape [B, H, W, C].")
    return int(shape[2]), int(shape[1])


def resolve_dimensions_from_inpaint_config(
    config: Mapping[str, Any],
    *,
    multiple: int,
) -> ResolvedDimensions:
    width, height = inpaint_image_dimensions(config)
    resolved_width = round_to_multiple(float(width), multiple)
    resolved_height = round_to_multiple(float(height), multiple)
    return ResolvedDimensions(
        width=resolved_width,
        height=resolved_height,
        max_side=max(resolved_width, resolved_height),
        aspect_ratio=infer_nearest_aspect_ratio(resolved_width, resolved_height),
        size_mode=INPAINT_SIZE_MODE,
        multiple_value=str(multiple),
    )


def prepare_inpaint_latent(
    *,
    vae: Any,
    config: Mapping[str, Any],
    width: int,
    height: int,
) -> tuple[dict[str, Any], Any, Any]:
    import nodes  # type: ignore

    normalized = normalize_inpaint_config(config)
    image = resize_image_to_dimensions(normalized["image"], width=width, height=height)
    mask = prepare_inpaint_mask(normalized, width=width, height=height)
    latent = nodes.VAEEncodeForInpaint().encode(
        vae,
        image,
        mask,
        int(normalized["mask_grow"]),
    )[0]
    return latent, image, mask


def prepare_inpaint_mask(config: Mapping[str, Any], *, width: int, height: int):
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    mask = config["mask"]
    if not hasattr(mask, "reshape"):
        raise ValueError("inpaint mask must be a MASK tensor.")
    mask = mask.float()
    if bool(config.get("mask_invert", False)):
        mask = 1.0 - mask
    mask = torch.clamp(mask, 0.0, 1.0)
    mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    if int(mask.shape[-1]) != int(width) or int(mask.shape[-2]) != int(height):
        mask = F.interpolate(mask, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    return torch.clamp(mask[:, 0, :, :], 0.0, 1.0)


def resize_image_to_dimensions(image: Any, *, width: int, height: int):
    if int(image.shape[2]) == int(width) and int(image.shape[1]) == int(height):
        return image

    import comfy.utils  # type: ignore

    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, int(width), int(height), "bilinear", "center")
    return resized.movedim(1, -1)


def apply_denoise_to_sigmas(sigmas: Any, denoise: float):
    value = float(denoise)
    if value >= 1.0:
        return sigmas
    shape = getattr(sigmas, "shape", None)
    sigma_count = int(shape[-1]) if shape is not None else len(sigmas)
    step_count = max(1, sigma_count - 1)
    keep_steps = max(1, int(round(step_count * max(0.0, value))))
    return sigmas[-(keep_steps + 1):]


def blend_inpaint_image(
    *,
    source_image: Any,
    generated_image: Any,
    mask: Any,
    feather: int,
) -> Any:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    height = int(generated_image.shape[1])
    width = int(generated_image.shape[2])
    source = resize_image_to_dimensions(source_image, width=width, height=height)
    source = source.to(device=generated_image.device, dtype=generated_image.dtype)
    blend_mask = mask.to(device=generated_image.device, dtype=generated_image.dtype)
    blend_mask = blend_mask.reshape((-1, 1, blend_mask.shape[-2], blend_mask.shape[-1]))
    if int(blend_mask.shape[-1]) != width or int(blend_mask.shape[-2]) != height:
        blend_mask = F.interpolate(blend_mask, size=(height, width), mode="bilinear", align_corners=False)
    radius = int(feather)
    if radius > 0:
        kernel = radius * 2 + 1
        blend_mask = F.avg_pool2d(blend_mask, kernel_size=kernel, stride=1, padding=radius)
    blend_mask = torch.clamp(blend_mask.movedim(1, -1), 0.0, 1.0)
    if int(blend_mask.shape[0]) == 1 and int(generated_image.shape[0]) > 1:
        blend_mask = blend_mask.repeat((int(generated_image.shape[0]), 1, 1, 1))
    if int(source.shape[0]) == 1 and int(generated_image.shape[0]) > 1:
        source = source.repeat((int(generated_image.shape[0]), 1, 1, 1))
    return generated_image * blend_mask + source * (1.0 - blend_mask)


def _get_value(source: Mapping[str, Any], key: str, explicit: Any, default: Any) -> Any:
    if explicit is not None:
        return explicit
    return source.get(key, default)


def _validate_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    resolved = int(value)
    if resolved < minimum or resolved > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return resolved


def _validate_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    resolved = float(value)
    if resolved < minimum or resolved > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return resolved
