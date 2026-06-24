"""Inpaint configuration and tensor helpers for AIO generation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    from .dimensions import ResolvedDimensions, infer_nearest_aspect_ratio, round_to_multiple
except ImportError:  # pragma: no cover - direct test imports
    from services.dimensions import ResolvedDimensions, infer_nearest_aspect_ratio, round_to_multiple


INPAINT_CONFIG_VERSION = 1
INPAINT_SIZE_MODE = "use inpaint image size"
CROP_INPAINT_CONTEXT_FACTOR = 1.6
CROP_INPAINT_OUTPUT_PADDING = "64"
CROP_INPAINT_DEVICE_MODE = "gpu (much faster)"


@dataclass(frozen=True)
class InpaintSource:
    image: Any
    mask: Any
    noise_mask: Any = None
    stitcher: Any = None
    used_crop: bool = False

    @property
    def sampling_mask(self) -> Any:
        return self.noise_mask if self.noise_mask is not None else self.mask


FluxInpaintSource = InpaintSource


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
        "mask_fill_holes": bool(source.get("mask_fill_holes", True)),
        "mask_hipass_filter": _validate_float(source.get("mask_hipass_filter", 0.1), "mask_hipass_filter", 0.0, 1.0),
        "context_from_mask_extend_factor": _validate_float(
            source.get("context_from_mask_extend_factor", CROP_INPAINT_CONTEXT_FACTOR),
            "context_from_mask_extend_factor",
            1.0,
            100.0,
        ),
        "crop_source_reference": bool(source.get("crop_source_reference", True)),
        "crop_downscale_algorithm": str(source.get("crop_downscale_algorithm", "bilinear")),
        "crop_upscale_algorithm": str(source.get("crop_upscale_algorithm", "bicubic")),
        "crop_output_padding": str(source.get("crop_output_padding", CROP_INPAINT_OUTPUT_PADDING)),
        "crop_device_mode": str(source.get("crop_device_mode", CROP_INPAINT_DEVICE_MODE)),
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
    latent = nodes.VAEEncode().encode(vae, image)[0]
    latent = latent.copy()
    latent["noise_mask"] = grow_inpaint_mask(mask, int(normalized["mask_grow"])).reshape(
        (-1, 1, mask.shape[-2], mask.shape[-1])
    )
    return latent, image, mask


def prepare_inpaint_source(
    *,
    config: Mapping[str, Any],
    width: int,
    height: int,
) -> InpaintSource:
    normalized = normalize_inpaint_config(config)
    crop_cls = _optional_node_class("InpaintCropImproved")
    if crop_cls is not None:
        stitcher, cropped_image, cropped_mask = crop_cls().inpaint_crop(
            image=normalized["image"],
            downscale_algorithm=normalized["crop_downscale_algorithm"],
            upscale_algorithm=normalized["crop_upscale_algorithm"],
            preresize=False,
            preresize_mode="ensure minimum resolution",
            preresize_min_width=1024,
            preresize_min_height=1024,
            preresize_max_width=16384,
            preresize_max_height=16384,
            extend_for_outpainting=False,
            extend_up_factor=1.0,
            extend_down_factor=1.0,
            extend_left_factor=1.0,
            extend_right_factor=1.0,
            mask_hipass_filter=float(normalized["mask_hipass_filter"]),
            mask_fill_holes=bool(normalized["mask_fill_holes"]),
            mask_expand_pixels=int(normalized["mask_grow"]),
            mask_invert=bool(normalized["mask_invert"]),
            mask_blend_pixels=int(normalized["mask_feather"]),
            context_from_mask_extend_factor=float(normalized["context_from_mask_extend_factor"]),
            output_resize_to_target_size=True,
            output_target_width=int(width),
            output_target_height=int(height),
            output_padding=str(normalized["crop_output_padding"]),
            device_mode=str(normalized["crop_device_mode"]),
            mask=normalized["mask"],
            optional_context_mask=config.get("context_mask"),
        )[:3]
        return InpaintSource(
            image=cropped_image,
            mask=cropped_mask,
            noise_mask=cropped_mask,
            stitcher=stitcher,
            used_crop=True,
        )

    image = resize_image_to_dimensions(normalized["image"], width=width, height=height)
    mask = prepare_inpaint_mask(normalized, width=width, height=height)
    noise_mask = grow_inpaint_mask(mask, int(normalized["mask_grow"]))
    return InpaintSource(image=image, mask=mask, noise_mask=noise_mask)


def prepare_flux_inpaint_source(
    *,
    config: Mapping[str, Any],
    width: int,
    height: int,
) -> InpaintSource:
    return prepare_inpaint_source(config=config, width=width, height=height)


def encode_inpaint_source_latent(
    *,
    vae: Any,
    source: InpaintSource,
) -> dict[str, Any]:
    import nodes  # type: ignore

    mask = source.sampling_mask
    latent = nodes.VAEEncode().encode(vae, source.image)[0]
    latent = latent.copy()
    latent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    return latent


def apply_inpaint_model_conditioning(
    *,
    vae: Any,
    positive: Any,
    negative: Any,
    image: Any,
    mask: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    node_cls = _required_node_class("InpaintModelConditioning")
    return node_cls().encode(
        positive=positive,
        negative=negative,
        pixels=image,
        vae=vae,
        mask=mask,
        noise_mask=True,
    )


def stitch_inpaint_image(*, stitcher: Any, inpainted_image: Any) -> Any:
    node_cls = _required_node_class("InpaintStitchImproved")
    return node_cls().inpaint_stitch(stitcher=stitcher, inpainted_image=inpainted_image)[0]


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


def grow_inpaint_mask(mask: Any, grow_mask_by: int):
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    amount = int(grow_mask_by)
    prepared = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).round()
    if amount <= 0:
        return torch.clamp(prepared[:, 0, :, :], 0.0, 1.0)
    kernel = torch.ones((1, 1, amount, amount), dtype=prepared.dtype, device=prepared.device)
    padding = math.ceil((amount - 1) / 2)
    grown = torch.clamp(F.conv2d(prepared, kernel, padding=padding), 0, 1)
    return grown[:, :, : mask.shape[-2], : mask.shape[-1]][:, 0, :, :]


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


def _optional_node_class(name: str):
    import nodes  # type: ignore

    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    if name in mappings:
        return mappings[name]
    return getattr(nodes, name, None)


def _required_node_class(name: str):
    node_cls = _optional_node_class(name)
    if node_cls is None:
        raise ValueError(f"{name} is required for this inpaint path.")
    return node_cls


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
