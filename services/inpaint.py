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
CROP_INPAINT_MASK_GROW_PERCENT = 8.0
CROP_INPAINT_MASK_GROW_MAX_PIXELS = 1024
CROP_INPAINT_MASK_FEATHER = 24
CROP_INPAINT_TARGET_WIDTH = 1024
CROP_INPAINT_TARGET_HEIGHT = 1024
CROP_INPAINT_CONTEXT_FACTOR = 1.6
CROP_INPAINT_OUTPUT_PADDING = "64"
CROP_INPAINT_DEVICE_MODE = "gpu (much faster)"
CROP_INPAINT_MAX_FULL_FRAME_MEGAPIXELS = 1.0
CROP_INPAINT_MAX_FULL_FRAME_SIDE = 1536
CPU_CROP_DEVICE_MODE = "cpu (compatible)"


@dataclass(frozen=True)
class InpaintSource:
    image: Any
    mask: Any
    noise_mask: Any = None
    stitcher: Any = None
    used_crop: bool = False
    width: int | None = None
    height: int | None = None

    @property
    def sampling_mask(self) -> Any:
        return self.noise_mask if self.noise_mask is not None else self.mask

    def working_dimensions(self, *, fallback_width: int, fallback_height: int) -> tuple[int, int]:
        return (
            int(self.width) if self.width is not None else int(fallback_width),
            int(self.height) if self.height is not None else int(fallback_height),
        )


FluxInpaintSource = InpaintSource


def normalize_inpaint_config(
    config: Mapping[str, Any] | None = None,
    *,
    image: Any = None,
    mask: Any = None,
    mask_invert: bool | None = None,
    mask_grow_percent: float | None = None,
    mask_feather: int | None = None,
    denoise: float | None = None,
    final_blend: bool | None = None,
    context_mask: Any = None,
    crop_target_width: int | None = None,
    crop_target_height: int | None = None,
    context_from_mask_extend_factor: float | None = None,
    crop_output_padding: str | None = None,
    mask_fill_holes: bool | None = None,
    mask_hipass_filter: float | None = None,
    max_full_frame_megapixels: float | None = None,
    max_full_frame_side: int | None = None,
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
        "mask_grow_percent": _validate_float(
            _get_value(source, "mask_grow_percent", mask_grow_percent, CROP_INPAINT_MASK_GROW_PERCENT),
            "mask_grow_percent",
            0.0,
            100.0,
        ),
        "mask_feather": _validate_int(
            _get_value(source, "mask_feather", mask_feather, CROP_INPAINT_MASK_FEATHER),
            "mask_feather",
            0,
            256,
        ),
        "denoise": _validate_float(_get_value(source, "denoise", denoise, 1.0), "denoise", 0.0, 1.0),
        "final_blend": bool(_get_value(source, "final_blend", final_blend, True)),
        "context_mask": context_mask if context_mask is not None else source.get("context_mask"),
        "crop_target_width": _validate_int(
            _get_value(source, "crop_target_width", crop_target_width, CROP_INPAINT_TARGET_WIDTH),
            "crop_target_width",
            64,
            16384,
        ),
        "crop_target_height": _validate_int(
            _get_value(source, "crop_target_height", crop_target_height, CROP_INPAINT_TARGET_HEIGHT),
            "crop_target_height",
            64,
            16384,
        ),
        "mask_fill_holes": bool(_get_value(source, "mask_fill_holes", mask_fill_holes, True)),
        "mask_hipass_filter": _validate_float(
            _get_value(source, "mask_hipass_filter", mask_hipass_filter, 0.1),
            "mask_hipass_filter",
            0.0,
            1.0,
        ),
        "context_from_mask_extend_factor": _validate_float(
            _get_value(
                source,
                "context_from_mask_extend_factor",
                context_from_mask_extend_factor,
                CROP_INPAINT_CONTEXT_FACTOR,
            ),
            "context_from_mask_extend_factor",
            1.0,
            100.0,
        ),
        "crop_source_reference": bool(source.get("crop_source_reference", True)),
        "crop_downscale_algorithm": str(source.get("crop_downscale_algorithm", "bilinear")),
        "crop_upscale_algorithm": str(source.get("crop_upscale_algorithm", "bicubic")),
        "crop_output_padding": str(
            _get_value(source, "crop_output_padding", crop_output_padding, CROP_INPAINT_OUTPUT_PADDING)
        ),
        "crop_device_mode": str(source.get("crop_device_mode", CROP_INPAINT_DEVICE_MODE)),
        "max_full_frame_megapixels": _validate_float(
            _get_value(
                source,
                "max_full_frame_megapixels",
                max_full_frame_megapixels,
                CROP_INPAINT_MAX_FULL_FRAME_MEGAPIXELS,
            ),
            "max_full_frame_megapixels",
            0.25,
            1024.0,
        ),
        "max_full_frame_side": _validate_int(
            _get_value(
                source,
                "max_full_frame_side",
                max_full_frame_side,
                CROP_INPAINT_MAX_FULL_FRAME_SIDE,
            ),
            "max_full_frame_side",
            64,
            16384,
        ),
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


def _image_dimensions(image: Any, *, fallback_width: int, fallback_height: int) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        return int(fallback_width), int(fallback_height)
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
    grow_pixels = resolve_mask_grow_pixels(normalized, mask=mask, width=width, height=height)
    latent["noise_mask"] = grow_inpaint_mask(mask, grow_pixels).reshape(
        (-1, 1, mask.shape[-2], mask.shape[-1])
    )
    return latent, image, mask


def prepare_inpaint_source(
    *,
    config: Mapping[str, Any],
    width: int,
    height: int,
    force_cpu_crop: bool = False,
    allow_full_frame_downscale: bool = True,
) -> InpaintSource:
    normalized = normalize_inpaint_config(config)
    crop_target_width = int(normalized["crop_target_width"])
    crop_target_height = int(normalized["crop_target_height"])
    crop_cls = _optional_node_class("InpaintCropImproved")
    if crop_cls is not None:
        source_input_width, source_input_height = _image_dimensions(
            normalized["image"],
            fallback_width=int(width),
            fallback_height=int(height),
        )
        mask_grow_pixels = resolve_mask_grow_pixels(
            normalized,
            width=source_input_width,
            height=source_input_height,
        )
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
            mask_expand_pixels=mask_grow_pixels,
            mask_invert=bool(normalized["mask_invert"]),
            mask_blend_pixels=int(normalized["mask_feather"]),
            context_from_mask_extend_factor=float(normalized["context_from_mask_extend_factor"]),
            output_resize_to_target_size=True,
            output_target_width=crop_target_width,
            output_target_height=crop_target_height,
            output_padding=str(normalized["crop_output_padding"]),
            device_mode=_resolve_crop_device_mode(normalized, force_cpu=force_cpu_crop),
            mask=normalized["mask"],
            optional_context_mask=normalized.get("context_mask"),
        )[:3]
        source_width, source_height = _image_dimensions(
            cropped_image,
            fallback_width=crop_target_width,
            fallback_height=crop_target_height,
        )
        return InpaintSource(
            image=cropped_image,
            mask=cropped_mask,
            noise_mask=cropped_mask,
            stitcher=stitcher,
            used_crop=True,
            width=source_width,
            height=source_height,
        )

    source_width, source_height = _fallback_full_frame_dimensions(
        normalized,
        width=int(width),
        height=int(height),
        allow_downscale=allow_full_frame_downscale,
    )
    image = resize_image_to_dimensions(normalized["image"], width=source_width, height=source_height)
    mask = prepare_inpaint_mask(normalized, width=source_width, height=source_height)
    grow_pixels = resolve_mask_grow_pixels(normalized, mask=mask, width=source_width, height=source_height)
    noise_mask = grow_inpaint_mask(mask, grow_pixels)
    return InpaintSource(
        image=image,
        mask=mask,
        noise_mask=noise_mask,
        width=source_width,
        height=source_height,
    )


def prepare_inpaint_output_mask(config: Mapping[str, Any]) -> Any:
    normalized = normalize_inpaint_config(config)
    width, height = inpaint_image_dimensions(normalized)
    source = prepare_inpaint_source(
        config=normalized,
        width=width,
        height=height,
        force_cpu_crop=True,
        allow_full_frame_downscale=False,
    )
    if source.used_crop:
        blend_mask = stitcher_blend_mask(source.stitcher, restore_original_size=True)
        return blend_mask if blend_mask is not None else source.mask
    return feather_inpaint_mask(source.sampling_mask, int(normalized["mask_feather"]))


def stitcher_blend_mask(stitcher: Any, *, restore_original_size: bool = False) -> Any:
    if not isinstance(stitcher, Mapping):
        return None
    masks = stitcher.get("cropped_mask_for_blend")
    if masks is None:
        return None
    if hasattr(masks, "shape"):
        normalized = [masks.reshape((-1, masks.shape[-2], masks.shape[-1]))]
    elif isinstance(masks, (list, tuple)) and masks:
        normalized = [mask.reshape((-1, mask.shape[-2], mask.shape[-1])) for mask in masks]
    else:
        return None

    if restore_original_size:
        restored = [_restore_stitcher_mask_to_original(stitcher, mask, index) for index, mask in enumerate(normalized)]
        restored = [mask for mask in restored if mask is not None]
        if not restored:
            return None
        if len(restored) == 1:
            return restored[0]

        import torch  # type: ignore

        return torch.cat(restored, dim=0)

    if len(normalized) == 1:
        return normalized[0]

    import torch  # type: ignore

    return torch.cat(normalized, dim=0)


def _restore_stitcher_mask_to_original(stitcher: Mapping[str, Any], mask: Any, index: int) -> Any:
    import torch  # type: ignore

    ctc_x = _stitcher_int(stitcher, "cropped_to_canvas_x", index)
    ctc_y = _stitcher_int(stitcher, "cropped_to_canvas_y", index)
    ctc_w = _stitcher_int(stitcher, "cropped_to_canvas_w", index)
    ctc_h = _stitcher_int(stitcher, "cropped_to_canvas_h", index)
    cto_x = _stitcher_int(stitcher, "canvas_to_orig_x", index)
    cto_y = _stitcher_int(stitcher, "canvas_to_orig_y", index)
    cto_w = _stitcher_int(stitcher, "canvas_to_orig_w", index)
    cto_h = _stitcher_int(stitcher, "canvas_to_orig_h", index)
    if None in {ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h}:
        return None

    canvas = _stitcher_item(stitcher, "canvas_image", index)
    if hasattr(canvas, "shape") and len(canvas.shape) >= 3:
        canvas_h = int(canvas.shape[1])
        canvas_w = int(canvas.shape[2])
    else:
        canvas_w = max(int(ctc_x) + int(ctc_w), int(cto_x) + int(cto_w))
        canvas_h = max(int(ctc_y) + int(ctc_h), int(cto_y) + int(cto_h))

    resized = _resize_mask_to_dimensions(
        mask,
        width=int(ctc_w),
        height=int(ctc_h),
        algorithm=_stitcher_resize_algorithm(stitcher, mask, int(ctc_w), int(ctc_h)),
    )
    canvas_mask = torch.zeros(
        (int(resized.shape[0]), canvas_h, canvas_w),
        dtype=resized.dtype,
        device=resized.device,
    )

    dst_x0 = max(0, int(ctc_x))
    dst_y0 = max(0, int(ctc_y))
    dst_x1 = min(canvas_w, int(ctc_x) + int(ctc_w))
    dst_y1 = min(canvas_h, int(ctc_y) + int(ctc_h))
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return canvas_mask[:, int(cto_y) : int(cto_y) + int(cto_h), int(cto_x) : int(cto_x) + int(cto_w)]

    src_x0 = dst_x0 - int(ctc_x)
    src_y0 = dst_y0 - int(ctc_y)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    canvas_mask[:, dst_y0:dst_y1, dst_x0:dst_x1] = resized[:, src_y0:src_y1, src_x0:src_x1]
    return canvas_mask[:, int(cto_y) : int(cto_y) + int(cto_h), int(cto_x) : int(cto_x) + int(cto_w)]


def _resize_mask_to_dimensions(mask: Any, *, width: int, height: int, algorithm: str) -> Any:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    prepared = mask.float().reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    mode = str(algorithm).lower()
    if int(prepared.shape[-1]) == int(width) and int(prepared.shape[-2]) == int(height):
        return torch.clamp(prepared[:, 0, :, :], 0.0, 1.0)
    if mode == "nearest":
        resized = F.interpolate(prepared, size=(int(height), int(width)), mode="nearest")
    elif mode in {"bilinear", "bicubic"}:
        resized = F.interpolate(prepared, size=(int(height), int(width)), mode=mode, align_corners=False)
    elif mode == "area":
        resized = F.interpolate(prepared, size=(int(height), int(width)), mode="area")
    else:
        resized = F.interpolate(prepared, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    return torch.clamp(resized[:, 0, :, :], 0.0, 1.0)


def _stitcher_resize_algorithm(stitcher: Mapping[str, Any], mask: Any, width: int, height: int) -> str:
    if int(width) > int(mask.shape[-1]) or int(height) > int(mask.shape[-2]):
        return str(stitcher.get("upscale_algorithm", "bicubic"))
    return str(stitcher.get("downscale_algorithm", "bilinear"))


def _stitcher_int(stitcher: Mapping[str, Any], key: str, index: int) -> int | None:
    value = _stitcher_item(stitcher, key, index)
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _stitcher_item(stitcher: Mapping[str, Any], key: str, index: int) -> Any:
    value = stitcher.get(key)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if index < len(value):
            return value[index]
        return value[0]
    return value


def prepare_flux_inpaint_source(
    *,
    config: Mapping[str, Any],
    width: int,
    height: int,
) -> InpaintSource:
    return prepare_inpaint_source(config=config, width=width, height=height)


def inpaint_full_frame_downscale_warning(
    config: Mapping[str, Any] | None,
    *,
    width: int,
    height: int,
) -> str | None:
    if config is None or _optional_node_class("InpaintCropImproved") is not None:
        return None
    normalized = normalize_inpaint_config(config)
    limited_width, limited_height = _fallback_full_frame_dimensions(
        normalized,
        width=int(width),
        height=int(height),
        allow_downscale=True,
    )
    if limited_width == int(width) and limited_height == int(height):
        return None
    return (
        "AIO Inpaint crop/stitch is unavailable; full-frame inpaint input will be "
        f"downscaled from {int(width)}x{int(height)} to {limited_width}x{limited_height} "
        "to reduce Flux VRAM use."
    )


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


def resolve_mask_grow_pixels(
    config: Mapping[str, Any],
    *,
    mask: Any | None = None,
    width: int | None = None,
    height: int | None = None,
) -> int:
    percent = float(config.get("mask_grow_percent", CROP_INPAINT_MASK_GROW_PERCENT))
    if percent <= 0.0:
        return 0

    source_mask = mask if mask is not None else config.get("mask")
    threshold = float(config.get("mask_hipass_filter", 0.0))
    bbox_side = _mask_active_bbox_max_side(source_mask, threshold=threshold)
    if bbox_side is None:
        bbox_side = _mask_shape_max_side(source_mask)
    if bbox_side is None and width is not None and height is not None:
        bbox_side = max(int(width), int(height))
    if bbox_side is None or bbox_side <= 0:
        return 0

    grow = int(math.ceil(float(bbox_side) * percent / 100.0))
    return max(0, min(grow, CROP_INPAINT_MASK_GROW_MAX_PIXELS))


def _mask_active_bbox_max_side(mask: Any, *, threshold: float) -> int | None:
    if not hasattr(mask, "reshape") or not hasattr(mask, "float"):
        return None
    shape = getattr(mask, "shape", None)
    if shape is None or len(shape) < 2:
        return None

    import torch  # type: ignore

    prepared = mask.float().reshape((-1, shape[-2], shape[-1]))
    active = prepared >= float(threshold) if threshold > 0.0 else prepared > 0.0
    if not torch.any(active):
        return 0

    max_side = 0
    for batch_mask in active:
        coordinates = torch.nonzero(batch_mask, as_tuple=False)
        if int(coordinates.numel()) == 0:
            continue
        y_values = coordinates[:, 0]
        x_values = coordinates[:, 1]
        bbox_width = int(x_values.max().item() - x_values.min().item() + 1)
        bbox_height = int(y_values.max().item() - y_values.min().item() + 1)
        max_side = max(max_side, bbox_width, bbox_height)
    return max_side


def _mask_shape_max_side(mask: Any) -> int | None:
    shape = getattr(mask, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return max(int(shape[-1]), int(shape[-2]))


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


def feather_inpaint_mask(mask: Any, feather: int):
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    prepared = mask.float().reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    radius = int(feather)
    if radius <= 0:
        return torch.clamp(prepared[:, 0, :, :], 0.0, 1.0)
    kernel = radius * 2 + 1
    feathered = F.avg_pool2d(prepared, kernel_size=kernel, stride=1, padding=radius)
    return torch.clamp(feathered[:, 0, :, :], 0.0, 1.0)


def resize_image_to_dimensions(image: Any, *, width: int, height: int):
    if int(image.shape[2]) == int(width) and int(image.shape[1]) == int(height):
        return image

    import comfy.utils  # type: ignore

    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, int(width), int(height), "bilinear", "center")
    return resized.movedim(1, -1)


def _resolve_crop_device_mode(config: Mapping[str, Any], *, force_cpu: bool = False) -> str:
    if force_cpu:
        return CPU_CROP_DEVICE_MODE
    configured = str(config.get("crop_device_mode", CROP_INPAINT_DEVICE_MODE))
    if configured == CPU_CROP_DEVICE_MODE:
        return CPU_CROP_DEVICE_MODE
    return CROP_INPAINT_DEVICE_MODE


def _fallback_full_frame_dimensions(
    config: Mapping[str, Any],
    *,
    width: int,
    height: int,
    allow_downscale: bool,
) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    if not allow_downscale:
        return width, height

    max_megapixels = float(config.get("max_full_frame_megapixels", CROP_INPAINT_MAX_FULL_FRAME_MEGAPIXELS))
    max_side = int(config.get("max_full_frame_side", CROP_INPAINT_MAX_FULL_FRAME_SIDE))
    max_pixels = max_megapixels * 1024 * 1024
    scale = 1.0
    current_pixels = max(1, width * height)
    if max_pixels > 0 and current_pixels > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / current_pixels))
    if max_side > 0 and max(width, height) > max_side:
        scale = min(scale, max_side / max(width, height))
    if scale >= 1.0:
        return width, height

    return (
        _floor_to_multiple(width * scale, 16),
        _floor_to_multiple(height * scale, 16),
    )


def _floor_to_multiple(value: float, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return max(multiple, int(math.floor(float(value) / multiple)) * multiple)


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
    try:
        import nodes  # type: ignore
    except ImportError:
        return None

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
