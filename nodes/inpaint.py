"""AIO inpaint configuration node."""

from __future__ import annotations

from typing import Any

try:
    from ..services.inpaint import normalize_inpaint_config, prepare_inpaint_output_mask
except ImportError:  # pragma: no cover - direct test imports
    from services.inpaint import normalize_inpaint_config, prepare_inpaint_output_mask


class AIOInpaint:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("AIO_INPAINT_CONFIG", "MASK")
    RETURN_NAMES = ("inpaint", "final_mask")
    FUNCTION = "configure"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Source image to preserve outside the inpaint mask."},
                ),
                "mask": (
                    "MASK",
                    {"tooltip": "White areas are regenerated; black areas are preserved."},
                ),
                "mask_invert": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Invert the mask before inpainting."},
                ),
                "mask_grow": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Expand the sampled mask in latent space to reduce seams.",
                    },
                ),
                "mask_feather": (
                    "INT",
                    {
                        "default": 24,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Soften the final image blend edge in pixels.",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How strongly to redraw the masked area.",
                    },
                ),
                "final_blend": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Composite the decoded result over the source image using the feathered mask.",
                    },
                ),
                "crop_target_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 16384,
                        "step": 1,
                        "advanced": True,
                        "tooltip": "Working crop width for optional crop/stitch inpaint.",
                    },
                ),
                "crop_target_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 16384,
                        "step": 1,
                        "advanced": True,
                        "tooltip": "Working crop height for optional crop/stitch inpaint.",
                    },
                ),
                "context_from_mask_extend_factor": (
                    "FLOAT",
                    {
                        "default": 1.6,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.01,
                        "advanced": True,
                        "tooltip": "Grow the crop context area around the processed mask.",
                    },
                ),
                "crop_output_padding": (
                    ["0", "8", "16", "32", "64", "128", "256", "512"],
                    {
                        "default": "64",
                        "advanced": True,
                        "tooltip": "Pad crop target dimensions to this multiple.",
                    },
                ),
                "mask_fill_holes": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "advanced": True,
                        "tooltip": "Fill enclosed holes in the crop/stitch mask.",
                    },
                ),
                "mask_hipass_filter": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "advanced": True,
                        "tooltip": "Ignore low mask values before crop/stitch processing.",
                    },
                ),
                "max_full_frame_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 1024.0,
                        "step": 0.25,
                        "advanced": True,
                        "tooltip": "Maximum full-frame fallback size in megapixels when crop/stitch is unavailable.",
                    },
                ),
                "max_full_frame_side": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 64,
                        "max": 16384,
                        "step": 1,
                        "advanced": True,
                        "tooltip": "Maximum full-frame fallback long side when crop/stitch is unavailable.",
                    },
                ),
            },
            "optional": {
                "context_mask": (
                    "MASK",
                    {"tooltip": "Optional extra area to include as crop/stitch context."},
                ),
            },
            "hidden": {},
        }

    def configure(
        self,
        image: Any = None,
        mask: Any = None,
        mask_invert: bool = False,
        mask_grow: int = 16,
        mask_feather: int = 24,
        denoise: float = 1.0,
        final_blend: bool = True,
        crop_target_width: int = 1024,
        crop_target_height: int = 1024,
        context_from_mask_extend_factor: float = 1.6,
        crop_output_padding: str = "64",
        mask_fill_holes: bool = True,
        mask_hipass_filter: float = 0.1,
        max_full_frame_megapixels: float = 1.0,
        max_full_frame_side: int = 1536,
        context_mask: Any = None,
    ):
        config = normalize_inpaint_config(
            image=image,
            mask=mask,
            mask_invert=mask_invert,
            mask_grow=mask_grow,
            mask_feather=mask_feather,
            denoise=denoise,
            final_blend=final_blend,
            context_mask=context_mask,
            crop_target_width=crop_target_width,
            crop_target_height=crop_target_height,
            context_from_mask_extend_factor=context_from_mask_extend_factor,
            crop_output_padding=crop_output_padding,
            mask_fill_holes=mask_fill_holes,
            mask_hipass_filter=mask_hipass_filter,
            max_full_frame_megapixels=max_full_frame_megapixels,
            max_full_frame_side=max_full_frame_side,
        )
        return (config, prepare_inpaint_output_mask(config))
