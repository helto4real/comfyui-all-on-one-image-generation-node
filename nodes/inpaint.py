"""AIO inpaint configuration node."""

from __future__ import annotations

from typing import Any

try:
    from ..services.inpaint import normalize_inpaint_config
except ImportError:  # pragma: no cover - direct test imports
    from services.inpaint import normalize_inpaint_config


class AIOInpaint:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("AIO_INPAINT_CONFIG",)
    RETURN_NAMES = ("inpaint",)
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
                        "default": 6,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Expand the sampled mask in latent space to reduce seams.",
                    },
                ),
                "mask_feather": (
                    "INT",
                    {
                        "default": 16,
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
            },
            "optional": {},
            "hidden": {},
        }

    def configure(
        self,
        image: Any = None,
        mask: Any = None,
        mask_invert: bool = False,
        mask_grow: int = 6,
        mask_feather: int = 16,
        denoise: float = 1.0,
        final_blend: bool = True,
    ):
        return (
            normalize_inpaint_config(
                image=image,
                mask=mask,
                mask_invert=mask_invert,
                mask_grow=mask_grow,
                mask_feather=mask_feather,
                denoise=denoise,
                final_blend=final_blend,
            ),
        )
