"""Classic ComfyUI settings node for FLUX.2 Klein 9B."""

from __future__ import annotations


REFERENCE_UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


class AIOFlux2Klein9BSettings:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("AIO_MODEL_SETTINGS",)
    FUNCTION = "build_settings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": (
                    ["distilled", "base"],
                    {"tooltip": "Select the FLUX.2 Klein 9B variant. Distilled uses the fast low-step defaults."},
                ),
                "guidance": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Model guidance value passed to FLUX. Higher values follow the prompt more tightly.",
                    },
                ),
                "reference_strength": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How strongly connected reference images influence the generated image.",
                    },
                ),
                "precision_policy": (
                    ["auto", "fp8", "bf16"],
                    {"tooltip": "Model precision preference. Auto chooses a practical format for the current runtime."},
                ),
                "memory_policy": (
                    ["auto", "low_vram", "balanced", "high_vram"],
                    {"tooltip": "Memory strategy for loading and running the model on your hardware."},
                ),
                "base_shift": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Base timestep shift used by the FLUX sampler schedule.",
                    },
                ),
                "max_shift": (
                    "FLOAT",
                    {
                        "default": 1.15,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Maximum timestep shift used by the FLUX sampler schedule.",
                    },
                ),
                "reference_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 16.0,
                        "step": 0.01,
                        "tooltip": "Target megapixels for resizing each reference image before encoding.",
                    },
                ),
                "reference_upscale_method": (
                    REFERENCE_UPSCALE_METHODS,
                    {"tooltip": "Resize filter used when scaling reference images for encoding."},
                ),
                "reference_resolution_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Resolution bucket step for reference image preprocessing.",
                    },
                ),
            }
        }

    def build_settings(
        self,
        variant: str,
        guidance: float,
        reference_strength: float | str,
        precision_policy: str | float | None = None,
        memory_policy: str | None = None,
        base_shift: float | str | None = None,
        max_shift: float | None = None,
        reference_megapixels: float = 1.0,
        reference_upscale_method: str | float = "area",
        reference_resolution_steps: int | str = 1,
        *legacy_values,
    ):
        if reference_strength in {"text_to_image", "single_reference", "multi_reference"}:
            reference_strength = precision_policy
            precision_policy = memory_policy
            memory_policy = base_shift
            base_shift = max_shift
            max_shift = reference_megapixels
            reference_megapixels = reference_upscale_method
            reference_upscale_method = reference_resolution_steps
            reference_resolution_steps = legacy_values[0] if legacy_values else 1

        return (
            {
                "family": "flux2_klein_9b",
                "variant": variant,
                "guidance": guidance,
                "reference_strength": reference_strength,
                "precision_policy": precision_policy,
                "memory_policy": memory_policy,
                "base_shift": base_shift,
                "max_shift": max_shift,
                "reference_megapixels": reference_megapixels,
                "reference_upscale_method": reference_upscale_method,
                "reference_resolution_steps": reference_resolution_steps,
            },
        )
