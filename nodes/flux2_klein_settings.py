"""Classic ComfyUI settings node for FLUX.2 Klein 9B."""

from __future__ import annotations

try:
    from ..services.performance import (
        ATTENTION_MODES,
        PERFORMANCE_APPLY_TIMINGS,
        TORCH_COMPILE_BACKENDS,
        TORCH_COMPILE_MODES,
    )
except ImportError:  # pragma: no cover - direct test imports
    from services.performance import (
        ATTENTION_MODES,
        PERFORMANCE_APPLY_TIMINGS,
        TORCH_COMPILE_BACKENDS,
        TORCH_COMPILE_MODES,
    )


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
                "precision_policy": (
                    ["auto", "fp8", "bf16"],
                    {"tooltip": "Model precision preference. Auto chooses a practical format for the current runtime."},
                ),
                "memory_policy": (
                    ["auto", "low_vram", "balanced", "high_vram"],
                    {"tooltip": "Memory strategy for loading and running the model on your hardware."},
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
                "attention_mode": (
                    ATTENTION_MODES,
                    {"default": "auto", "tooltip": "Attention backend preference. Auto selects the best installed option."},
                ),
                "torch_compile_mode": (
                    TORCH_COMPILE_MODES,
                    {"default": "off", "tooltip": "Torch compile behavior for the diffusion model."},
                ),
                "torch_compile_backend": (
                    TORCH_COMPILE_BACKENDS,
                    {"default": "inductor", "tooltip": "Torch compile backend. Inductor is the Triton-backed path."},
                ),
                "performance_apply_timing": (
                    PERFORMANCE_APPLY_TIMINGS,
                    {"default": "after_loras", "tooltip": "Apply attention and compile settings before or after AIO LoRAs."},
                ),
            }
        }

    def build_settings(
        self,
        variant: str,
        guidance: float,
        precision_policy: str = "auto",
        memory_policy: str = "balanced",
        reference_megapixels: float | str = 1.0,
        reference_upscale_method: str | float = "area",
        reference_resolution_steps: int | str = 1,
        attention_mode: str = "auto",
        torch_compile_mode: str = "off",
        torch_compile_backend: str = "inductor",
        performance_apply_timing: str = "after_loras",
    ):
        return (
            {
                "family": "flux2_klein_9b",
                "variant": variant,
                "guidance": guidance,
                "precision_policy": precision_policy,
                "memory_policy": memory_policy,
                "reference_megapixels": reference_megapixels,
                "reference_upscale_method": reference_upscale_method,
                "reference_resolution_steps": reference_resolution_steps,
                "attention_mode": attention_mode,
                "torch_compile_mode": torch_compile_mode,
                "torch_compile_backend": torch_compile_backend,
                "performance_apply_timing": performance_apply_timing,
            },
        )
