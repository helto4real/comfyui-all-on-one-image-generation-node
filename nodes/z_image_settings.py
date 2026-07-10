"""Classic ComfyUI settings node for Z-Image Turbo."""

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


class AIOZImageTurboSettings:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("AIO_MODEL_SETTINGS",)
    FUNCTION = "build_settings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 50,
                        "tooltip": "Exact sampling step count for Z-Image Turbo.",
                    },
                ),
                "precision_policy": (
                    ["auto", "fp8", "bf16"],
                    {"tooltip": "Model precision preference. Auto chooses a practical format for the current runtime."},
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
        force_steps: int,
        precision_policy: str,
        attention_mode: str = "auto",
        torch_compile_mode: str = "off",
        torch_compile_backend: str = "inductor",
        performance_apply_timing: str = "after_loras",
    ):
        return (
            {
                "family": "z_image_turbo",
                "force_steps": force_steps,
                "precision_policy": precision_policy,
                "attention_mode": attention_mode,
                "torch_compile_mode": torch_compile_mode,
                "torch_compile_backend": torch_compile_backend,
                "performance_apply_timing": performance_apply_timing,
            },
        )
