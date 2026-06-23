"""Classic ComfyUI settings node for Krea 2."""

from __future__ import annotations

try:
    from ..services.krea2_rebalance import DEFAULT_KREA2_REBALANCE_WEIGHTS
    from ..services.performance import (
        ATTENTION_MODES,
        PERFORMANCE_APPLY_TIMINGS,
        TORCH_COMPILE_BACKENDS,
        TORCH_COMPILE_MODES,
    )
except ImportError:  # pragma: no cover - direct test imports
    from services.krea2_rebalance import DEFAULT_KREA2_REBALANCE_WEIGHTS
    from services.performance import (
        ATTENTION_MODES,
        PERFORMANCE_APPLY_TIMINGS,
        TORCH_COMPILE_BACKENDS,
        TORCH_COMPILE_MODES,
    )


class AIOKrea2Settings:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("AIO_MODEL_SETTINGS",)
    FUNCTION = "build_settings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rebalance_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply Krea 2 per-layer conditioning rebalance to the positive prompt conditioning.",
                    },
                ),
                "rebalance_multiplier": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": -1000000000.0,
                        "max": 1000000000.0,
                        "step": 0.01,
                        "tooltip": "Global multiplier applied by the Krea 2 conditioning rebalance.",
                    },
                ),
                "rebalance_per_layer_weights": (
                    "STRING",
                    {
                        "default": DEFAULT_KREA2_REBALANCE_WEIGHTS,
                        "multiline": False,
                        "tooltip": "Comma-separated Krea 2 conditioning layer weights.",
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
                "fp16_accumulation_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable torch CUDA fp16 accumulation callbacks during Krea 2 sampling when supported.",
                    },
                ),
            }
        }

    def build_settings(
        self,
        rebalance_enabled: bool,
        rebalance_multiplier: float,
        rebalance_per_layer_weights: str,
        precision_policy: str,
        attention_mode: str = "auto",
        torch_compile_mode: str = "off",
        torch_compile_backend: str = "inductor",
        performance_apply_timing: str = "after_loras",
        fp16_accumulation_enabled: bool = True,
    ):
        return (
            {
                "family": "krea2",
                "rebalance_enabled": bool(rebalance_enabled),
                "rebalance_multiplier": rebalance_multiplier,
                "rebalance_per_layer_weights": rebalance_per_layer_weights,
                "precision_policy": precision_policy,
                "attention_mode": attention_mode,
                "torch_compile_mode": torch_compile_mode,
                "torch_compile_backend": torch_compile_backend,
                "performance_apply_timing": performance_apply_timing,
                "fp16_accumulation_enabled": bool(fp16_accumulation_enabled),
            },
        )
