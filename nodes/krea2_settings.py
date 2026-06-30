"""Classic ComfyUI settings node for Krea 2."""

from __future__ import annotations

try:
    from ..services import privacy
    from ..services.performance import (
        ATTENTION_MODES,
        PERFORMANCE_APPLY_TIMINGS,
        TORCH_COMPILE_BACKENDS,
        TORCH_COMPILE_MODES,
    )
except ImportError:  # pragma: no cover - direct test imports
    from services import privacy
    from services.performance import (
        ATTENTION_MODES,
        PERFORMANCE_APPLY_TIMINGS,
        TORCH_COMPILE_BACKENDS,
        TORCH_COMPILE_MODES,
    )


KREA_PROMPT_BUILDER_SOURCE = "krea2_prompt_builder"


class AIOKrea2Settings:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("AIO_MODEL_SETTINGS",)
    FUNCTION = "build_settings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enhancer_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply the Krea2T prompt-adherence enhancer to the diffusion model during sampling.",
                    },
                ),
                "enhancer_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Blend amount for the Krea2T prompt-adherence enhancer.",
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
                "inpaint_positive_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Optional positive prompt to use only for Krea 2 inpaint runs.",
                    },
                ),
            },
            "optional": {
                "prompt_builder": (
                    "AIO_IDEOGRAM4_PROMPT",
                    {
                        "tooltip": "Optional structured prompt and dimensions from the AIO Ideogram 4 Prompt Builder.",
                    },
                ),
            },
        }

    def build_settings(
        self,
        enhancer_enabled: bool,
        enhancer_strength: float,
        precision_policy: str,
        attention_mode: str = "auto",
        torch_compile_mode: str = "off",
        torch_compile_backend: str = "inductor",
        performance_apply_timing: str = "after_loras",
        fp16_accumulation_enabled: bool = True,
        inpaint_positive_prompt: str = "",
        prompt_builder=None,
    ):
        settings = {
            "family": "krea2",
            "enhancer_enabled": bool(enhancer_enabled),
            "enhancer_strength": enhancer_strength,
            "precision_policy": precision_policy,
            "attention_mode": attention_mode,
            "torch_compile_mode": torch_compile_mode,
            "torch_compile_backend": torch_compile_backend,
            "performance_apply_timing": performance_apply_timing,
            "fp16_accumulation_enabled": bool(fp16_accumulation_enabled),
        }

        builder_prompt_applied = False
        if isinstance(prompt_builder, dict):
            prompt_is_private = bool(prompt_builder.get("privacy_mode"))
            builder_prompt = privacy.decrypt_text_if_encrypted(prompt_builder.get("prompt", "")).strip()
            if builder_prompt:
                settings["positive_prompt_override"] = (
                    privacy.encrypt_state({"value": builder_prompt}) if prompt_is_private else builder_prompt
                )
                settings["positive_prompt_source"] = KREA_PROMPT_BUILDER_SOURCE
                if prompt_is_private:
                    settings["prompt_builder_privacy_mode"] = True
                for key in ("width", "height", "max_side", "aspect_ratio", "multiple_value"):
                    if key in prompt_builder:
                        settings[f"prompt_builder_{key}"] = prompt_builder[key]
                builder_prompt_applied = True

        prompt = privacy.decrypt_text_if_encrypted(inpaint_positive_prompt).strip()
        if prompt and not builder_prompt_applied:
            settings["positive_prompt_override"] = prompt
            settings["positive_prompt_source"] = "krea2_inpaint_settings"
        return (
            settings,
        )
