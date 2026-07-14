"""Classic ComfyUI settings node for Ideogram 4."""

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


DEFAULT_UNCONDITIONAL_MODEL = "diffusion_models/ideogram4/ideogram4_unconditional_fp8_scaled.safetensors"
IDEOGRAM4_PRESETS = {
    "Default": {"steps": 20, "mu": 0.0, "std": 1.75, "schedule_mode": "ideogram4"},
    "Quality": {"steps": 48, "mu": 0.0, "std": 1.5, "schedule_mode": "ideogram4"},
    "Turbo": {"steps": 12, "mu": 0.5, "std": 1.75, "schedule_mode": "ideogram4"},
    "Workflow Compatible": {
        "steps": 28,
        "mu": 0.0,
        "std": 1.75,
        "schedule_mode": "basic",
        "scheduler": "simple",
        "cfg_override_start_percent": 0.9,
    },
}


def _diffusion_model_names() -> list[str]:
    names: list[str] = []
    try:
        import folder_paths  # type: ignore

        for category in ("diffusion_models", "unet", "checkpoints"):
            try:
                names.extend(f"{category}/{name}" for name in folder_paths.get_filename_list(category))
            except Exception:
                continue
    except Exception:
        pass
    if DEFAULT_UNCONDITIONAL_MODEL not in names:
        names.insert(0, DEFAULT_UNCONDITIONAL_MODEL)
    return list(dict.fromkeys(names))


class AIOIdeogram4Settings:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("AIO_MODEL_SETTINGS",)
    FUNCTION = "build_settings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    list(IDEOGRAM4_PRESETS),
                    {"default": "Default", "tooltip": "Ideogram 4 sampling preset."},
                ),
                "unconditional_model": (
                    _diffusion_model_names(),
                    {
                        "default": DEFAULT_UNCONDITIONAL_MODEL,
                        "tooltip": "Unconditional Ideogram 4 diffusion model used for dual-model guidance.",
                    },
                ),
                "dual_cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Dual-model CFG value passed to the Ideogram guider.",
                    },
                ),
                "cfg_override_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply a final CFG override to the conditional model.",
                    },
                ),
                "cfg_override": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "CFG value used by the final CFG override window.",
                    },
                ),
                "cfg_override_start_percent": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Start percent for the final CFG override window.",
                    },
                ),
                "cfg_override_end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "End percent for the final CFG override window.",
                    },
                ),
                "sampling_shift": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "AuraFlow model sampling shift applied to the conditional model.",
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
                    {"default": "off", "tooltip": "Torch compile behavior for the diffusion models."},
                ),
                "torch_compile_backend": (
                    TORCH_COMPILE_BACKENDS,
                    {"default": "inductor", "tooltip": "Torch compile backend. Inductor is the Triton-backed path."},
                ),
                "performance_apply_timing": (
                    PERFORMANCE_APPLY_TIMINGS,
                    {"default": "after_loras", "tooltip": "Apply attention and compile settings before or after AIO LoRAs."},
                ),
                "run_unconditional_model": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Run the separate unconditional Ideogram 4 model for dual-model guidance. Disable for turbo LoRA workflows.",
                    },
                ),
            },
            "optional": {
                "prompt_builder": (
                    "AIO_IDEOGRAM4_PROMPT",
                    {
                        "tooltip": "Optional prompt and dimensions from the AIO Ideogram 4 Prompt Builder.",
                    },
                ),
            },
        }

    def build_settings(
        self,
        preset: str,
        unconditional_model: str,
        dual_cfg: float,
        cfg_override_enabled: bool,
        cfg_override: float,
        cfg_override_start_percent: float,
        cfg_override_end_percent: float,
        sampling_shift: float,
        precision_policy: str,
        attention_mode: str = "auto",
        torch_compile_mode: str = "off",
        torch_compile_backend: str = "inductor",
        performance_apply_timing: str = "after_loras",
        run_unconditional_model: bool = True,
        prompt_builder=None,
    ):
        preset_values = dict(IDEOGRAM4_PRESETS.get(preset, IDEOGRAM4_PRESETS["Default"]))
        if preset == "Workflow Compatible" and cfg_override_start_percent == 0.7:
            cfg_override_start_percent = float(preset_values["cfg_override_start_percent"])
        settings = {
            "family": "ideogram4",
            "preset": preset,
            "unconditional_model": unconditional_model,
            "preset_steps": int(preset_values["steps"]),
            "mu": float(preset_values["mu"]),
            "std": float(preset_values["std"]),
            "schedule_mode": preset_values["schedule_mode"],
            "scheduler": preset_values.get("scheduler", "ideogram4"),
            "dual_cfg": dual_cfg,
            "cfg_override_enabled": cfg_override_enabled,
            "cfg_override": cfg_override,
            "cfg_override_start_percent": cfg_override_start_percent,
            "cfg_override_end_percent": cfg_override_end_percent,
            "sampling_shift": sampling_shift,
            "precision_policy": precision_policy,
            "attention_mode": attention_mode,
            "torch_compile_mode": torch_compile_mode,
            "torch_compile_backend": torch_compile_backend,
            "performance_apply_timing": performance_apply_timing,
            "run_unconditional_model": bool(run_unconditional_model),
        }
        if isinstance(prompt_builder, dict):
            prompt_is_private = bool(prompt_builder.get("privacy_mode"))
            prompt_value = prompt_builder.get("prompt", "")
            if not isinstance(prompt_value, str):
                raise ValueError(
                    "Protected Ideogram prompts require managed private execution."
                )
            prompt = prompt_value.strip()
            if prompt:
                settings["positive_prompt_override"] = prompt
                settings["positive_prompt_source"] = "ideogram4_prompt_builder"
            if prompt_is_private:
                settings["prompt_builder_privacy_mode"] = True
            for key in ("width", "height", "max_side", "aspect_ratio", "multiple_value"):
                if key in prompt_builder:
                    settings[f"prompt_builder_{key}"] = prompt_builder[key]
        return (settings,)
