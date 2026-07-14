"""Classic ComfyUI settings node for Krea 2."""

from __future__ import annotations

from collections.abc import Mapping

try:
    from ..services.managed_privacy_execution import (
        aio_subject_requires_private_execution,
        consume_aio_subject_mode,
    )
    from ..services.managed_prompt_privacy import KREA_SUBJECT_MODE_BINDING_ID
    from ..services.prompt_resolution import (
        KREA_EXECUTION_RESOURCE_ID,
        prompt_input_is_link,
    )
    from ..services.performance import (
        ATTENTION_MODES,
        PERFORMANCE_APPLY_TIMINGS,
        TORCH_COMPILE_BACKENDS,
        TORCH_COMPILE_MODES,
    )
except ImportError:  # pragma: no cover - direct test imports
    from services.managed_privacy_execution import (
        aio_subject_requires_private_execution,
        consume_aio_subject_mode,
    )
    from services.managed_prompt_privacy import KREA_SUBJECT_MODE_BINDING_ID
    from services.prompt_resolution import (
        KREA_EXECUTION_RESOURCE_ID,
        prompt_input_is_link,
    )
    from services.performance import (
        ATTENTION_MODES,
        PERFORMANCE_APPLY_TIMINGS,
        TORCH_COMPILE_BACKENDS,
        TORCH_COMPILE_MODES,
    )


KREA_PROMPT_BUILDER_SOURCE = "krea2_prompt_builder"
KREA2_DEFAULT_MAX_LENGTH = 4096
_MANAGED_EXECUTION_CAPABILITY = object()


def _plain_prompt(value: object) -> str:
    if isinstance(value, Mapping):
        raise ValueError("Protected Krea prompts require managed private execution.")
    return "" if value is None else str(value)


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
                "max_length": (
                    "INT",
                    {
                        "default": KREA2_DEFAULT_MAX_LENGTH,
                        "min": 1,
                        "max": KREA2_DEFAULT_MAX_LENGTH,
                        "step": 1,
                        "tooltip": "Maximum Krea 2 text-conditioning token chunk length.",
                    },
                ),
            },
            "optional": {
                "privacy_mode_reference": (
                    "STRING",
                    {
                        "default": "",
                        "socketless": True,
                        "hidden": True,
                        "tooltip": "Managed subject privacy-mode reference injected by the shared privacy barrier.",
                    },
                ),
                "private_execution": (
                    "STRING",
                    {
                        "default": "",
                        "socketless": True,
                        "hidden": True,
                        "tooltip": "Managed private Krea prompt execution reference injected by the shared privacy barrier.",
                    },
                ),
                "prompt_builder": (
                    "AIO_IDEOGRAM4_PROMPT",
                    {
                        "tooltip": "Optional structured prompt and dimensions from the AIO Ideogram 4 Prompt Builder.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
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
        max_length: int = KREA2_DEFAULT_MAX_LENGTH,
        prompt_builder=None,
        privacy_mode_reference: str = "",
        private_execution: str = "",
        unique_id: str | None = None,
        prompt=None,
        _subject_mode_lease: object = None,
        _managed_execution_capability: object = None,
    ):
        if _subject_mode_lease is None and privacy_mode_reference:
            inputs = dict(locals())
            inputs.pop("self")
            inputs.pop("_subject_mode_lease")
            inputs.pop("_managed_execution_capability")
            with consume_aio_subject_mode(
                privacy_mode_reference,
                KREA_SUBJECT_MODE_BINDING_ID,
                unique_id,
            ) as lease:
                inputs["_subject_mode_lease"] = lease
                return self.build_settings(**inputs)
        if (
            _subject_mode_lease is not None
            and aio_subject_requires_private_execution(
                _subject_mode_lease,
                KREA_SUBJECT_MODE_BINDING_ID,
            )
            and not private_execution
            and _managed_execution_capability is not _MANAGED_EXECUTION_CAPABILITY
        ):
            raise ValueError(
                "Private Krea settings require a managed execution reference."
            )
        private_builder_prompt = (
            isinstance(prompt_builder, dict)
            and bool(prompt_builder.get("privacy_mode"))
        )
        if _subject_mode_lease is None and (
            private_execution or unique_id is not None or private_builder_prompt
        ):
            raise ValueError(
                "Krea settings require managed references for subject-mode and execution."
            )
        if private_execution:
            bound_inputs = dict(locals())
            try:
                from ..services.managed_prompt_privacy import dispatch_aio_prompt_execution
            except ImportError:  # pragma: no cover - direct test imports
                from services.managed_prompt_privacy import dispatch_aio_prompt_execution
            product_inputs = {
                key: value
                for key, value in bound_inputs.items()
                if key
                not in {
                    "self",
                    "bound_inputs",
                    "private_execution",
                    "unique_id",
                    "prompt",
                    "private_builder_prompt",
                    "privacy_mode_reference",
                    "_subject_mode_lease",
                    "_managed_execution_capability",
                }
            }

            def build_resolved_settings(semantic):
                resolved_inputs = dict(product_inputs)
                resolved_inputs["inpaint_positive_prompt"] = semantic[
                    "positive_prompt_override"
                ]
                resolved_inputs["_subject_mode_lease"] = _subject_mode_lease
                resolved_inputs["_managed_execution_capability"] = (
                    _MANAGED_EXECUTION_CAPABILITY
                )
                return self.build_settings(**resolved_inputs)

            return dispatch_aio_prompt_execution(
                private_execution,
                KREA_EXECUTION_RESOURCE_ID,
                {
                    "linked_inputs": {
                        "inpaint_positive_prompt": prompt_input_is_link(
                            prompt,
                            unique_id,
                            "inpaint_positive_prompt",
                        ),
                    },
                    "prompt_inputs": {
                        "inpaint_positive_prompt": inpaint_positive_prompt,
                    },
                    "dispatch": build_resolved_settings,
                },
                subject_id=unique_id,
            )
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
            "max_length": int(max_length),
        }

        builder_prompt_applied = False
        if isinstance(prompt_builder, dict):
            prompt_is_private = bool(prompt_builder.get("privacy_mode"))
            builder_prompt = _plain_prompt(prompt_builder.get("prompt", "")).strip()
            if builder_prompt:
                if prompt_is_private and _subject_mode_lease is None:
                    raise ValueError(
                        "Private builder prompts require managed private execution."
                    )
                settings["positive_prompt_override"] = builder_prompt
                settings["positive_prompt_source"] = KREA_PROMPT_BUILDER_SOURCE
                if prompt_is_private:
                    settings["prompt_builder_privacy_mode"] = True
                for key in ("width", "height", "max_side", "aspect_ratio", "multiple_value"):
                    if key in prompt_builder:
                        settings[f"prompt_builder_{key}"] = prompt_builder[key]
                builder_prompt_applied = True

        prompt = _plain_prompt(inpaint_positive_prompt).strip()
        if prompt and not builder_prompt_applied:
            settings["positive_prompt_override"] = prompt
            settings["positive_prompt_source"] = "krea2_inpaint_settings"
        return (
            settings,
        )
