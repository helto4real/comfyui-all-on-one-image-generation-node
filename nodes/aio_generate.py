"""Classic all-in-one ComfyUI facade node."""

from __future__ import annotations

from typing import Any

try:
    from ..adapters import Flux2Klein9BAdapter, ZImageTurboAdapter  # noqa: F401
    from ..services.dimensions import (
        ASPECT_RATIOS,
        MULTIPLE_VALUES,
        SIZE_MODES,
        resolve_dimensions_from_controls,
    )
    from ..services.progress import ProgressReporter
    from ..services.registry import get_adapter, get_profile, list_model_types
    from ..services.model_resolution import infer_model_format
    from ..services.lora_config import normalize_lora_config, summarize_loras
    from ..services.reference_inputs import (
        REFERENCE_IMAGE_INPUT_NAMES,
        normalize_reference_inputs,
    )
    from ..services.run_info import build_run_info, to_json
    from ..services.validation import (
        validate_model_type,
        validate_settings_family,
    )
except ImportError:  # pragma: no cover - direct test imports
    from adapters import Flux2Klein9BAdapter, ZImageTurboAdapter  # noqa: F401
    from services.dimensions import (
        ASPECT_RATIOS,
        MULTIPLE_VALUES,
        SIZE_MODES,
        resolve_dimensions_from_controls,
    )
    from services.progress import ProgressReporter
    from services.registry import get_adapter, get_profile, list_model_types
    from services.model_resolution import infer_model_format
    from services.lora_config import normalize_lora_config, summarize_loras
    from services.reference_inputs import (
        REFERENCE_IMAGE_INPUT_NAMES,
        normalize_reference_inputs,
    )
    from services.run_info import build_run_info, to_json
    from services.validation import (
        validate_model_type,
        validate_settings_family,
    )


DEFAULT_PROMPT = "A luminous studio portrait, crisp details, natural color, soft light"


def _filename_list(category: str) -> list[str]:
    try:
        import folder_paths  # type: ignore

        return list(folder_paths.get_filename_list(category))
    except Exception:
        return []


def _combined_filenames(categories: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for category in categories:
        for name in _filename_list(category):
            value = f"{category}/{name}" if len(categories) > 1 else name
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values or [""]


def _samplers() -> list[str]:
    try:
        import comfy.samplers  # type: ignore

        return ["auto", *list(comfy.samplers.KSampler.SAMPLERS)]
    except Exception:
        return ["auto"]


def _schedulers() -> list[str]:
    try:
        import comfy.samplers  # type: ignore

        return ["auto", *list(comfy.samplers.KSampler.SCHEDULERS)]
    except Exception:
        return ["auto"]


class AIOImageGenerate:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("image", "latent", "run_info")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (list_model_types(),),
                "diffusion_model": (
                    _combined_filenames(
                        (
                            "diffusion_models",
                            "unet",
                            "checkpoints",
                            "unet_gguf",
                            "model_gguf",
                        )
                    ),
                ),
                "text_encoder": (_combined_filenames(("text_encoders", "clip", "clip_gguf")),),
                "vae": (_combined_filenames(("vae", "vae_gguf")),),
                "positive_prompt": (
                    "STRING",
                    {"default": DEFAULT_PROMPT, "multiline": True, "dynamicPrompts": True},
                ),
                "negative_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
                "size mode": (list(SIZE_MODES),),
                "max side": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 1}),
                "aspect ratio": (list(ASPECT_RATIOS),),
                "multiple value": (list(MULTIPLE_VALUES),),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**63 - 1, "control_after_generate": True},
                ),
                "steps": ("INT", {"default": 0, "min": 0, "max": 100}),
                "cfg": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "sampler": (_samplers(),),
                "scheduler": (_schedulers(),),
            },
            "optional": {
                "model_settings": ("AIO_MODEL_SETTINGS",),
                "lora_config": ("AIO_LORA_CONFIG",),
                **{name: ("IMAGE",) for name in REFERENCE_IMAGE_INPUT_NAMES},
                "mask": ("MASK",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def generate(
        self,
        model_type: str,
        diffusion_model: str,
        text_encoder: str,
        vae: str,
        positive_prompt: str,
        negative_prompt: str,
        width: int | None = None,
        height: int | None = None,
        seed: int = 0,
        steps: int = 0,
        cfg: float = 0.0,
        sampler: str = "auto",
        scheduler: str = "auto",
        model_settings: dict[str, Any] | None = None,
        lora_config: dict[str, Any] | None = None,
        reference_image: Any = None,
        mask: Any = None,
        unique_id: str | None = None,
        prompt: Any = None,
        extra_pnginfo: Any = None,
        weight_format: str | None = None,
        **reference_values: Any,
    ):
        del prompt, extra_pnginfo, weight_format
        reference_inputs = normalize_reference_inputs(
            reference_values,
            reference_image=reference_image,
            mask=mask,
        )

        validate_model_type(model_type)
        profile = get_profile(model_type)
        validate_settings_family(model_type, model_settings)
        normalized_lora_config = normalize_lora_config(lora_config)
        lora_summary = summarize_loras(normalized_lora_config)

        adapter = get_adapter(model_type)
        dimensions = resolve_dimensions_from_controls(
            size_mode=reference_values.get("size mode"),
            max_side=reference_values.get("max side"),
            aspect_ratio=reference_values.get("aspect ratio"),
            reference_inputs=reference_inputs,
            legacy_width=width,
            legacy_height=height,
            default_width=profile.default_width,
            default_height=profile.default_height,
            multiple_value=reference_values.get("multiple value"),
        )
        settings = adapter.resolve_settings(
            model_settings=model_settings,
            width=dimensions.width,
            height=dimensions.height,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
        )
        settings["max_side"] = dimensions.max_side
        settings["aspect_ratio"] = dimensions.aspect_ratio
        settings["size_mode"] = dimensions.size_mode
        settings["multiple_value"] = dimensions.multiple_value
        effective_width = int(settings["width"])
        effective_height = int(settings["height"])
        effective_steps = int(settings["steps"])
        effective_cfg = float(settings["cfg"])
        effective_sampler = str(settings["sampler"])
        effective_scheduler = str(settings["scheduler"])

        warnings = adapter.validate_inputs(
            diffusion_model=diffusion_model,
            text_encoder=text_encoder,
            vae=vae,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=effective_width,
            height=effective_height,
            settings=settings,
            reference_inputs=reference_inputs,
        )

        progress = ProgressReporter(total_steps=effective_steps, node_id=unique_id)
        progress.phase("resolving models")
        image, latent = adapter.generate(
            diffusion_model=diffusion_model,
            text_encoder=text_encoder,
            vae=vae,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=effective_width,
            height=effective_height,
            seed=seed,
            settings=settings,
            sampler=effective_sampler,
            scheduler=effective_scheduler,
            lora_config=normalized_lora_config,
            reference_inputs=reference_inputs,
            progress=progress,
        )
        progress.done()

        run_info = build_run_info(
            model_type=model_type,
            display_name=profile.display_name,
            diffusion_model=diffusion_model,
            diffusion_model_format=infer_model_format(diffusion_model),
            text_encoder=text_encoder,
            text_encoder_format=infer_model_format(text_encoder),
            vae=vae,
            vae_format=infer_model_format(vae),
            width=effective_width,
            height=effective_height,
            seed=seed,
            steps=effective_steps,
            cfg=effective_cfg,
            sampler=effective_sampler,
            scheduler=effective_scheduler,
            settings=settings,
            warnings=warnings,
            adapter_version=adapter.version,
            loras=lora_summary,
        )
        return image, latent, to_json(run_info)
