"""Classic all-in-one ComfyUI facade node."""

from __future__ import annotations

from typing import Any

try:
    from ..adapters import Flux2Klein9BAdapter, ZImageTurboAdapter  # noqa: F401
    from ..services import pipeline
    from ..services.dimensions import (
        ASPECT_RATIOS,
        MULTIPLE_VALUES,
        SIZE_MODE_ASPECT_RATIO,
        SIZE_MODE_IMAGE_1,
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
    from services import pipeline
    from services.dimensions import (
        ASPECT_RATIOS,
        MULTIPLE_VALUES,
        SIZE_MODE_ASPECT_RATIO,
        SIZE_MODE_IMAGE_1,
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
PID_LATENT_OUTPUT_INDEX = 6
PID_SIGMA_OUTPUT_INDEX = 7


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


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
    )


def _class_is_output_node(class_type: str) -> bool:
    try:
        import nodes as comfy_nodes  # type: ignore

        class_def = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(class_type)
    except Exception:
        class_def = None
    if class_def is not None:
        return bool(getattr(class_def, "OUTPUT_NODE", False))
    return class_type in {"PreviewImage", "SaveImage"}


def output_is_reachable(
    prompt: Any,
    unique_id: str | None,
    socket_index: int,
    *,
    default: bool = False,
) -> bool:
    if not isinstance(prompt, dict) or unique_id is None:
        return default

    target_id = str(unique_id)
    consumers: dict[tuple[str, int], set[str]] = {}
    consumers_by_node: dict[str, set[str]] = {}
    output_nodes: list[str] = []
    for raw_node_id, node_data in prompt.items():
        node_id = str(raw_node_id)
        if not isinstance(node_data, dict):
            continue
        class_type = node_data.get("class_type")
        if isinstance(class_type, str) and _class_is_output_node(class_type):
            output_nodes.append(node_id)
        inputs = node_data.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if not _is_link(value):
                continue
            from_node_id, from_socket = str(value[0]), int(value[1])
            consumers.setdefault((from_node_id, from_socket), set()).add(node_id)
            consumers_by_node.setdefault(from_node_id, set()).add(node_id)

    if not output_nodes:
        return default

    nodes_to_visit = list(consumers.get((target_id, socket_index), set()))
    visited: set[str] = set()
    output_node_ids = set(output_nodes)
    while nodes_to_visit:
        node_id = nodes_to_visit.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id in output_node_ids:
            return True
        nodes_to_visit.extend(consumers_by_node.get(node_id, ()))
    return False


def image_output_is_required(prompt: Any, unique_id: str | None) -> bool:
    return output_is_reachable(prompt, unique_id, 0, default=True)


def workflow_output_has_link(extra_pnginfo: Any, unique_id: str | None, socket_index: int) -> bool:
    if unique_id is None or not isinstance(extra_pnginfo, dict):
        return False

    workflow = extra_pnginfo.get("workflow")
    if not isinstance(workflow, dict):
        return False

    target_id = str(unique_id)
    for link in workflow.get("links", []) or []:
        if (
            isinstance(link, (list, tuple))
            and len(link) >= 3
            and str(link[1]) == target_id
            and link[2] == socket_index
        ):
            return True

    for node in workflow.get("nodes", []) or []:
        if not isinstance(node, dict) or str(node.get("id")) != target_id:
            continue
        outputs = node.get("outputs", [])
        if not isinstance(outputs, list) or socket_index >= len(outputs):
            return False
        output = outputs[socket_index]
        if not isinstance(output, dict):
            return False
        links = output.get("links")
        return bool(links)
    return False


def output_is_connected(
    prompt: Any,
    extra_pnginfo: Any,
    unique_id: str | None,
    socket_index: int,
    *,
    default: bool = False,
) -> bool:
    return workflow_output_has_link(extra_pnginfo, unique_id, socket_index) or output_is_reachable(
        prompt,
        unique_id,
        socket_index,
        default=default,
    )


class AIOImageGenerate:
    CATEGORY = "AIO/Image"
    RETURN_TYPES = (
        "IMAGE",
        "LATENT",
        "STRING",
        "CONDITIONING",
        "CONDITIONING",
        "VAE",
        "LATENT",
        "FLOAT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "image",
        "latent",
        "run_info",
        "positive",
        "negative",
        "vae",
        "pid_latent",
        "pid_sigma",
        "width",
        "height",
    )
    FUNCTION = "generate"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        del kwargs
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        reference_tooltips = {
            name: (
                f"Optional reference image {index}. Connect images in order starting "
                "with image 1. Currently used by FLUX.2 Klein 9B image-reference workflows."
            )
            for index, name in enumerate(REFERENCE_IMAGE_INPUT_NAMES, start=1)
        }
        return {
            "required": {
                "model_type": (
                    list_model_types(),
                    {"tooltip": "Select the model family/profile that controls defaults, validation, and adapter behavior."},
                ),
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
                    {"tooltip": "Diffusion model file to load. Supports standard and GGUF model folders when available."},
                ),
                "text_encoder": (
                    _combined_filenames(("text_encoders", "clip", "clip_gguf")),
                    {"tooltip": "Text encoder or CLIP file used to encode the prompts for the selected model family."},
                ),
                "vae": (
                    _combined_filenames(("vae", "vae_gguf")),
                    {"tooltip": "VAE file used to decode generated latents into the final image."},
                ),
                "positive_prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_PROMPT,
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Prompt describing what the generated image should contain.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Prompt describing content to avoid. Some model families ignore this input by default.",
                    },
                ),
                "size mode": (
                    list(SIZE_MODES),
                    {"tooltip": "Choose whether output dimensions come from the aspect ratio controls or from image 1."},
                ),
                "max side": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Longest output edge in pixels when using aspect-ratio sizing.",
                    },
                ),
                "aspect ratio": (
                    list(ASPECT_RATIOS),
                    {"tooltip": "Output shape to use with max side when size mode is set to use aspect ratio."},
                ),
                "multiple value": (
                    list(MULTIPLE_VALUES),
                    {"tooltip": "Round generated dimensions to a multiple required or preferred by the selected model."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**63 - 1,
                        "control_after_generate": True,
                        "tooltip": "Random seed for generation. Reuse the same seed and settings for repeatable results.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "tooltip": "Sampling step count. Use 0 to let the selected model profile choose its default.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance scale. Use 0 to let the selected profile choose its default.",
                    },
                ),
                "sampler": (
                    _samplers(),
                    {"tooltip": "Sampling algorithm. Auto lets the selected model profile choose a compatible sampler."},
                ),
                "scheduler": (
                    _schedulers(),
                    {"tooltip": "Noise schedule used during sampling. Auto lets the selected model profile choose a default."},
                ),
                "pid_capture_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Main sampler step to capture for PID. Use 0 to auto-select a step near the end.",
                    },
                ),
            },
            "optional": {
                "model_settings": (
                    "AIO_MODEL_SETTINGS",
                    {"tooltip": "Optional settings object from a matching AIO model settings node."},
                ),
                "lora_config": (
                    "AIO_LORA_CONFIG",
                    {"tooltip": "Optional LoRA stack from the AIO LoRA Configuration node."},
                ),
                "model": (
                    "MODEL",
                    {"tooltip": "Optional externally loaded or patched post-LoRA model. Connect with clip to skip internal model loading and LoRA application."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "Optional externally loaded or patched post-LoRA CLIP. Connect with model to skip internal text encoder loading and LoRA application."},
                ),
                **{name: ("IMAGE", {"tooltip": tooltip}) for name, tooltip in reference_tooltips.items()},
                "mask": (
                    "MASK",
                    {"tooltip": "Optional mask for reference-image workflows. Connect image 1 before using a mask."},
                ),
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
        pid_capture_step: int = 0,
        model_settings: dict[str, Any] | None = None,
        lora_config: dict[str, Any] | None = None,
        model: Any = None,
        clip: Any = None,
        reference_image: Any = None,
        mask: Any = None,
        unique_id: str | None = None,
        prompt: Any = None,
        extra_pnginfo: Any = None,
        weight_format: str | None = None,
        **reference_values: Any,
    ):
        del weight_format
        image_connected = output_is_connected(prompt, extra_pnginfo, unique_id, 0, default=True)
        vae_connected = output_is_connected(prompt, extra_pnginfo, unique_id, 5)
        pid_latent_connected = output_is_connected(prompt, extra_pnginfo, unique_id, PID_LATENT_OUTPUT_INDEX)
        pid_sigma_connected = output_is_connected(prompt, extra_pnginfo, unique_id, PID_SIGMA_OUTPUT_INDEX)
        pid_capture_connected = pid_latent_connected or pid_sigma_connected
        decode_image = image_connected
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
        size_mode = reference_values.get("size mode")
        if size_mode == SIZE_MODE_IMAGE_1 and reference_inputs.count == 0:
            size_mode = SIZE_MODE_ASPECT_RATIO
        dimensions = resolve_dimensions_from_controls(
            size_mode=size_mode,
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
        resolved_pid_capture_step = pipeline.resolve_pid_capture_step(
            pid_capture_step,
            effective_steps,
        ) if pid_capture_connected else None
        image, latent, positive, negative, loaded_vae = adapter.generate(
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
            loaded_model=model,
            loaded_clip=clip,
            reference_inputs=reference_inputs,
            decode_image=decode_image,
            return_vae=vae_connected,
            pid_capture_step=resolved_pid_capture_step,
            progress=progress,
        )

        pid_capture = latent.get(pipeline.PID_CAPTURE_KEY) if isinstance(latent, dict) else None
        pid_latent = pid_capture["latent"] if pid_capture_connected and pid_capture else None
        pid_sigma = float(pid_capture["sigma"]) if pid_capture_connected and pid_capture else 0.0
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
        return (
            image,
            latent,
            to_json(run_info),
            positive,
            negative,
            loaded_vae,
            pid_latent,
            pid_sigma,
            effective_width,
            effective_height,
        )
