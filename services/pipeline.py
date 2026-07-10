"""Lazy ComfyUI text-to-image pipeline helpers.

This module intentionally keeps all ComfyUI, torch, and custom-node imports inside
execution-time functions so importing this custom node pack remains lightweight.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    from ..loaders import gguf_backend, safetensors_backend
    from .dimensions import image_tensor_dimensions, parse_multiple_value, round_to_multiple
    from . import inpaint as inpaint_service
    from . import krea2_enhancer
    from .lora_application import apply_lora_config
    from .lora_config import normalize_lora_config
    from .model_resolution import infer_model_format, strip_category_prefix
    from .performance import (
        apply_memory_policy_before_sampling,
        apply_performance_settings,
        normalize_performance_apply_timing,
        performance_settings_present,
    )
except ImportError:  # pragma: no cover - direct test imports
    from loaders import gguf_backend, safetensors_backend
    from services.dimensions import image_tensor_dimensions, parse_multiple_value, round_to_multiple
    from services import inpaint as inpaint_service
    from services import krea2_enhancer
    from services.lora_application import apply_lora_config
    from services.lora_config import normalize_lora_config
    from services.model_resolution import infer_model_format, strip_category_prefix
    from services.performance import (
        apply_memory_policy_before_sampling,
        apply_performance_settings,
        normalize_performance_apply_timing,
        performance_settings_present,
    )


PID_CAPTURE_KEY = "pid_capture"
SECOND_PASS_INFO_KEY = "aio_second_pass_info"
SECOND_PASS_ORIGINAL_IMAGE_KEY = "aio_second_pass_original_image"
SECOND_PASS_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "lanczos")
INPAINT_PREVIEW_SOURCE = "inpaint_source"
INPAINT_PREVIEW_SAMPLE = "inpaint_sample"
INPAINT_PREVIEW_MASK = "inpaint_mask"
INPAINT_PREVIEW_REQUESTED = "requested"
SEED_WRAP = 2**63
KREA2_DEFAULT_MAX_LENGTH = 4096
KREA2_MAX_MAX_LENGTH = 4096
KREA2_MIN_MAX_LENGTH = 1


@dataclass
class GenerationResult:
    image: Any
    latent: Any
    positive: Any
    negative: Any
    vae: Any
    model: Any = None
    clip: Any = None

    def __iter__(self):
        yield self.image
        yield self.latent
        yield self.positive
        yield self.negative
        yield self.vae


def incrementing_batch_seeds(seed: int, batch_count: int) -> tuple[int, ...]:
    count = int(batch_count)
    if count < 1:
        raise ValueError("batch_count must be at least 1.")
    base = int(seed) % SEED_WRAP
    return tuple((base + index) % SEED_WRAP for index in range(count))


def _concat_optional_tensors(values: list[Any]) -> Any:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    import torch  # type: ignore

    return torch.cat(present, dim=0)


def _combine_pid_capture_batch(captures: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not captures:
        return None
    first = captures[0]
    latents = [capture.get("latent") for capture in captures]
    if not all(isinstance(latent, dict) and "samples" in latent for latent in latents):
        return first
    combined_latent = latents[0].copy()
    combined_latent["samples"] = _concat_optional_tensors([latent["samples"] for latent in latents])
    return {
        "latent": combined_latent,
        "sigma": float(first.get("sigma", 0.0)),
        "step": int(first.get("step", combined_latent.get("pid_capture_step", 0))),
    }


def _combine_second_pass_info_batch(infos: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not infos:
        return None
    first = dict(infos[0])
    if len(infos) > 1:
        first["batch_count"] = len(infos)
        first["items"] = infos
    return first


def _combine_latent_batch(latents: list[Any]) -> Any:
    present = [latent for latent in latents if latent is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    if not all(isinstance(latent, dict) and "samples" in latent for latent in present):
        return present[0]

    combined = present[0].copy()
    combined["samples"] = _concat_optional_tensors([latent["samples"] for latent in present])

    captures = [
        latent[PID_CAPTURE_KEY]
        for latent in present
        if isinstance(latent.get(PID_CAPTURE_KEY), dict)
    ]
    if captures:
        pid_capture = _combine_pid_capture_batch(captures)
        if pid_capture is not None:
            combined[PID_CAPTURE_KEY] = pid_capture

    second_pass_infos = [
        latent[SECOND_PASS_INFO_KEY]
        for latent in present
        if isinstance(latent.get(SECOND_PASS_INFO_KEY), dict)
    ]
    if second_pass_infos:
        combined[SECOND_PASS_INFO_KEY] = _combine_second_pass_info_batch(second_pass_infos)

    original_images = [
        latent.get(SECOND_PASS_ORIGINAL_IMAGE_KEY)
        for latent in present
        if latent.get(SECOND_PASS_ORIGINAL_IMAGE_KEY) is not None
    ]
    if original_images:
        combined[SECOND_PASS_ORIGINAL_IMAGE_KEY] = _concat_optional_tensors(original_images)

    return combined


def _inpaint_preview_requested(previews: dict[str, Any] | None, name: str) -> bool:
    if not isinstance(previews, dict):
        return False
    requested = previews.get(INPAINT_PREVIEW_REQUESTED)
    return isinstance(requested, dict) and bool(requested.get(name))


def _set_inpaint_preview(previews: dict[str, Any] | None, name: str, value: Any) -> None:
    if _inpaint_preview_requested(previews, name):
        previews[name] = value


def _filename_only(filename: str) -> str:
    return strip_category_prefix(filename)[1]


def _node_output_first(value: Any) -> Any:
    if hasattr(value, "result"):
        result = value.result
        if isinstance(result, tuple):
            return result[0]
        return result
    if isinstance(value, tuple):
        return value[0]
    return value


def _phase(progress: Any, message: str) -> None:
    if progress is not None:
        progress.phase(message)


def _size_info(width_height: tuple[int, int] | None) -> dict[str, int] | None:
    if width_height is None:
        return None
    return {"width": int(width_height[0]), "height": int(width_height[1])}


def _validate_second_pass_float(
    value: Any,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if resolved < minimum or resolved > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return resolved


def _validate_second_pass_int(
    value: Any,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    try:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if resolved < minimum or resolved > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return resolved


def normalize_second_pass_config(config: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    source = dict(config or {})
    source.update({key: value for key, value in overrides.items() if value is not None})
    upscale_method = str(source.get("upscale_method", "lanczos")).lower()
    if upscale_method not in SECOND_PASS_UPSCALE_METHODS:
        raise ValueError(
            "second_pass_upscale_method must be one of "
            f"{', '.join(SECOND_PASS_UPSCALE_METHODS)}."
        )
    return {
        "enabled": bool(source.get("enabled", False)),
        "denoise": _validate_second_pass_float(
            source.get("denoise", 0.15),
            "second_pass_denoise",
            0.0,
            1.0,
        ),
        "steps_input": _validate_second_pass_int(
            source.get("steps_input", source.get("steps", 0)),
            "second_pass_steps",
            0,
            100,
        ),
        "upscale_ratio": _validate_second_pass_float(
            source.get("upscale_ratio", 1.5),
            "second_pass_upscale_ratio",
            1.0,
            8.0,
        ),
        "upscale_method": upscale_method,
        "decode_image": bool(source.get("decode_image", True)),
        "return_image_original": bool(source.get("return_image_original", False)),
    }


def resolve_second_pass_steps(config: dict[str, Any], main_steps: int) -> int:
    steps_input = int(config.get("steps_input", 0))
    if steps_input <= 0:
        return max(1, int(main_steps))
    return steps_input


def second_pass_status(
    config: dict[str, Any],
    *,
    applied: bool = False,
    first_pass_size: tuple[int, int] | None = None,
    final_size: tuple[int, int] | None = None,
    main_steps: int | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    steps_input = int(config.get("steps_input", 0))
    steps = config.get("steps")
    if steps is None and main_steps is not None:
        steps = resolve_second_pass_steps(config, main_steps)
    info = {
        "enabled": bool(config.get("enabled", False)),
        "applied": bool(applied),
        "denoise": float(config.get("denoise", 0.15)),
        "steps_input": steps_input,
        "steps": int(steps) if steps is not None else None,
        "upscale_ratio": float(config.get("upscale_ratio", 1.5)),
        "upscale_method": str(config.get("upscale_method", "lanczos")),
        "first_pass_size": _size_info(first_pass_size),
        "final_size": _size_info(final_size),
    }
    if reason:
        info["reason"] = reason
    return info


def _has_reference_attention_context(reference_inputs: Any = None, inpaint_config: Any = None) -> bool:
    if inpaint_config is not None:
        return True
    if reference_inputs is None:
        return False
    return bool(getattr(reference_inputs, "mask", None) is not None or getattr(reference_inputs, "images", ()))


def apply_model_performance(
    *,
    model: Any,
    settings: dict[str, Any],
    reference_inputs: Any = None,
    inpaint_config: Any = None,
) -> Any:
    return apply_performance_settings(
        model=model,
        settings=settings,
        has_mask_or_reference=_has_reference_attention_context(reference_inputs, inpaint_config),
    )


def _apply_model_performance_if_configured(
    *,
    model: Any,
    settings: dict[str, Any],
    reference_inputs: Any = None,
    inpaint_config: Any = None,
    progress: Any = None,
) -> Any:
    if performance_settings_present(settings):
        _phase(progress, "applying performance settings")
    return apply_model_performance(
        model=model,
        settings=settings,
        reference_inputs=reference_inputs,
        inpaint_config=inpaint_config,
    )


def _apply_memory_policy_before_sampling_if_configured(
    *,
    settings: dict[str, Any],
    progress: Any = None,
) -> None:
    if "memory_policy" in settings:
        _phase(progress, "applying memory policy")
    apply_memory_policy_before_sampling(settings)


def _filter_duplicate_inpaint_reference_images(
    reference_images: tuple[Any, ...],
    *,
    inpaint_config: dict[str, Any] | None,
    settings: dict[str, Any],
) -> tuple[Any, ...]:
    if inpaint_config is None or not reference_images:
        return reference_images

    inpaint_image = inpaint_config.get("image")
    if inpaint_image is None:
        return reference_images

    filtered = tuple(reference for reference in reference_images if reference is not inpaint_image)
    skipped_count = len(reference_images) - len(filtered)
    if skipped_count > 0:
        settings["duplicate_inpaint_reference_skipped"] = True
        settings["duplicate_inpaint_reference_count"] = skipped_count
    return filtered


def _clip_type(name: str):
    import comfy.sd  # type: ignore

    return getattr(comfy.sd.CLIPType, name.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)


def _model_options_from_precision(precision_policy: str | None) -> dict[str, Any]:
    if precision_policy in (None, "auto"):
        return {}
    import torch  # type: ignore

    if precision_policy == "fp8":
        return {"dtype": torch.float8_e4m3fn}
    if precision_policy == "bf16":
        return {"dtype": torch.bfloat16}
    return {}


def _resolve_node_class(name: str):
    import nodes  # type: ignore

    try:
        return nodes.NODE_CLASS_MAPPINGS[name]
    except KeyError as exc:
        raise ValueError(gguf_backend.explain_missing()) from exc


def load_diffusion_model(
    *,
    diffusion_model: str,
    precision_policy: str | None = None,
):
    if infer_model_format(diffusion_model) == "gguf":
        loader_cls = _resolve_node_class("UnetLoaderGGUF")
        return loader_cls().load_unet(_filename_only(diffusion_model))[0]

    import comfy.sd  # type: ignore

    path = safetensors_backend.diffusion_model_path(diffusion_model)
    return comfy.sd.load_diffusion_model(
        str(path),
        model_options=_model_options_from_precision(precision_policy),
    )


def load_text_encoder(
    *,
    text_encoder: str,
    clip_type: str,
):
    if infer_model_format(text_encoder) == "gguf":
        loader_cls = _resolve_node_class("CLIPLoaderGGUF")
        return loader_cls().load_clip(_filename_only(text_encoder), type=clip_type)[0]

    import comfy.sd  # type: ignore
    import folder_paths  # type: ignore

    path = safetensors_backend.text_encoder_path(text_encoder)
    return comfy.sd.load_clip(
        ckpt_paths=[str(path)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=_clip_type(clip_type),
        model_options={},
    )


def text_encoder_clip_type(model_type: str) -> str:
    if model_type == "ideogram4":
        return "ideogram4"
    if model_type == "flux2_klein_9b":
        return "flux2"
    if model_type == "krea2":
        return "krea2"
    return "stable_diffusion"


def load_vae(*, vae: str):
    _, vae_name = strip_category_prefix(vae)
    if infer_model_format(vae) == "gguf":
        loader_cls = _resolve_node_class("VaeGGUF")
        return loader_cls().load_vae(vae_name)[0]

    import nodes  # type: ignore

    return nodes.VAELoader().load_vae(vae_name)[0]


def encode_z_image_prompt(*, clip: Any, prompt: str):
    from comfy_extras.nodes_zimage import TextEncodeZImageOmni  # type: ignore

    return _node_output_first(TextEncodeZImageOmni.execute(clip=clip, prompt=prompt))


def encode_flux2_prompt(*, clip: Any, prompt: str, guidance: float):
    import node_helpers  # type: ignore

    tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    return node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})


def encode_ideogram4_prompt(*, clip: Any, prompt: str):
    import nodes  # type: ignore

    return nodes.CLIPTextEncode().encode(clip, prompt)[0]


def normalize_krea2_max_length(value: Any) -> int:
    if value in (None, ""):
        return KREA2_DEFAULT_MAX_LENGTH
    try:
        max_length = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Krea 2 max_length must be an integer.") from exc
    if max_length < KREA2_MIN_MAX_LENGTH or max_length > KREA2_MAX_MAX_LENGTH:
        raise ValueError(
            f"Krea 2 max_length must be between {KREA2_MIN_MAX_LENGTH} and {KREA2_MAX_MAX_LENGTH}."
        )
    return max_length


def _clip_tokenizer_max_length_targets(clip: Any) -> tuple[Any, ...]:
    tokenizer = getattr(clip, "tokenizer", None)
    if tokenizer is None:
        return ()

    targets: list[Any] = []

    def add(target: Any) -> None:
        if (
            target is not None
            and hasattr(target, "max_length")
            and not any(target is seen for seen in targets)
        ):
            targets.append(target)

    add(tokenizer)
    for name in (
        getattr(tokenizer, "clip", None),
        getattr(tokenizer, "clip_name", None),
        "qwen3vl_4b",
    ):
        if isinstance(name, str):
            add(getattr(tokenizer, name, None))
    return tuple(targets)


def _tokenize_with_max_length(clip: Any, prompt: str, max_length: int):
    previous: list[tuple[Any, Any]] = []
    try:
        for target in _clip_tokenizer_max_length_targets(clip):
            previous.append((target, target.max_length))
            target.max_length = max_length
        return clip.tokenize(prompt)
    finally:
        for target, old_max_length in previous:
            target.max_length = old_max_length


def encode_krea2_prompt(*, clip: Any, prompt: str, max_length: int | None = None):
    if max_length is None:
        import nodes  # type: ignore

        return nodes.CLIPTextEncode().encode(clip, prompt)[0]

    tokens = _tokenize_with_max_length(
        clip,
        prompt,
        normalize_krea2_max_length(max_length),
    )
    return clip.encode_from_tokens_scheduled(tokens)


def zero_out_conditioning(conditioning: Any):
    import nodes  # type: ignore

    return nodes.ConditioningZeroOut().zero_out(conditioning)[0]


def use_zero_negative_conditioning(settings: dict[str, Any]) -> bool:
    return bool(settings.get("use_zero_negative_conditioning", True))


def scale_image_to_total_pixels(
    *,
    image: Any,
    megapixels: float = 1.0,
    upscale_method: str = "area",
    resolution_steps: int = 1,
    multiple_value: str | int | None = "none",
):
    import comfy.utils  # type: ignore

    samples = image.movedim(-1, 1)
    total = megapixels * 1024 * 1024
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    multiple = parse_multiple_value(multiple_value)
    width = round_to_multiple(
        round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps,
        multiple,
    )
    height = round_to_multiple(
        round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps,
        multiple,
    )
    resized = comfy.utils.common_upscale(samples, int(width), int(height), upscale_method, "center")
    return resized.movedim(1, -1)


def encode_image_to_latent(*, vae: Any, image: Any):
    import nodes  # type: ignore

    return nodes.VAEEncode().encode(vae, image)[0]


def upscale_image_by_ratio(
    *,
    image: Any,
    upscale_ratio: float,
    upscale_method: str,
    dimension_multiple: int | None,
) -> tuple[Any, int, int]:
    import comfy.utils  # type: ignore

    dimensions = image_tensor_dimensions(image)
    if dimensions is None:
        raise ValueError("second pass source image must be an IMAGE tensor with shape [B, H, W, C].")
    source_width, source_height = dimensions
    target_width = round_to_multiple(source_width * float(upscale_ratio), dimension_multiple)
    target_height = round_to_multiple(source_height * float(upscale_ratio), dimension_multiple)
    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(
        samples,
        int(target_width),
        int(target_height),
        str(upscale_method),
        "disabled",
    )
    return resized.movedim(1, -1), int(target_width), int(target_height)


def apply_second_sampler_pass(
    *,
    config: dict[str, Any] | None,
    image: Any,
    latent: dict[str, Any],
    vae: str,
    loaded_vae: Any,
    sample_latent: Callable[[dict[str, Any], int, int, float, int], dict[str, Any]],
    dimension_multiple: int | None,
    main_steps: int,
    progress: Any = None,
) -> tuple[Any, dict[str, Any], Any]:
    second_pass = normalize_second_pass_config(config)
    second_pass["steps"] = resolve_second_pass_steps(second_pass, main_steps)
    if not second_pass["enabled"]:
        return image, latent, loaded_vae

    first_pass_size = image_tensor_dimensions(image)
    if image is None or first_pass_size is None:
        out = latent.copy()
        out[SECOND_PASS_INFO_KEY] = second_pass_status(
            second_pass,
            applied=False,
            reason="first_pass_image_unavailable",
        )
        return image, out, loaded_vae

    if loaded_vae is None:
        _phase(progress, "loading vae")
        loaded_vae = load_vae(vae=vae)

    _phase(progress, "upscaling second pass")
    upscaled_image, upscaled_width, upscaled_height = upscale_image_by_ratio(
        image=image,
        upscale_ratio=float(second_pass["upscale_ratio"]),
        upscale_method=str(second_pass["upscale_method"]),
        dimension_multiple=dimension_multiple,
    )
    _phase(progress, "encoding second pass image")
    second_latent = encode_image_to_latent(vae=loaded_vae, image=upscaled_image)

    if float(second_pass["denoise"]) <= 0.0:
        sampled_latent = second_latent
    else:
        _phase(progress, "sampling second pass")
        sampled_latent = sample_latent(
            second_latent,
            upscaled_width,
            upscaled_height,
            float(second_pass["denoise"]),
            int(second_pass["steps"]),
        )

    sampled_latent = sampled_latent.copy()
    if isinstance(latent, dict) and PID_CAPTURE_KEY in latent:
        sampled_latent[PID_CAPTURE_KEY] = latent[PID_CAPTURE_KEY]
    sampled_latent[SECOND_PASS_INFO_KEY] = second_pass_status(
        second_pass,
        applied=True,
        first_pass_size=first_pass_size,
        final_size=(upscaled_width, upscaled_height),
    )
    if second_pass.get("return_image_original"):
        sampled_latent[SECOND_PASS_ORIGINAL_IMAGE_KEY] = image

    output_image = None
    if second_pass.get("decode_image"):
        _phase(progress, "decoding second pass")
        output_image = decode_latent(vae=loaded_vae, latent=sampled_latent)
    return output_image, sampled_latent, loaded_vae


def apply_reference_latents_to_conditioning(
    *,
    positive: Any,
    negative: Any,
    reference_latents: list[dict[str, Any]],
):
    import node_helpers  # type: ignore

    for latent in reference_latents:
        value = {"reference_latents": [latent["samples"]]}
        positive = node_helpers.conditioning_set_values(positive, value, append=True)
        negative = node_helpers.conditioning_set_values(negative, value, append=True)
    return positive, negative


def make_empty_z_image_latent(*, width: int, height: int):
    import nodes  # type: ignore

    return nodes.EmptyLatentImage().generate(width, height, batch_size=1)[0]


def make_empty_flux2_latent(*, width: int, height: int):
    import comfy.model_management  # type: ignore
    import torch  # type: ignore

    latent = torch.zeros(
        [1, 128, height // 16, width // 16],
        device=comfy.model_management.intermediate_device(),
    )
    return {"samples": latent}


def make_empty_ideogram4_latent(*, width: int, height: int):
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage  # type: ignore

    return _node_output_first(EmptyFlux2LatentImage.execute(width=width, height=height, batch_size=1))


def make_empty_krea2_latent(*, width: int, height: int):
    import nodes  # type: ignore

    return nodes.EmptyLatentImage().generate(width, height, batch_size=1)[0]


def resolve_pid_capture_step(capture_step: int | None, steps: int) -> int | None:
    if capture_step is None:
        return None
    effective_steps = max(1, int(steps))
    step = int(capture_step)
    if step <= 0:
        step = effective_steps if effective_steps <= 4 else effective_steps - 4
    return min(max(1, step), effective_steps)


def _copy_pid_capture_latent(latent: dict[str, Any], samples: Any, sigma: float, step: int) -> dict[str, Any]:
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    out["pid_sigma"] = float(sigma)
    out["pid_capture_step"] = int(step)
    return out


def _attach_pid_capture(
    *,
    latent: dict[str, Any],
    source_latent: dict[str, Any],
    captured: dict[str, Any],
    fallback_samples: Any,
    target_step: int | None,
) -> dict[str, Any]:
    if target_step is None:
        return latent
    samples = captured.get("samples")
    sigma = captured.get("sigma")
    if samples is None:
        samples = fallback_samples
        sigma = 0.0
    capture_latent = _copy_pid_capture_latent(
        source_latent,
        samples,
        float(sigma if sigma is not None else 0.0),
        target_step,
    )
    out = latent.copy()
    out[PID_CAPTURE_KEY] = {
        "latent": capture_latent,
        "sigma": float(sigma if sigma is not None else 0.0),
        "step": int(target_step),
    }
    return out


def _capture_sigma_at(sigmas: Any, step_index: int) -> float:
    try:
        return float(sigmas[int(step_index)].detach().float().cpu().item())
    except Exception:
        try:
            return float(sigmas[int(step_index)])
        except Exception:
            return 0.0


def _prepare_sampling_callback(
    *,
    model: Any,
    steps: int,
    progress: Any = None,
    x0_output_dict: dict[str, Any] | None = None,
):
    if progress is not None and hasattr(progress, "prepare_sampling_callback"):
        try:
            return progress.prepare_sampling_callback(model, steps, x0_output_dict=x0_output_dict)
        except TypeError:
            return progress.prepare_sampling_callback(model, steps)
        except Exception:
            pass

    import latent_preview  # type: ignore

    try:
        return latent_preview.prepare_callback(model, steps, x0_output_dict)
    except TypeError:
        return latent_preview.prepare_callback(model, steps)


def sample_with_comfy_ksampler(
    *,
    model: Any,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
    denoise: float = 1.0,
    pid_capture_step: int | None = None,
    progress: Any = None,
):
    import comfy.sample  # type: ignore
    import comfy.samplers  # type: ignore
    import comfy.utils  # type: ignore

    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    noise_mask = latent.get("noise_mask")
    sampler_obj = comfy.samplers.KSampler(
        model,
        steps=steps,
        device=model.load_device,
        sampler=sampler,
        scheduler=scheduler,
        denoise=float(denoise),
        model_options=model.model_options,
    )
    sigmas = sampler_obj.sigmas
    effective_steps = max(0, int(sigmas.shape[0]) - 1)
    target_step = resolve_pid_capture_step(pid_capture_step, effective_steps)
    captured: dict[str, Any] = {}
    preview_callback = _prepare_sampling_callback(
        model=model,
        steps=effective_steps or steps,
        progress=progress,
    )

    def callback(step, x0, x, total_steps):
        preview_callback(step, x0, x, total_steps)
        if target_step is not None and int(step) + 1 == target_step:
            captured["samples"] = x.detach().to("cpu").contiguous()
            captured["sigma"] = _capture_sigma_at(sigmas, int(step))

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=float(denoise),
        disable_noise=False,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return _attach_pid_capture(
        latent=out,
        source_latent=latent,
        captured=captured,
        fallback_samples=samples,
        target_step=target_step,
    )


def _pid_capture_callback_from_sigmas(
    *,
    model: Any,
    sigmas: Any,
    pid_capture_step: int | None,
    progress: Any = None,
    x0_output_dict: dict[str, Any] | None = None,
):
    effective_steps = max(0, int(sigmas.shape[-1]) - 1)
    target_step = resolve_pid_capture_step(pid_capture_step, effective_steps)
    captured: dict[str, Any] = {}
    preview_callback = _prepare_sampling_callback(
        model=model,
        steps=effective_steps,
        progress=progress,
        x0_output_dict=x0_output_dict,
    )

    def callback(step, x0, x, total_steps):
        preview_callback(step, x0, x, total_steps)
        if target_step is not None and int(step) + 1 == target_step:
            captured["samples"] = x.detach().to("cpu").contiguous()
            captured["sigma"] = _capture_sigma_at(sigmas, int(step))

    return callback, captured, target_step


def sample_with_sigmas(
    *,
    model: Any,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
    sigmas: Any,
    pid_capture_step: int | None = None,
    progress: Any = None,
):
    import comfy.sample  # type: ignore
    import comfy.utils  # type: ignore

    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    callback, captured, target_step = _pid_capture_callback_from_sigmas(
        model=model,
        sigmas=sigmas,
        pid_capture_step=pid_capture_step,
        progress=progress,
    )
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    noise_mask = latent.get("noise_mask")
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler,
        scheduler,
        positive,
        negative,
        latent_image,
        sigmas=sigmas,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return _attach_pid_capture(
        latent=out,
        source_latent=latent,
        captured=captured,
        fallback_samples=samples,
        target_step=target_step,
    )


def decode_latent(*, vae: Any, latent: dict[str, Any]):
    import nodes  # type: ignore

    return nodes.VAEDecode().decode(vae, latent)[0]


def apply_lora_config_model_only(
    *,
    model: Any,
    lora_config: dict[str, Any] | None,
) -> tuple[Any, list[dict[str, Any]]]:
    normalized = normalize_lora_config(lora_config)
    loras = normalized["loras"]
    if not loras:
        return model, []

    import nodes  # type: ignore

    loader = nodes.LoraLoaderModelOnly()
    applied: list[dict[str, Any]] = []
    for lora in loras:
        model = loader.load_lora_model_only(
            model,
            lora["name"],
            lora["strength_model"],
        )[0]
        applied.append(dict(lora))
    return model, applied


def apply_model_sampling_aura(*, model: Any, shift: float):
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow  # type: ignore

    return ModelSamplingAuraFlow().patch_aura(model, shift)[0]


def apply_cfg_override(
    *,
    model: Any,
    cfg: float,
    start_percent: float,
    end_percent: float,
):
    from comfy_extras.nodes_custom_sampler import CFGOverride  # type: ignore

    return _node_output_first(
        CFGOverride.execute(
            model=model,
            cfg=cfg,
            start_percent=start_percent,
            end_percent=end_percent,
        )
    )


def build_dual_model_guider(
    *,
    model: Any,
    model_negative: Any,
    positive: Any,
    negative: Any,
    cfg: float,
):
    from comfy_extras.nodes_custom_sampler import DualModelGuider  # type: ignore

    return _node_output_first(
        DualModelGuider.execute(
            model=model,
            positive=positive,
            cfg=cfg,
            model_negative=model_negative,
            negative=negative,
        )
    )


def ideogram4_sigmas(*, steps: int, width: int, height: int, mu: float, std: float):
    from comfy_extras.nodes_ideogram4 import ideogram4_sigmas as make_sigmas  # type: ignore

    return make_sigmas(steps, width, height, mu, std)


def basic_sigmas(*, model: Any, scheduler: str, steps: int):
    from comfy_extras.nodes_custom_sampler import BasicScheduler  # type: ignore

    return _node_output_first(
        BasicScheduler.execute(
            model=model,
            scheduler=scheduler,
            steps=steps,
            denoise=1.0,
        )
    )


def sample_with_custom_guider(
    *,
    guider: Any,
    seed: int,
    sampler: str,
    sigmas: Any,
    latent: dict[str, Any],
    pid_capture_step: int | None = None,
    progress: Any = None,
):
    import comfy.model_management  # type: ignore
    import comfy.sample  # type: ignore
    import comfy.utils  # type: ignore
    from comfy_extras.nodes_custom_sampler import KSamplerSelect, RandomNoise  # type: ignore

    noise = _node_output_first(RandomNoise.execute(noise_seed=seed))
    sampler_obj = _node_output_first(KSamplerSelect.execute(sampler_name=sampler))
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    sampling_latent = latent.copy()
    sampling_latent["samples"] = latent_image
    noise_mask = sampling_latent.get("noise_mask")
    callback, captured, target_step = _pid_capture_callback_from_sigmas(
        model=guider.model_patcher,
        sigmas=sigmas,
        pid_capture_step=pid_capture_step,
        progress=progress,
    )
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(
        noise.generate_noise(sampling_latent),
        latent_image,
        sampler_obj,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=noise.seed,
    )
    if hasattr(samples, "to"):
        samples = samples.to(comfy.model_management.intermediate_device())
    sampled = sampling_latent.copy()
    sampled.pop("downscale_ratio_spacial", None)
    sampled.pop("downscale_ratio_temporal", None)
    sampled["samples"] = samples
    if target_step is None:
        return sampled
    return _attach_pid_capture(
        latent=sampled,
        source_latent=latent,
        captured=captured,
        fallback_samples=samples,
        target_step=target_step,
    )


def flux2_sigmas(*, steps: int, width: int, height: int):
    from comfy_extras import nodes_flux  # type: ignore

    seq_len = width * height / (16 * 16)
    return nodes_flux.get_schedule(steps, round(seq_len))


def generate_ideogram4_t2i(
    *,
    diffusion_model: str,
    unconditional_model: str,
    text_encoder: str,
    vae: str,
    positive_prompt: str,
    negative_prompt: str = "",
    width: int,
    height: int,
    seed: int,
    steps: int,
    sampler: str,
    scheduler: str,
    settings: dict[str, Any],
    batch_count: int = 1,
    lora_config: dict[str, Any] | None = None,
    loaded_model: Any = None,
    loaded_clip: Any = None,
    inpaint_config: dict[str, Any] | None = None,
    inpaint_previews: dict[str, Any] | None = None,
    decode_image: bool = True,
    return_vae: bool = False,
    second_pass_config: dict[str, Any] | None = None,
    second_pass_dimension_multiple: int | None = 16,
    pid_capture_step: int | None = None,
    progress: Any = None,
):
    using_connected_model_pair = loaded_model is not None and loaded_clip is not None
    model = loaded_model
    if model is None:
        _phase(progress, "loading diffusion model")
        model = load_diffusion_model(
            diffusion_model=diffusion_model,
            precision_policy=settings.get("precision_policy"),
        )
    run_unconditional_model = bool(settings.get("run_unconditional_model", True))
    model_negative = None
    if run_unconditional_model:
        _phase(progress, "loading unconditional diffusion model")
        model_negative = load_diffusion_model(
            diffusion_model=unconditional_model,
            precision_policy=settings.get("precision_policy"),
        )
    clip = loaded_clip
    if clip is None:
        _phase(progress, "loading text encoder")
        clip = load_text_encoder(
            text_encoder=text_encoder,
            clip_type=text_encoder_clip_type("ideogram4"),
        )
    _phase(progress, "applying model sampling")
    model = apply_model_sampling_aura(
        model=model,
        shift=float(settings.get("sampling_shift", 5.0)),
    )
    scheduler_model = model
    apply_timing = normalize_performance_apply_timing(settings)
    if apply_timing == "before_loras":
        model = _apply_model_performance_if_configured(model=model, settings=settings, progress=progress)
        if model_negative is not None:
            model_negative = _apply_model_performance_if_configured(model=model_negative, settings=settings, progress=progress)
    if not using_connected_model_pair:
        _phase(progress, "applying loras")
        model, _ = apply_lora_config_model_only(model=model, lora_config=lora_config)
    if apply_timing == "after_loras":
        model = _apply_model_performance_if_configured(model=model, settings=settings, progress=progress)
        if model_negative is not None:
            model_negative = _apply_model_performance_if_configured(model=model_negative, settings=settings, progress=progress)
    if settings.get("cfg_override_enabled", True):
        _phase(progress, "applying cfg override")
        model = apply_cfg_override(
            model=model,
            cfg=float(settings.get("cfg_override", 3.0)),
            start_percent=float(settings.get("cfg_override_start_percent", 0.7)),
            end_percent=float(settings.get("cfg_override_end_percent", 1.0)),
        )
    _phase(progress, "encoding prompts")
    positive = encode_ideogram4_prompt(clip=clip, prompt=positive_prompt)
    if use_zero_negative_conditioning(settings):
        negative = zero_out_conditioning(positive)
    else:
        negative = encode_ideogram4_prompt(clip=clip, prompt=negative_prompt or "")
    loaded_vae = None
    if inpaint_config is not None:
        _phase(progress, "loading vae")
        loaded_vae = load_vae(vae=vae)
        _phase(progress, "preparing inpaint source")
        inpaint_source = inpaint_service.prepare_inpaint_source(
            config=inpaint_config,
            width=width,
            height=height,
        )
        _set_inpaint_preview(inpaint_previews, INPAINT_PREVIEW_SOURCE, inpaint_source.image)
        _set_inpaint_preview(inpaint_previews, INPAINT_PREVIEW_MASK, inpaint_source.sampling_mask)
        _phase(progress, "encoding inpaint image")
        latent = inpaint_service.encode_inpaint_source_latent(
            vae=loaded_vae,
            source=inpaint_source,
        )
    else:
        inpaint_source = None
        latent = make_empty_ideogram4_latent(width=width, height=height)
    sampling_width, sampling_height = (
        inpaint_source.working_dimensions(fallback_width=width, fallback_height=height)
        if inpaint_source is not None
        else (width, height)
    )
    _phase(progress, "preparing guider")
    guider = build_dual_model_guider(
        model=model,
        model_negative=model_negative,
        positive=positive,
        negative=negative,
        cfg=float(settings.get("dual_cfg", settings.get("cfg", 7.0))),
    )
    inpaint_denoise = float(inpaint_config.get("denoise", 1.0)) if inpaint_config is not None else 1.0
    inpaint_steps = inpaint_service.resolve_inpaint_steps(inpaint_config, steps)
    decode_inpaint_sample = (
        inpaint_config is not None
        and _inpaint_preview_requested(inpaint_previews, INPAINT_PREVIEW_SAMPLE)
    )
    use_sampling = not (inpaint_config is not None and inpaint_denoise <= 0.0)
    sigmas = None
    if use_sampling:
        schedule_steps = (
            inpaint_service.resolve_denoise_schedule_steps(inpaint_steps, inpaint_denoise)
            if inpaint_config is not None
            else steps
        )
        if settings.get("schedule_mode") == "basic":
            sigmas = basic_sigmas(model=scheduler_model, scheduler=scheduler, steps=schedule_steps)
        else:
            sigmas = ideogram4_sigmas(
                steps=schedule_steps,
                width=sampling_width,
                height=sampling_height,
                mu=float(settings.get("mu", 0.0)),
                std=float(settings.get("std", 1.75)),
            )
        if inpaint_config is not None:
            sigmas = inpaint_service.apply_denoise_to_sigmas(
                sigmas,
                inpaint_denoise,
                steps=inpaint_steps,
            )
    batch_seeds = incrementing_batch_seeds(seed, batch_count)
    images: list[Any] = []
    latents: list[Any] = []
    inpaint_sample_images: list[Any] = []

    for index, batch_seed in enumerate(batch_seeds):
        if not use_sampling:
            sampled_latent = latent
        else:
            _phase(progress, "sampling" if len(batch_seeds) == 1 else f"sampling {index + 1}/{len(batch_seeds)}")
            sampled_latent = sample_with_custom_guider(
                guider=guider,
                seed=batch_seed,
                sampler=sampler,
                sigmas=sigmas,
                latent=latent,
                pid_capture_step=pid_capture_step,
                progress=progress,
            )
        image = None
        if decode_image or decode_inpaint_sample or return_vae:
            if loaded_vae is None:
                _phase(progress, "loading vae")
                loaded_vae = load_vae(vae=vae)
        if decode_image or decode_inpaint_sample:
            _phase(progress, "decoding")
            decoded_image = decode_latent(vae=loaded_vae, latent=sampled_latent)
            if decode_inpaint_sample:
                inpaint_sample_images.append(decoded_image)
            image = decoded_image if decode_image else None
        if decode_image:
            if inpaint_config is not None and inpaint_source is not None:
                if inpaint_source.stitcher is not None:
                    _phase(progress, "stitching inpaint")
                    image = inpaint_service.stitch_inpaint_image(
                        stitcher=inpaint_source.stitcher,
                        inpainted_image=image,
                    )
                elif bool(inpaint_config.get("final_blend", True)):
                    _phase(progress, "blending inpaint")
                    image = inpaint_service.blend_inpaint_image(
                        source_image=inpaint_source.image,
                        generated_image=image,
                        mask=inpaint_source.mask,
                        feather=int(inpaint_config.get("mask_feather", 24)),
                    )

        def sample_second_pass(
            second_latent: dict[str, Any],
            second_width: int,
            second_height: int,
            denoise: float,
            second_steps: int,
        ):
            if settings.get("schedule_mode") == "basic":
                second_sigmas = basic_sigmas(model=scheduler_model, scheduler=scheduler, steps=second_steps)
            else:
                second_sigmas = ideogram4_sigmas(
                    steps=second_steps,
                    width=second_width,
                    height=second_height,
                    mu=float(settings.get("mu", 0.0)),
                    std=float(settings.get("std", 1.75)),
                )
            second_sigmas = inpaint_service.apply_denoise_to_sigmas(second_sigmas, denoise)
            return sample_with_custom_guider(
                guider=guider,
                seed=batch_seed,
                sampler=sampler,
                sigmas=second_sigmas,
                latent=second_latent,
                progress=progress,
            )

        image, sampled_latent, loaded_vae = apply_second_sampler_pass(
            config=second_pass_config,
            image=image,
            latent=sampled_latent,
            vae=vae,
            loaded_vae=loaded_vae,
            sample_latent=sample_second_pass,
            dimension_multiple=second_pass_dimension_multiple,
            main_steps=steps,
            progress=progress,
        )
        images.append(image)
        latents.append(sampled_latent)

    if decode_inpaint_sample:
        _set_inpaint_preview(
            inpaint_previews,
            INPAINT_PREVIEW_SAMPLE,
            _concat_optional_tensors(inpaint_sample_images),
        )
    return GenerationResult(
        image=_concat_optional_tensors(images),
        latent=_combine_latent_batch(latents),
        positive=positive,
        negative=negative,
        vae=loaded_vae,
        model=model,
        clip=clip,
    )


def generate_z_image_turbo_t2i(
    *,
    diffusion_model: str,
    text_encoder: str,
    vae: str,
    positive_prompt: str,
    negative_prompt: str = "",
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    settings: dict[str, Any],
    batch_count: int = 1,
    lora_config: dict[str, Any] | None = None,
    loaded_model: Any = None,
    loaded_clip: Any = None,
    decode_image: bool = True,
    return_vae: bool = False,
    second_pass_config: dict[str, Any] | None = None,
    second_pass_dimension_multiple: int | None = 16,
    pid_capture_step: int | None = None,
    progress: Any = None,
):
    using_connected_model_pair = loaded_model is not None and loaded_clip is not None
    model = loaded_model
    if model is None:
        _phase(progress, "loading diffusion model")
        model = load_diffusion_model(
            diffusion_model=diffusion_model,
            precision_policy=settings.get("precision_policy"),
        )
    clip = loaded_clip
    if clip is None:
        _phase(progress, "loading text encoder")
        clip = load_text_encoder(
            text_encoder=text_encoder,
            clip_type=text_encoder_clip_type("z_image_turbo"),
        )
    apply_timing = normalize_performance_apply_timing(settings)
    if apply_timing == "before_loras":
        model = _apply_model_performance_if_configured(model=model, settings=settings, progress=progress)
    if not using_connected_model_pair:
        _phase(progress, "applying loras")
        model, clip, _ = apply_lora_config(model=model, clip=clip, lora_config=lora_config)
    if apply_timing == "after_loras":
        model = _apply_model_performance_if_configured(model=model, settings=settings, progress=progress)
    _phase(progress, "encoding prompts")
    positive = encode_z_image_prompt(clip=clip, prompt=positive_prompt)
    if use_zero_negative_conditioning(settings):
        negative = zero_out_conditioning(positive)
    else:
        negative = encode_z_image_prompt(clip=clip, prompt=negative_prompt or "")
    batch_seeds = incrementing_batch_seeds(seed, batch_count)
    images: list[Any] = []
    latents: list[Any] = []
    loaded_vae = None

    for index, batch_seed in enumerate(batch_seeds):
        latent = make_empty_z_image_latent(width=width, height=height)
        _phase(progress, "sampling" if len(batch_seeds) == 1 else f"sampling {index + 1}/{len(batch_seeds)}")
        sampled_latent = sample_with_comfy_ksampler(
            model=model,
            seed=batch_seed,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent,
            pid_capture_step=pid_capture_step,
            progress=progress,
        )
        image = None
        if decode_image or return_vae:
            if loaded_vae is None:
                _phase(progress, "loading vae")
                loaded_vae = load_vae(vae=vae)
        if decode_image:
            _phase(progress, "decoding")
            image = decode_latent(vae=loaded_vae, latent=sampled_latent)

        def sample_second_pass(
            second_latent: dict[str, Any],
            second_width: int,
            second_height: int,
            denoise: float,
            second_steps: int,
        ):
            del second_width, second_height
            return sample_with_comfy_ksampler(
                model=model,
                seed=batch_seed,
                steps=second_steps,
                cfg=cfg,
                sampler=sampler,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=second_latent,
                denoise=denoise,
                progress=progress,
            )

        image, sampled_latent, loaded_vae = apply_second_sampler_pass(
            config=second_pass_config,
            image=image,
            latent=sampled_latent,
            vae=vae,
            loaded_vae=loaded_vae,
            sample_latent=sample_second_pass,
            dimension_multiple=second_pass_dimension_multiple,
            main_steps=steps,
            progress=progress,
        )
        images.append(image)
        latents.append(sampled_latent)

    return GenerationResult(
        image=_concat_optional_tensors(images),
        latent=_combine_latent_batch(latents),
        positive=positive,
        negative=negative,
        vae=loaded_vae,
        model=model,
        clip=clip,
    )


def generate_krea2_t2i(
    *,
    diffusion_model: str,
    text_encoder: str,
    vae: str,
    positive_prompt: str,
    negative_prompt: str = "",
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    settings: dict[str, Any],
    batch_count: int = 1,
    lora_config: dict[str, Any] | None = None,
    loaded_model: Any = None,
    loaded_clip: Any = None,
    inpaint_config: dict[str, Any] | None = None,
    inpaint_previews: dict[str, Any] | None = None,
    decode_image: bool = True,
    return_vae: bool = False,
    second_pass_config: dict[str, Any] | None = None,
    second_pass_dimension_multiple: int | None = 16,
    pid_capture_step: int | None = None,
    progress: Any = None,
):
    using_connected_model_pair = loaded_model is not None and loaded_clip is not None
    model = loaded_model
    if model is None:
        _phase(progress, "loading diffusion model")
        model = load_diffusion_model(
            diffusion_model=diffusion_model,
            precision_policy=settings.get("precision_policy"),
        )
    clip = loaded_clip
    if clip is None:
        _phase(progress, "loading text encoder")
        clip = load_text_encoder(
            text_encoder=text_encoder,
            clip_type=text_encoder_clip_type("krea2"),
        )
    apply_timing = normalize_performance_apply_timing(settings)
    if apply_timing == "before_loras":
        model = _apply_model_performance_if_configured(
            model=model,
            settings=settings,
            inpaint_config=inpaint_config,
            progress=progress,
        )
    if not using_connected_model_pair:
        _phase(progress, "applying loras")
        model, clip, _ = apply_lora_config(model=model, clip=clip, lora_config=lora_config)
    if apply_timing == "after_loras":
        model = _apply_model_performance_if_configured(
            model=model,
            settings=settings,
            inpaint_config=inpaint_config,
            progress=progress,
        )
    if settings.get("enhancer_enabled", True):
        _phase(progress, "applying Krea2T enhancer")
    model = krea2_enhancer.apply_krea2_enhancer(
        model,
        enabled=bool(settings.get("enhancer_enabled", True)),
        strength=settings.get("enhancer_strength", 1.0),
    )
    _phase(progress, "encoding prompts")
    max_length = normalize_krea2_max_length(settings.get("max_length"))
    positive = encode_krea2_prompt(clip=clip, prompt=positive_prompt, max_length=max_length)
    if use_zero_negative_conditioning(settings):
        negative = zero_out_conditioning(positive)
    else:
        negative = encode_krea2_prompt(clip=clip, prompt=negative_prompt or "", max_length=max_length)
    loaded_vae = None
    if inpaint_config is not None:
        _phase(progress, "loading vae")
        loaded_vae = load_vae(vae=vae)
        _phase(progress, "preparing inpaint source")
        inpaint_source = inpaint_service.prepare_inpaint_source(
            config=inpaint_config,
            width=width,
            height=height,
        )
        _set_inpaint_preview(inpaint_previews, INPAINT_PREVIEW_SOURCE, inpaint_source.image)
        _set_inpaint_preview(inpaint_previews, INPAINT_PREVIEW_MASK, inpaint_source.sampling_mask)
        _phase(progress, "encoding inpaint image")
        latent = inpaint_service.encode_inpaint_source_latent(
            vae=loaded_vae,
            source=inpaint_source,
        )
    else:
        inpaint_source = None
        latent = make_empty_krea2_latent(width=width, height=height)
    inpaint_denoise = float(inpaint_config.get("denoise", 1.0)) if inpaint_config is not None else 1.0
    inpaint_steps = inpaint_service.resolve_inpaint_steps(inpaint_config, steps)
    decode_inpaint_sample = (
        inpaint_config is not None
        and _inpaint_preview_requested(inpaint_previews, INPAINT_PREVIEW_SAMPLE)
    )
    batch_seeds = incrementing_batch_seeds(seed, batch_count)
    images: list[Any] = []
    latents: list[Any] = []
    inpaint_sample_images: list[Any] = []

    for index, batch_seed in enumerate(batch_seeds):
        if inpaint_config is not None and inpaint_denoise <= 0.0:
            sampled_latent = latent
        else:
            _phase(progress, "sampling" if len(batch_seeds) == 1 else f"sampling {index + 1}/{len(batch_seeds)}")
            sampled_latent = sample_with_comfy_ksampler(
                model=model,
                seed=batch_seed,
                steps=inpaint_steps,
                cfg=cfg,
                sampler=sampler,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                denoise=inpaint_denoise,
                pid_capture_step=pid_capture_step,
                progress=progress,
            )
        image = None
        if (decode_image or decode_inpaint_sample or return_vae) and loaded_vae is None:
            _phase(progress, "loading vae")
            loaded_vae = load_vae(vae=vae)
        if decode_image or decode_inpaint_sample:
            _phase(progress, "decoding")
            decoded_image = decode_latent(vae=loaded_vae, latent=sampled_latent)
            if decode_inpaint_sample:
                inpaint_sample_images.append(decoded_image)
            image = decoded_image if decode_image else None
        if decode_image:
            if inpaint_config is not None and inpaint_source is not None:
                if inpaint_source.stitcher is not None:
                    _phase(progress, "stitching inpaint")
                    image = inpaint_service.stitch_inpaint_image(
                        stitcher=inpaint_source.stitcher,
                        inpainted_image=image,
                    )
                elif bool(inpaint_config.get("final_blend", True)):
                    _phase(progress, "blending inpaint")
                    image = inpaint_service.blend_inpaint_image(
                        source_image=inpaint_source.image,
                        generated_image=image,
                        mask=inpaint_source.mask,
                        feather=int(inpaint_config.get("mask_feather", 24)),
                    )

        def sample_second_pass(
            second_latent: dict[str, Any],
            second_width: int,
            second_height: int,
            denoise: float,
            second_steps: int,
        ):
            del second_width, second_height
            return sample_with_comfy_ksampler(
                model=model,
                seed=batch_seed,
                steps=second_steps,
                cfg=cfg,
                sampler=sampler,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=second_latent,
                denoise=denoise,
                progress=progress,
            )

        image, sampled_latent, loaded_vae = apply_second_sampler_pass(
            config=second_pass_config,
            image=image,
            latent=sampled_latent,
            vae=vae,
            loaded_vae=loaded_vae,
            sample_latent=sample_second_pass,
            dimension_multiple=second_pass_dimension_multiple,
            main_steps=steps,
            progress=progress,
        )
        images.append(image)
        latents.append(sampled_latent)

    if decode_inpaint_sample:
        _set_inpaint_preview(
            inpaint_previews,
            INPAINT_PREVIEW_SAMPLE,
            _concat_optional_tensors(inpaint_sample_images),
        )
    return GenerationResult(
        image=_concat_optional_tensors(images),
        latent=_combine_latent_batch(latents),
        positive=positive,
        negative=negative,
        vae=loaded_vae,
        model=model,
        clip=clip,
    )


def generate_flux2_klein_t2i(
    *,
    diffusion_model: str,
    text_encoder: str,
    vae: str,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    settings: dict[str, Any],
    batch_count: int = 1,
    lora_config: dict[str, Any] | None = None,
    loaded_model: Any = None,
    loaded_clip: Any = None,
    reference_inputs: Any = None,
    inpaint_config: dict[str, Any] | None = None,
    inpaint_previews: dict[str, Any] | None = None,
    decode_image: bool = True,
    return_vae: bool = False,
    second_pass_config: dict[str, Any] | None = None,
    second_pass_dimension_multiple: int | None = 16,
    pid_capture_step: int | None = None,
    progress: Any = None,
):
    using_connected_model_pair = loaded_model is not None and loaded_clip is not None
    model = loaded_model
    if model is None:
        _phase(progress, "loading diffusion model")
        model = load_diffusion_model(
            diffusion_model=diffusion_model,
            precision_policy=settings.get("precision_policy"),
        )
    clip = loaded_clip
    if clip is None:
        _phase(progress, "loading text encoder")
        clip = load_text_encoder(
            text_encoder=text_encoder,
            clip_type=text_encoder_clip_type("flux2_klein_9b"),
        )
    apply_timing = normalize_performance_apply_timing(settings)
    if apply_timing == "before_loras":
        model = _apply_model_performance_if_configured(
            model=model,
            settings=settings,
            reference_inputs=reference_inputs,
            inpaint_config=inpaint_config,
            progress=progress,
        )
    if not using_connected_model_pair:
        _phase(progress, "applying loras")
        model, clip, _ = apply_lora_config(model=model, clip=clip, lora_config=lora_config)
    if apply_timing == "after_loras":
        model = _apply_model_performance_if_configured(
            model=model,
            settings=settings,
            reference_inputs=reference_inputs,
            inpaint_config=inpaint_config,
            progress=progress,
        )
    _phase(progress, "encoding prompts")
    guidance = float(settings.get("guidance", settings.get("cfg", 1.0)))
    zero_negative_conditioning = use_zero_negative_conditioning(settings)
    positive = encode_flux2_prompt(clip=clip, prompt=positive_prompt, guidance=guidance)
    negative = None
    if not zero_negative_conditioning:
        negative = encode_flux2_prompt(clip=clip, prompt=negative_prompt or "", guidance=guidance)
    reference_images = _filter_duplicate_inpaint_reference_images(
        tuple(getattr(reference_inputs, "images", ()) or ()),
        inpaint_config=inpaint_config,
        settings=settings,
    )
    loaded_vae = None
    if reference_images or inpaint_config is not None:
        _phase(progress, "loading vae")
        loaded_vae = load_vae(vae=vae)
    flux_inpaint_source = None
    if inpaint_config is not None:
        _phase(progress, "preparing inpaint source")
        flux_inpaint_source = inpaint_service.prepare_inpaint_source(
            config=inpaint_config,
            width=width,
            height=height,
        )
        _set_inpaint_preview(inpaint_previews, INPAINT_PREVIEW_SOURCE, flux_inpaint_source.image)
        _set_inpaint_preview(inpaint_previews, INPAINT_PREVIEW_MASK, flux_inpaint_source.sampling_mask)
    if reference_images:
        _phase(progress, "encoding reference images")
        reference_latents = []
        for reference_image in reference_images:
            scaled_image = scale_image_to_total_pixels(
                image=reference_image,
                megapixels=float(settings.get("reference_megapixels", 1.0)),
                upscale_method=str(settings.get("reference_upscale_method", "area")),
                resolution_steps=int(settings.get("reference_resolution_steps", 1)),
                multiple_value=settings.get("multiple_value", "none"),
            )
            reference_latents.append(encode_image_to_latent(vae=loaded_vae, image=scaled_image))
        if (
            flux_inpaint_source is not None
            and bool(inpaint_config.get("crop_source_reference", True))
        ):
            reference_latents.append(encode_image_to_latent(vae=loaded_vae, image=flux_inpaint_source.image))
        if zero_negative_conditioning:
            positive, _ = apply_reference_latents_to_conditioning(
                positive=positive,
                negative=positive,
                reference_latents=reference_latents,
            )
        else:
            positive, negative = apply_reference_latents_to_conditioning(
                positive=positive,
                negative=negative,
                reference_latents=reference_latents,
            )
    elif (
        flux_inpaint_source is not None
        and bool(inpaint_config.get("crop_source_reference", True))
    ):
        _phase(progress, "encoding inpaint reference")
        reference_latents = [encode_image_to_latent(vae=loaded_vae, image=flux_inpaint_source.image)]
        if zero_negative_conditioning:
            positive, _ = apply_reference_latents_to_conditioning(
                positive=positive,
                negative=positive,
                reference_latents=reference_latents,
            )
        else:
            positive, negative = apply_reference_latents_to_conditioning(
                positive=positive,
                negative=negative,
                reference_latents=reference_latents,
            )
    if zero_negative_conditioning:
        negative = zero_out_conditioning(positive)
    if inpaint_config is not None:
        _phase(progress, "conditioning inpaint")
        positive, negative, latent = inpaint_service.apply_inpaint_model_conditioning(
            vae=loaded_vae,
            positive=positive,
            negative=negative,
            image=flux_inpaint_source.image,
            mask=flux_inpaint_source.sampling_mask,
        )
    else:
        latent = make_empty_flux2_latent(width=width, height=height)
    inpaint_denoise = float(inpaint_config.get("denoise", 1.0)) if inpaint_config is not None else 1.0
    inpaint_steps = inpaint_service.resolve_inpaint_steps(inpaint_config, steps)
    decode_inpaint_sample = (
        inpaint_config is not None
        and _inpaint_preview_requested(inpaint_previews, INPAINT_PREVIEW_SAMPLE)
    )
    use_sampling = not (inpaint_config is not None and inpaint_denoise <= 0.0)
    sigmas = None
    if use_sampling:
        _apply_memory_policy_before_sampling_if_configured(settings=settings, progress=progress)
        if scheduler == "auto":
            sigma_width, sigma_height = (
                flux_inpaint_source.working_dimensions(fallback_width=width, fallback_height=height)
                if flux_inpaint_source is not None
                else (width, height)
            )
            schedule_steps = (
                inpaint_service.resolve_denoise_schedule_steps(inpaint_steps, inpaint_denoise)
                if inpaint_config is not None
                else steps
            )
            sigmas = flux2_sigmas(steps=schedule_steps, width=sigma_width, height=sigma_height)
            if inpaint_config is not None:
                sigmas = inpaint_service.apply_denoise_to_sigmas(
                    sigmas,
                    inpaint_denoise,
                    steps=inpaint_steps,
                )
    batch_seeds = incrementing_batch_seeds(seed, batch_count)
    images: list[Any] = []
    latents: list[Any] = []
    inpaint_sample_images: list[Any] = []

    for index, batch_seed in enumerate(batch_seeds):
        if not use_sampling:
            sampled_latent = latent
        elif scheduler == "auto":
            _phase(progress, "sampling" if len(batch_seeds) == 1 else f"sampling {index + 1}/{len(batch_seeds)}")
            sampled_latent = sample_with_sigmas(
                model=model,
                seed=batch_seed,
                steps=inpaint_steps if inpaint_config is not None else steps,
                cfg=cfg,
                sampler=sampler,
                scheduler="normal",
                positive=positive,
                negative=negative,
                latent=latent,
                sigmas=sigmas,
                pid_capture_step=pid_capture_step,
                progress=progress,
            )
        else:
            _phase(progress, "sampling" if len(batch_seeds) == 1 else f"sampling {index + 1}/{len(batch_seeds)}")
            sampled_latent = sample_with_comfy_ksampler(
                model=model,
                seed=batch_seed,
                steps=inpaint_steps,
                cfg=cfg,
                sampler=sampler,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                denoise=inpaint_denoise,
                pid_capture_step=pid_capture_step,
                progress=progress,
            )
        image = None
        if (decode_image or decode_inpaint_sample or return_vae) and loaded_vae is None:
            _phase(progress, "loading vae")
            loaded_vae = load_vae(vae=vae)
        if decode_image or decode_inpaint_sample:
            _phase(progress, "decoding")
            decoded_image = decode_latent(vae=loaded_vae, latent=sampled_latent)
            if decode_inpaint_sample:
                inpaint_sample_images.append(decoded_image)
            image = decoded_image if decode_image else None
        if decode_image:
            if inpaint_config is not None and flux_inpaint_source is not None:
                color_match_strength = float(inpaint_config.get("color_match_strength", 0.0))
                if color_match_strength > 0.0:
                    _phase(progress, "matching inpaint color")
                    image = inpaint_service.apply_inpaint_color_match(
                        target_image=image,
                        reference_image=flux_inpaint_source.image,
                        exclude_mask=flux_inpaint_source.sampling_mask,
                        strength=color_match_strength,
                    )
                if flux_inpaint_source.stitcher is not None:
                    _phase(progress, "stitching inpaint")
                    image = inpaint_service.stitch_inpaint_image(
                        stitcher=flux_inpaint_source.stitcher,
                        inpainted_image=image,
                    )
                elif bool(inpaint_config.get("final_blend", True)):
                    _phase(progress, "blending inpaint")
                    image = inpaint_service.blend_inpaint_image(
                        source_image=flux_inpaint_source.image,
                        generated_image=image,
                        mask=flux_inpaint_source.mask,
                        feather=int(inpaint_config.get("mask_feather", 24)),
                    )

        def sample_second_pass(
            second_latent: dict[str, Any],
            second_width: int,
            second_height: int,
            denoise: float,
            second_steps: int,
        ):
            if scheduler == "auto":
                second_sigmas = flux2_sigmas(steps=second_steps, width=second_width, height=second_height)
                second_sigmas = inpaint_service.apply_denoise_to_sigmas(second_sigmas, denoise)
                return sample_with_sigmas(
                    model=model,
                    seed=batch_seed,
                    steps=second_steps,
                    cfg=cfg,
                    sampler=sampler,
                    scheduler="normal",
                    positive=positive,
                    negative=negative,
                    latent=second_latent,
                    sigmas=second_sigmas,
                    progress=progress,
                )
            return sample_with_comfy_ksampler(
                model=model,
                seed=batch_seed,
                steps=second_steps,
                cfg=cfg,
                sampler=sampler,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=second_latent,
                denoise=denoise,
                progress=progress,
            )

        image, sampled_latent, loaded_vae = apply_second_sampler_pass(
            config=second_pass_config,
            image=image,
            latent=sampled_latent,
            vae=vae,
            loaded_vae=loaded_vae,
            sample_latent=sample_second_pass,
            dimension_multiple=second_pass_dimension_multiple,
            main_steps=steps,
            progress=progress,
        )
        images.append(image)
        latents.append(sampled_latent)

    if decode_inpaint_sample:
        _set_inpaint_preview(
            inpaint_previews,
            INPAINT_PREVIEW_SAMPLE,
            _concat_optional_tensors(inpaint_sample_images),
        )
    return GenerationResult(
        image=_concat_optional_tensors(images),
        latent=_combine_latent_batch(latents),
        positive=positive,
        negative=negative,
        vae=loaded_vae,
        model=model,
        clip=clip,
    )
