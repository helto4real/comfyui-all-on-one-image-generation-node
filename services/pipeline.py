"""Lazy ComfyUI text-to-image pipeline helpers.

This module intentionally keeps all ComfyUI, torch, and custom-node imports inside
execution-time functions so importing this custom node pack remains lightweight.
"""

from __future__ import annotations

import math
import re
from typing import Any

try:
    from ..loaders import gguf_backend, safetensors_backend
    from .dimensions import parse_multiple_value, round_to_multiple
    from .lora_application import apply_lora_config
    from .model_resolution import infer_model_format, strip_category_prefix
except ImportError:  # pragma: no cover - direct test imports
    from loaders import gguf_backend, safetensors_backend
    from services.dimensions import parse_multiple_value, round_to_multiple
    from services.lora_application import apply_lora_config
    from services.model_resolution import infer_model_format, strip_category_prefix


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
    if model_type == "flux2_klein_9b":
        return "flux2"
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


def zero_out_conditioning(conditioning: Any):
    import nodes  # type: ignore

    return nodes.ConditioningZeroOut().zero_out(conditioning)[0]


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
):
    import nodes  # type: ignore

    return nodes.common_ksampler(
        model,
        seed,
        steps,
        cfg,
        sampler,
        scheduler,
        positive,
        negative,
        latent,
    )[0]


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
):
    import comfy.sample  # type: ignore
    import comfy.utils  # type: ignore
    import latent_preview  # type: ignore

    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    callback = latent_preview.prepare_callback(model, steps)
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
        sigmas=sigmas,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def decode_latent(*, vae: Any, latent: dict[str, Any]):
    import nodes  # type: ignore

    return nodes.VAEDecode().decode(vae, latent)[0]


def resolve_pid_target_dimensions(
    *,
    pid_diffusion_model: str,
    source_width: int,
    source_height: int,
) -> dict[str, int]:
    _, model_name = strip_category_prefix(pid_diffusion_model)
    match = re.search(r"(?P<input>\d+)_to_(?P<output>\d+)", model_name)
    if match is None:
        raise ValueError(
            "PID diffusion model name must include an input/output size pattern "
            "like '512_to_2048' or '1024_to_4096'."
        )

    input_size = int(match.group("input"))
    output_size = int(round_to_multiple(int(match.group("output")), 16))
    if source_width <= 0 or source_height <= 0:
        raise ValueError("PID source dimensions must be positive.")

    if source_width >= source_height:
        target_width = output_size
        target_height = int(round_to_multiple(round(output_size * source_height / source_width), 16))
    else:
        target_height = output_size
        target_width = int(round_to_multiple(round(output_size * source_width / source_height), 16))

    return {
        "input_size": input_size,
        "output_size": output_size,
        "width": max(16, target_width),
        "height": max(16, target_height),
    }


def detect_pid_backbone(pid_diffusion_model: str) -> str:
    _, model_name = strip_category_prefix(pid_diffusion_model)
    normalized = model_name.lower()
    if "pid_flux2" in normalized:
        return "flux2"
    if "pid_flux1" in normalized or "pid_flux" in normalized:
        return "flux1"
    if "pid_sd3" in normalized:
        return "sd3"
    return "unknown"


def pid_source_latent_channels(source_latent: dict[str, Any]) -> int | None:
    samples = source_latent.get("samples")
    shape = getattr(samples, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1])


def validate_pid_backbone_compatibility(
    *,
    pid_diffusion_model: str,
    source_latent: dict[str, Any],
    latent_format: str,
) -> dict[str, Any]:
    backbone = detect_pid_backbone(pid_diffusion_model)
    source_channels = pid_source_latent_channels(source_latent)
    expected_channels = {
        "flux1": 16,
        "flux2": 128,
        "sd3": 16,
    }.get(backbone)

    if expected_channels is not None and source_channels is None:
        raise ValueError(
            "PID source latent must include samples with a channel dimension for "
            f"{backbone} compatibility validation."
        )
    if expected_channels is not None and source_channels != expected_channels:
        raise ValueError(
            f"Selected PID diffusion model appears to be {backbone}, which expects "
            f"{expected_channels} latent channels, but the generated latent has "
            f"{source_channels} channels. Select a PID model matching the base model."
        )
    if backbone == "sd3" and latent_format != "sd3":
        raise ValueError(
            "Selected PID diffusion model appears to be sd3, so pid_latent_format "
            "must be set to 'sd3'."
        )

    return {
        "pid_backbone": backbone,
        "source_latent_channels": source_channels,
        "expected_latent_channels": expected_channels,
        "selected_model_compatible": None if expected_channels is None else True,
        "validation": "passed" if expected_channels is not None else "unknown",
    }


def pid_backbone_warnings(*, model_type: str, pid_diffusion_model: str) -> list[str]:
    backbone = detect_pid_backbone(pid_diffusion_model)
    if model_type == "flux2_klein_9b" and backbone not in ("flux2", "unknown"):
        return [
            "PID diffusion model appears to be Flux1/SD3-compatible, but the base "
            "model is Flux2. Use a pid_flux2 checkpoint to avoid latent/checkpoint "
            "mismatches."
        ]
    if model_type.startswith("z_image") and backbone == "flux2":
        return [
            "PID diffusion model appears to be Flux2-compatible, but Z-Image uses "
            "the Flux1-compatible PID checkpoint path."
        ]
    if model_type.startswith("z_image") and backbone == "sd3":
        return [
            "PID diffusion model appears to be SD3-compatible, but Z-Image uses "
            "the Flux1-compatible PID checkpoint path."
        ]
    return []


def purge_vram_and_cache() -> None:
    import comfy.model_management  # type: ignore

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()


def encode_pid_prompt(*, clip: Any, prompt: str):
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def apply_pid_conditioning(
    *,
    positive: Any,
    latent: dict[str, Any],
    latent_format: str,
    degrade_sigma: float = 0.0,
):
    from comfy_extras.nodes_pid import PiDConditioning  # type: ignore

    return _node_output_first(
        PiDConditioning.execute(
            positive=positive,
            latent=latent,
            latent_format=latent_format,
            degrade_sigma=degrade_sigma,
        )
    )


def make_empty_chroma_radiance_latent(*, width: int, height: int):
    from comfy_extras.nodes_chroma_radiance import EmptyChromaRadianceLatentImage  # type: ignore

    return _node_output_first(
        EmptyChromaRadianceLatentImage.execute(width=width, height=height, batch_size=1)
    )


def pid_sampler(*, sampler_name: str = "lcm"):
    from comfy_extras.nodes_custom_sampler import KSamplerSelect  # type: ignore

    return _node_output_first(KSamplerSelect.execute(sampler_name=sampler_name))


def pid_sigmas(*, model: Any, scheduler: str = "simple", steps: int = 4, denoise: float = 1.0):
    from comfy_extras.nodes_custom_sampler import BasicScheduler  # type: ignore

    return _node_output_first(
        BasicScheduler.execute(model=model, scheduler=scheduler, steps=steps, denoise=denoise)
    )


def sample_pid_custom(
    *,
    model: Any,
    seed: int,
    positive: Any,
    negative: Any,
    sampler: Any,
    sigmas: Any,
    latent: dict[str, Any],
    cfg: float = 1.0,
):
    from comfy_extras.nodes_custom_sampler import SamplerCustom  # type: ignore

    return _node_output_first(
        SamplerCustom.execute(
            model=model,
            add_noise=True,
            noise_seed=seed,
            cfg=cfg,
            positive=positive,
            negative=negative,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent,
        )
    )


def generate_pid_upscale(
    *,
    pid_diffusion_model: str,
    pid_text_encoder: str,
    pid_vae: str,
    positive_prompt: str,
    source_latent: dict[str, Any],
    source_width: int,
    source_height: int,
    seed: int,
    latent_format: str = "flux",
    save_vram: bool = False,
    progress: Any = None,
):
    target = resolve_pid_target_dimensions(
        pid_diffusion_model=pid_diffusion_model,
        source_width=source_width,
        source_height=source_height,
    )
    compatibility = validate_pid_backbone_compatibility(
        pid_diffusion_model=pid_diffusion_model,
        source_latent=source_latent,
        latent_format=latent_format,
    )

    if save_vram:
        _phase(progress, "purging vram before pid")
        purge_vram_and_cache()

    try:
        _phase(progress, "loading pid diffusion model")
        model = load_diffusion_model(diffusion_model=pid_diffusion_model)
        _phase(progress, "loading pid text encoder")
        clip = load_text_encoder(text_encoder=pid_text_encoder, clip_type="pixeldit")
        _phase(progress, "encoding pid prompt")
        pid_text_positive = encode_pid_prompt(clip=clip, prompt=positive_prompt)
        _phase(progress, "conditioning pid latent")
        positive = apply_pid_conditioning(
            positive=pid_text_positive,
            latent=source_latent,
            latent_format=latent_format,
            degrade_sigma=0.0,
        )
        negative = zero_out_conditioning(pid_text_positive)
        _phase(progress, "preparing pid latent")
        target_latent = make_empty_chroma_radiance_latent(
            width=target["width"],
            height=target["height"],
        )
        sampler = pid_sampler(sampler_name="lcm")
        sigmas = pid_sigmas(model=model, scheduler="simple", steps=4, denoise=1.0)
        _phase(progress, "sampling pid")
        pid_latent = sample_pid_custom(
            model=model,
            seed=seed,
            positive=positive,
            negative=negative,
            sampler=sampler,
            sigmas=sigmas,
            latent=target_latent,
            cfg=1.0,
        )
        _phase(progress, "loading pid vae")
        vae = load_vae(vae=pid_vae)
        _phase(progress, "decoding pid")
        image = decode_latent(vae=vae, latent=pid_latent)
        return image, {
            "input_size": target["input_size"],
            "output_size": target["output_size"],
            "source_width": source_width,
            "source_height": source_height,
            "target_width": target["width"],
            "target_height": target["height"],
            **compatibility,
        }
    finally:
        if save_vram:
            _phase(progress, "purging vram after pid")
            purge_vram_and_cache()


def flux2_sigmas(*, steps: int, width: int, height: int):
    from comfy_extras import nodes_flux  # type: ignore

    seq_len = width * height / (16 * 16)
    return nodes_flux.get_schedule(steps, round(seq_len))


def generate_z_image_turbo_t2i(
    *,
    diffusion_model: str,
    text_encoder: str,
    vae: str,
    positive_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    settings: dict[str, Any],
    lora_config: dict[str, Any] | None = None,
    loaded_model: Any = None,
    loaded_clip: Any = None,
    decode_image: bool = True,
    return_vae: bool = False,
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
    if not using_connected_model_pair:
        _phase(progress, "applying loras")
        model, clip, _ = apply_lora_config(model=model, clip=clip, lora_config=lora_config)
    _phase(progress, "encoding prompts")
    positive = encode_z_image_prompt(clip=clip, prompt=positive_prompt)
    negative = encode_z_image_prompt(clip=clip, prompt="")
    latent = make_empty_z_image_latent(width=width, height=height)
    _phase(progress, "sampling")
    sampled_latent = sample_with_comfy_ksampler(
        model=model,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent=latent,
    )
    image = None
    loaded_vae = None
    if decode_image or return_vae:
        _phase(progress, "loading vae")
        loaded_vae = load_vae(vae=vae)
    if decode_image:
        _phase(progress, "decoding")
        image = decode_latent(vae=loaded_vae, latent=sampled_latent)
    return image, sampled_latent, positive, negative, loaded_vae


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
    lora_config: dict[str, Any] | None = None,
    loaded_model: Any = None,
    loaded_clip: Any = None,
    reference_inputs: Any = None,
    decode_image: bool = True,
    return_vae: bool = False,
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
    if not using_connected_model_pair:
        _phase(progress, "applying loras")
        model, clip, _ = apply_lora_config(model=model, clip=clip, lora_config=lora_config)
    _phase(progress, "encoding prompts")
    guidance = float(settings.get("guidance", settings.get("cfg", 1.0)))
    zero_negative_conditioning = math.isclose(float(cfg), 1.0)
    positive = encode_flux2_prompt(clip=clip, prompt=positive_prompt, guidance=guidance)
    negative = None
    if not zero_negative_conditioning:
        negative = encode_flux2_prompt(clip=clip, prompt=negative_prompt or "", guidance=guidance)
    reference_images = tuple(getattr(reference_inputs, "images", ()) or ())
    loaded_vae = None
    if reference_images:
        _phase(progress, "loading vae")
        loaded_vae = load_vae(vae=vae)
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
    latent = make_empty_flux2_latent(width=width, height=height)
    _phase(progress, "sampling")
    if scheduler == "auto":
        sampled_latent = sample_with_sigmas(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler="normal",
            positive=positive,
            negative=negative,
            latent=latent,
            sigmas=flux2_sigmas(steps=steps, width=width, height=height),
        )
    else:
        sampled_latent = sample_with_comfy_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent,
        )
    image = None
    if (decode_image or return_vae) and loaded_vae is None:
        _phase(progress, "loading vae")
        loaded_vae = load_vae(vae=vae)
    if decode_image:
        _phase(progress, "decoding")
        image = decode_latent(vae=loaded_vae, latent=sampled_latent)
    return image, sampled_latent, positive, negative, loaded_vae
