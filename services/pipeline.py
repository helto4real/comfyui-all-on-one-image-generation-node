"""Lazy ComfyUI text-to-image pipeline helpers.

This module intentionally keeps all ComfyUI, torch, and custom-node imports inside
execution-time functions so importing this custom node pack remains lightweight.
"""

from __future__ import annotations

import math
from typing import Any

try:
    from ..loaders import gguf_backend, safetensors_backend
    from .dimensions import parse_multiple_value, round_to_multiple
    from . import inpaint as inpaint_service
    from . import krea2_rebalance
    from .lora_application import apply_lora_config
    from .lora_config import normalize_lora_config
    from .model_resolution import infer_model_format, strip_category_prefix
    from .performance import apply_performance_settings, normalize_performance_apply_timing, performance_settings_present
except ImportError:  # pragma: no cover - direct test imports
    from loaders import gguf_backend, safetensors_backend
    from services.dimensions import parse_multiple_value, round_to_multiple
    from services import inpaint as inpaint_service
    from services import krea2_rebalance
    from services.lora_application import apply_lora_config
    from services.lora_config import normalize_lora_config
    from services.model_resolution import infer_model_format, strip_category_prefix
    from services.performance import apply_performance_settings, normalize_performance_apply_timing, performance_settings_present


PID_CAPTURE_KEY = "pid_capture"


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


def encode_krea2_prompt(*, clip: Any, prompt: str):
    import nodes  # type: ignore

    return nodes.CLIPTextEncode().encode(clip, prompt)[0]


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
):
    if pid_capture_step is None:
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
            denoise=float(denoise),
        )[0]

    import comfy.sample  # type: ignore
    import comfy.samplers  # type: ignore
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
    preview_callback = latent_preview.prepare_callback(model, effective_steps)

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
):
    import latent_preview  # type: ignore

    effective_steps = max(0, int(sigmas.shape[-1]) - 1)
    target_step = resolve_pid_capture_step(pid_capture_step, effective_steps)
    captured: dict[str, Any] = {}
    preview_callback = latent_preview.prepare_callback(model, effective_steps)

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
):
    from comfy_extras.nodes_custom_sampler import KSamplerSelect, RandomNoise, SamplerCustomAdvanced  # type: ignore

    noise = _node_output_first(RandomNoise.execute(noise_seed=seed))
    sampler_obj = _node_output_first(KSamplerSelect.execute(sampler_name=sampler))
    sampled = _node_output_first(
        SamplerCustomAdvanced.execute(
            noise=noise,
            guider=guider,
            sampler=sampler_obj,
            sigmas=sigmas,
            latent_image=latent,
        )
    )
    target_step = resolve_pid_capture_step(pid_capture_step, max(1, int(getattr(sigmas, "shape", [1])[-1]) - 1))
    if target_step is None:
        return sampled
    return _attach_pid_capture(
        latent=sampled,
        source_latent=latent,
        captured={},
        fallback_samples=sampled["samples"],
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
    width: int,
    height: int,
    seed: int,
    steps: int,
    sampler: str,
    scheduler: str,
    settings: dict[str, Any],
    lora_config: dict[str, Any] | None = None,
    loaded_model: Any = None,
    loaded_clip: Any = None,
    inpaint_config: dict[str, Any] | None = None,
    decode_image: bool = True,
    return_vae: bool = False,
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
    negative = zero_out_conditioning(positive)
    loaded_vae = None
    if inpaint_config is not None:
        _phase(progress, "loading vae")
        loaded_vae = load_vae(vae=vae)
        _phase(progress, "preparing inpaint crop")
        inpaint_source = inpaint_service.prepare_inpaint_source(
            config=inpaint_config,
            width=width,
            height=height,
        )
        _phase(progress, "encoding inpaint image")
        latent = inpaint_service.encode_inpaint_source_latent(
            vae=loaded_vae,
            source=inpaint_source,
        )
    else:
        inpaint_source = None
        latent = make_empty_ideogram4_latent(width=width, height=height)
    _phase(progress, "preparing guider")
    guider = build_dual_model_guider(
        model=model,
        model_negative=model_negative,
        positive=positive,
        negative=negative,
        cfg=float(settings.get("dual_cfg", settings.get("cfg", 7.0))),
    )
    if settings.get("schedule_mode") == "basic":
        sigmas = basic_sigmas(model=scheduler_model, scheduler=scheduler, steps=steps)
    else:
        sigmas = ideogram4_sigmas(
            steps=steps,
            width=width,
            height=height,
            mu=float(settings.get("mu", 0.0)),
            std=float(settings.get("std", 1.75)),
        )
    if inpaint_config is not None:
        sigmas = inpaint_service.apply_denoise_to_sigmas(
            sigmas,
            float(inpaint_config.get("denoise", 1.0)),
        )
    if inpaint_config is not None and float(inpaint_config.get("denoise", 1.0)) <= 0.0:
        sampled_latent = latent
    else:
        _phase(progress, "sampling")
        sampled_latent = sample_with_custom_guider(
            guider=guider,
            seed=seed,
            sampler=sampler,
            sigmas=sigmas,
            latent=latent,
            pid_capture_step=pid_capture_step,
        )
    image = None
    if decode_image or return_vae:
        if loaded_vae is None:
            _phase(progress, "loading vae")
            loaded_vae = load_vae(vae=vae)
    if decode_image:
        _phase(progress, "decoding")
        image = decode_latent(vae=loaded_vae, latent=sampled_latent)
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
                    feather=int(inpaint_config.get("mask_feather", 16)),
                )
    return image, sampled_latent, positive, negative, loaded_vae


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
        pid_capture_step=pid_capture_step,
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


def generate_krea2_t2i(
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
        model = _apply_model_performance_if_configured(model=model, settings=settings, progress=progress)
    if not using_connected_model_pair:
        _phase(progress, "applying loras")
        model, clip, _ = apply_lora_config(model=model, clip=clip, lora_config=lora_config)
    if apply_timing == "after_loras":
        model = _apply_model_performance_if_configured(model=model, settings=settings, progress=progress)
    _phase(progress, "encoding prompts")
    positive = encode_krea2_prompt(clip=clip, prompt=positive_prompt)
    negative = zero_out_conditioning(positive)
    if settings.get("rebalance_enabled", True):
        _phase(progress, "rebalancing conditioning")
        positive = krea2_rebalance.rebalance_conditioning(
            positive,
            multiplier=float(settings.get("rebalance_multiplier", 4.0)),
            per_layer_weights=settings.get("rebalance_per_layer_weights"),
        )
    latent = make_empty_krea2_latent(width=width, height=height)
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
        pid_capture_step=pid_capture_step,
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
    inpaint_config: dict[str, Any] | None = None,
    decode_image: bool = True,
    return_vae: bool = False,
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
    zero_negative_conditioning = math.isclose(float(cfg), 1.0)
    positive = encode_flux2_prompt(clip=clip, prompt=positive_prompt, guidance=guidance)
    negative = None
    if not zero_negative_conditioning:
        negative = encode_flux2_prompt(clip=clip, prompt=negative_prompt or "", guidance=guidance)
    reference_images = tuple(getattr(reference_inputs, "images", ()) or ())
    loaded_vae = None
    if reference_images or inpaint_config is not None:
        _phase(progress, "loading vae")
        loaded_vae = load_vae(vae=vae)
    flux_inpaint_source = None
    if inpaint_config is not None:
        _phase(progress, "preparing inpaint crop")
        flux_inpaint_source = inpaint_service.prepare_inpaint_source(
            config=inpaint_config,
            width=width,
            height=height,
        )
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
    if inpaint_config is not None and inpaint_denoise <= 0.0:
        sampled_latent = latent
    elif scheduler == "auto":
        sigmas = flux2_sigmas(steps=steps, width=width, height=height)
        if inpaint_config is not None:
            sigmas = inpaint_service.apply_denoise_to_sigmas(sigmas, inpaint_denoise)
        _phase(progress, "sampling")
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
            sigmas=sigmas,
            pid_capture_step=pid_capture_step,
        )
    else:
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
            denoise=inpaint_denoise,
            pid_capture_step=pid_capture_step,
        )
    image = None
    if (decode_image or return_vae) and loaded_vae is None:
        _phase(progress, "loading vae")
        loaded_vae = load_vae(vae=vae)
    if decode_image:
        _phase(progress, "decoding")
        image = decode_latent(vae=loaded_vae, latent=sampled_latent)
        if inpaint_config is not None and flux_inpaint_source is not None:
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
                    feather=int(inpaint_config.get("mask_feather", 16)),
                )
    return image, sampled_latent, positive, negative, loaded_vae
