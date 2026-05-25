"""Lazy ComfyUI text-to-image pipeline helpers.

This module intentionally keeps all ComfyUI, torch, and custom-node imports inside
execution-time functions so importing this custom node pack remains lightweight.
"""

from __future__ import annotations

from typing import Any

try:
    from ..loaders import gguf_backend, safetensors_backend
    from .lora_application import apply_lora_config
    from .model_resolution import infer_model_format, strip_category_prefix
except ImportError:  # pragma: no cover - direct test imports
    from loaders import gguf_backend, safetensors_backend
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
    progress: Any = None,
):
    _phase(progress, "loading diffusion model")
    model = load_diffusion_model(
        diffusion_model=diffusion_model,
        precision_policy=settings.get("precision_policy"),
    )
    _phase(progress, "loading text encoder")
    clip = load_text_encoder(
        text_encoder=text_encoder,
        clip_type="stable_diffusion",
    )
    _phase(progress, "applying loras")
    model, clip, _ = apply_lora_config(model=model, clip=clip, lora_config=lora_config)
    _phase(progress, "loading vae")
    loaded_vae = load_vae(vae=vae)
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
    _phase(progress, "decoding")
    image = decode_latent(vae=loaded_vae, latent=sampled_latent)
    return image, sampled_latent


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
    progress: Any = None,
):
    _phase(progress, "loading diffusion model")
    model = load_diffusion_model(
        diffusion_model=diffusion_model,
        precision_policy=settings.get("precision_policy"),
    )
    _phase(progress, "loading text encoder")
    clip = load_text_encoder(
        text_encoder=text_encoder,
        clip_type="flux2",
    )
    _phase(progress, "applying loras")
    model, clip, _ = apply_lora_config(model=model, clip=clip, lora_config=lora_config)
    _phase(progress, "loading vae")
    loaded_vae = load_vae(vae=vae)
    _phase(progress, "encoding prompts")
    guidance = float(settings.get("guidance", settings.get("cfg", 1.0)))
    positive = encode_flux2_prompt(clip=clip, prompt=positive_prompt, guidance=guidance)
    negative = encode_flux2_prompt(clip=clip, prompt=negative_prompt or "", guidance=guidance)
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
    _phase(progress, "decoding")
    image = decode_latent(vae=loaded_vae, latent=sampled_latent)
    return image, sampled_latent
