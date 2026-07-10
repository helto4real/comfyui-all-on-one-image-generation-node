import json
import sys
from types import SimpleNamespace

import pytest

from adapters import flux2_klein_9b, ideogram4, krea2, z_image_turbo
from adapters.flux2_klein_9b import Flux2Klein9BAdapter
from adapters.ideogram4 import Ideogram4Adapter
from adapters.krea2 import Krea2Adapter
from adapters.z_image_turbo import ZImageTurboAdapter
from nodes.aio_generate import AIOImageGenerate
from services.reference_inputs import ReferenceInputs


def test_z_image_adapter_calls_real_generation_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(z_image_turbo.pipeline, "generate_z_image_turbo_t2i", fake_generate)
    adapter = ZImageTurboAdapter()
    settings = adapter.resolve_settings(
        model_settings={"family": "z_image_turbo", "force_steps": 8},
        width=1024,
        height=1024,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    image, latent, positive, negative, loaded_vae = adapter.generate(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="ignored",
        width=1024,
        height=1024,
        seed=123,
        settings=settings,
        sampler="auto",
        scheduler="auto",
        loaded_model="patched_model",
        loaded_clip="patched_clip",
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert calls["steps"] == 8
    assert calls["positive_prompt"] == "prompt"
    assert calls["negative_prompt"] == "ignored"
    assert calls["loaded_model"] == "patched_model"
    assert calls["loaded_clip"] == "patched_clip"
    assert calls["decode_image"] is True
    assert calls["return_vae"] is False


def test_krea2_adapter_calls_real_generation_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(krea2.pipeline, "generate_krea2_t2i", fake_generate)
    adapter = Krea2Adapter()
    inpaint_config = {"image": "image", "mask": "mask", "denoise": 0.7}
    inpaint_previews = {"requested": {}}
    settings = adapter.resolve_settings(
        model_settings={"family": "krea2", "enhancer_enabled": True},
        width=1344,
        height=2048,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    image, latent, positive, negative, loaded_vae = adapter.generate(
        diffusion_model="krea/krea2_turbo_fp8.safetensors",
        text_encoder="qwen3vl_4b_fp8_scaled.safetensors",
        vae="qwen_image_vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="ignored",
        width=1344,
        height=2048,
        seed=123,
        settings=settings,
        sampler=settings["sampler"],
        scheduler=settings["scheduler"],
        loaded_model="patched_model",
        loaded_clip="patched_clip",
        inpaint_config=inpaint_config,
        inpaint_previews=inpaint_previews,
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert calls["steps"] == 8
    assert calls["cfg"] == 1.0
    assert calls["sampler"] == "er_sde"
    assert calls["scheduler"] == "simple"
    assert calls["settings"]["max_length"] == 4096
    assert calls["positive_prompt"] == "prompt"
    assert calls["negative_prompt"] == "ignored"
    assert calls["loaded_model"] == "patched_model"
    assert calls["loaded_clip"] == "patched_clip"
    assert calls["inpaint_config"] is inpaint_config
    assert calls["inpaint_previews"] is inpaint_previews
    assert calls["decode_image"] is True
    assert calls["return_vae"] is False


def test_krea2_adapter_accepts_gguf_and_delegates_generation(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(krea2.gguf_backend, "is_available", lambda: True)
    monkeypatch.setattr(krea2.pipeline, "generate_krea2_t2i", fake_generate)
    adapter = Krea2Adapter()
    settings = adapter.resolve_settings(
        model_settings={"family": "krea2"},
        width=1344,
        height=2048,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    warnings = adapter.validate_inputs(
        diffusion_model="unet_gguf/krea2_turbo_q4.gguf",
        text_encoder="clip_gguf/qwen3vl_4b_q4.gguf",
        vae="qwen_image_vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="",
        width=1344,
        height=2048,
        settings=settings,
    )

    image, latent, positive, negative, loaded_vae = adapter.generate(
        diffusion_model="unet_gguf/krea2_turbo_q4.gguf",
        text_encoder="clip_gguf/qwen3vl_4b_q4.gguf",
        vae="qwen_image_vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="ignored",
        width=1344,
        height=2048,
        seed=123,
        settings=settings,
        sampler=settings["sampler"],
        scheduler=settings["scheduler"],
    )

    assert warnings == []
    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert loaded_vae == "vae"
    assert calls["diffusion_model"] == "unet_gguf/krea2_turbo_q4.gguf"
    assert calls["text_encoder"] == "clip_gguf/qwen3vl_4b_q4.gguf"
    assert calls["vae"] == "qwen_image_vae.safetensors"
    assert calls["sampler"] == "er_sde"
    assert calls["scheduler"] == "simple"


def test_flux2_adapter_calls_real_generation_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(flux2_klein_9b.pipeline, "generate_flux2_klein_t2i", fake_generate)
    adapter = Flux2Klein9BAdapter()
    settings = adapter.resolve_settings(
        model_settings={"family": "flux2_klein_9b", "variant": "distilled"},
        width=1024,
        height=1024,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    image, latent, positive, negative, loaded_vae = adapter.generate(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        seed=123,
        settings=settings,
        sampler="auto",
        scheduler="auto",
        loaded_model="patched_model",
        loaded_clip="patched_clip",
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert calls["steps"] == 4
    assert calls["negative_prompt"] == "negative"
    assert calls["loaded_model"] == "patched_model"
    assert calls["loaded_clip"] == "patched_clip"
    assert calls["decode_image"] is True
    assert calls["return_vae"] is False


def test_flux2_adapter_passes_reference_inputs_to_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(flux2_klein_9b.pipeline, "generate_flux2_klein_t2i", fake_generate)
    adapter = Flux2Klein9BAdapter()
    reference_inputs = ReferenceInputs(images=("first", "second"))

    image, latent, positive, negative, loaded_vae = adapter.generate(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        seed=123,
        settings={"steps": 4, "cfg": 1.0},
        sampler="auto",
        scheduler="auto",
        reference_inputs=reference_inputs,
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert calls["reference_inputs"] is reference_inputs


def test_ideogram4_adapter_calls_real_generation_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(ideogram4.pipeline, "generate_ideogram4_t2i", fake_generate)
    adapter = Ideogram4Adapter()
    settings = adapter.resolve_settings(
        model_settings={
            "family": "ideogram4",
            "unconditional_model": "uncond.safetensors",
            "preset_steps": 20,
            "dual_cfg": 7.0,
            "scheduler": "ideogram4",
        },
        width=1024,
        height=1024,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    image, latent, positive, negative, loaded_vae = adapter.generate(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="ignored",
        width=1024,
        height=1024,
        seed=123,
        settings=settings,
        sampler=settings["sampler"],
        scheduler=settings["scheduler"],
        loaded_model="patched_model",
        loaded_clip="patched_clip",
    )

    assert image == "image"
    assert latent == {"samples": "latent"}
    assert positive == "positive"
    assert negative == "negative"
    assert calls["unconditional_model"] == "uncond.safetensors"
    assert calls["steps"] == 20
    assert calls["sampler"] == "euler"
    assert calls["scheduler"] == "ideogram4"
    assert calls["loaded_model"] == "patched_model"
    assert calls["loaded_clip"] == "patched_clip"
    assert calls["decode_image"] is True
    assert calls["return_vae"] is False


def test_ideogram4_adapter_passes_inpaint_config_to_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(ideogram4.pipeline, "generate_ideogram4_t2i", fake_generate)
    adapter = Ideogram4Adapter()
    inpaint_config = {"image": "image", "mask": "mask", "denoise": 1.0}
    inpaint_previews = {"requested": {}}

    adapter.generate(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="ignored",
        width=1024,
        height=1024,
        seed=123,
        settings={
            "steps": 20,
            "sampler": "euler",
            "scheduler": "ideogram4",
            "unconditional_model": "uncond.safetensors",
        },
        sampler="euler",
        scheduler="ideogram4",
        inpaint_config=inpaint_config,
        inpaint_previews=inpaint_previews,
    )

    assert calls["inpaint_config"] is inpaint_config
    assert calls["inpaint_previews"] is inpaint_previews


def test_flux2_adapter_accepts_inpaint_and_passes_config_to_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(flux2_klein_9b.pipeline, "generate_flux2_klein_t2i", fake_generate)
    monkeypatch.setattr(flux2_klein_9b.gguf_backend, "is_available", lambda: True)
    adapter = Flux2Klein9BAdapter()
    settings = {"steps": 4, "cfg": 1.0}
    inpaint_config = {"image": "image", "mask": "mask", "denoise": 0.8}
    inpaint_previews = {"requested": {}}

    warnings = adapter.validate_inputs(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        settings=settings,
        inpaint_config=inpaint_config,
    )

    adapter.generate(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        seed=123,
        settings=settings,
        sampler="auto",
        scheduler="auto",
        inpaint_config=inpaint_config,
        inpaint_previews=inpaint_previews,
    )

    assert warnings == []
    assert settings["edit_mode"] == "inpaint"
    assert calls["inpaint_config"] is inpaint_config
    assert calls["inpaint_previews"] is inpaint_previews


def test_flux2_adapter_warns_when_no_crop_inpaint_will_downscale(monkeypatch):
    monkeypatch.setattr(flux2_klein_9b.gguf_backend, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))
    adapter = Flux2Klein9BAdapter()
    settings = {"steps": 4, "cfg": 1.0}

    warnings = adapter.validate_inputs(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=2048,
        height=2048,
        settings=settings,
        inpaint_config={"image": "image", "mask": "mask"},
    )

    assert warnings == [
        "AIO Inpaint crop/stitch is unavailable; full-frame inpaint input will be "
        "downscaled from 2048x2048 to 1024x1024 to reduce sampler VRAM use."
    ]


def test_flux2_adapter_warns_when_full_image_inpaint_will_downscale_with_crop_available(monkeypatch):
    monkeypatch.setattr(flux2_klein_9b.gguf_backend, "is_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "nodes",
        SimpleNamespace(NODE_CLASS_MAPPINGS={"InpaintCropImproved": object}),
    )
    adapter = Flux2Klein9BAdapter()
    settings = {"steps": 4, "cfg": 1.0}

    warnings = adapter.validate_inputs(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=2048,
        height=2048,
        settings=settings,
        inpaint_config={"image": "image", "mask": "mask", "source_latent_mode": "full image"},
    )

    assert warnings == [
        "AIO Inpaint full-image source latent mode is selected; full-frame inpaint input will be "
        "downscaled from 2048x2048 to 1024x1024 to reduce sampler VRAM use."
    ]


def test_flux2_adapter_warns_when_reference_duplicates_inpaint_source(monkeypatch):
    monkeypatch.setattr(flux2_klein_9b.gguf_backend, "is_available", lambda: True)
    adapter = Flux2Klein9BAdapter()
    settings = {"steps": 4, "cfg": 1.0}
    image = object()

    warnings = adapter.validate_inputs(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        settings=settings,
        reference_inputs=ReferenceInputs(images=(image,)),
        inpaint_config={"image": image, "mask": "mask"},
    )

    assert warnings == [
        "connected Flux reference image duplicates the AIO Inpaint source; "
        "using the cropped inpaint reference instead."
    ]


def test_adapters_pass_decode_image_and_return_vae_to_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return None, {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(z_image_turbo.pipeline, "generate_z_image_turbo_t2i", fake_generate)

    adapter = ZImageTurboAdapter()
    adapter.generate(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="ignored",
        width=1024,
        height=1024,
        seed=123,
        settings={"steps": 8, "cfg": 1.0},
        sampler="auto",
        scheduler="auto",
        decode_image=False,
        return_vae=True,
    )

    assert calls["decode_image"] is False
    assert calls["return_vae"] is True


def test_flux2_infers_single_reference_mode_from_one_reference(monkeypatch):
    adapter = Flux2Klein9BAdapter()
    monkeypatch.setattr(flux2_klein_9b.gguf_backend, "is_available", lambda: True)
    settings = {}

    warnings = adapter.validate_inputs(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        settings=settings,
        reference_inputs=ReferenceInputs(images=("first",)),
    )

    assert warnings == []
    assert settings["edit_mode"] == "single_reference"


def test_flux2_infers_multi_reference_mode_from_multiple_references(monkeypatch):
    adapter = Flux2Klein9BAdapter()
    monkeypatch.setattr(flux2_klein_9b.gguf_backend, "is_available", lambda: True)
    settings = {"edit_mode": "text_to_image"}

    warnings = adapter.validate_inputs(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        settings=settings,
        reference_inputs=ReferenceInputs(images=("first", "second")),
    )

    assert warnings == []
    assert settings["edit_mode"] == "multi_reference"


def test_flux2_rejects_more_than_four_references(monkeypatch):
    adapter = Flux2Klein9BAdapter()
    monkeypatch.setattr(flux2_klein_9b.gguf_backend, "is_available", lambda: True)

    try:
        adapter.validate_inputs(
            diffusion_model="model.safetensors",
            text_encoder="text.safetensors",
            vae="vae.safetensors",
            positive_prompt="prompt",
            negative_prompt="negative",
            width=1024,
            height=1024,
            settings={},
            reference_inputs=ReferenceInputs(images=("1", "2", "3", "4", "5")),
        )
    except ValueError as error:
        assert str(error) == "FLUX.2 Klein supports at most four connected reference images."
    else:
        raise AssertionError("Expected ValueError.")


def test_ideogram4_validation_rejects_invalid_inputs():
    adapter = Ideogram4Adapter()
    base = {
        "diffusion_model": "model.safetensors",
        "text_encoder": "text.safetensors",
        "vae": "vae.safetensors",
        "positive_prompt": "prompt",
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "settings": {"unconditional_model": "uncond.safetensors"},
    }

    with pytest.raises(ValueError, match="positive_prompt is required"):
        adapter.validate_inputs(**{**base, "positive_prompt": ""})
    with pytest.raises(ValueError, match="reference_image was connected"):
        adapter.validate_inputs(**base, reference_inputs=ReferenceInputs(images=("first",)))
    with pytest.raises(ValueError, match="mask was connected"):
        adapter.validate_inputs(**base, reference_inputs=ReferenceInputs(images=(), mask="mask"))
    with pytest.raises(ValueError, match="multiples of 16"):
        adapter.validate_inputs(**{**base, "width": 1025})
    with pytest.raises(ValueError, match="between 256 and 2048"):
        adapter.validate_inputs(**{**base, "width": 4096})
    with pytest.raises(ValueError, match="aspect ratio must not exceed 6:1"):
        adapter.validate_inputs(**{**base, "width": 2048, "height": 256})
    with pytest.raises(ValueError, match="unconditional_model is required"):
        adapter.validate_inputs(**{**base, "settings": {"unconditional_model": ""}})

    warnings = adapter.validate_inputs(
        **{**base, "settings": {"unconditional_model": "", "run_unconditional_model": False}}
    )
    assert warnings == []


def test_ideogram4_negative_prompt_returns_warning():
    adapter = Ideogram4Adapter()

    warnings = adapter.validate_inputs(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="ignored",
        width=1024,
        height=1024,
        settings={"unconditional_model": "uncond.safetensors"},
    )

    assert warnings == [
        "Ideogram 4 profile does not use negative prompts by default; "
        "negative_prompt was ignored."
    ]

    assert adapter.validate_inputs(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="used",
        width=1024,
        height=1024,
        settings={
            "unconditional_model": "uncond.safetensors",
            "use_zero_negative_conditioning": False,
        },
    ) == []


def test_krea2_validation_rejects_unsupported_inputs_and_warns_for_negative_prompt(monkeypatch):
    adapter = Krea2Adapter()
    base = {
        "diffusion_model": "krea/krea2_turbo_fp8.safetensors",
        "text_encoder": "qwen3vl_4b_fp8_scaled.safetensors",
        "vae": "qwen_image_vae.safetensors",
        "positive_prompt": "prompt",
        "negative_prompt": "",
        "width": 1344,
        "height": 2048,
        "settings": {"family": "krea2"},
    }

    with pytest.raises(ValueError, match="positive_prompt is required"):
        adapter.validate_inputs(**{**base, "positive_prompt": ""})
    with pytest.raises(ValueError, match="reference_image was connected"):
        adapter.validate_inputs(**base, reference_inputs=ReferenceInputs(images=("first",)))
    with pytest.raises(ValueError, match="mask was connected"):
        adapter.validate_inputs(**base, reference_inputs=ReferenceInputs(images=(), mask="mask"))
    with pytest.raises(ValueError, match="max_length must be between 1 and 4096"):
        adapter.validate_inputs(**{**base, "settings": {"family": "krea2", "max_length": 4097}})
    assert adapter.validate_inputs(**base, inpaint_config={"image": "image", "mask": "mask"}) == []
    monkeypatch.setattr(krea2.gguf_backend, "is_available", lambda: False)
    with pytest.raises(ValueError, match="A GGUF model file was selected"):
        adapter.validate_inputs(**{**base, "diffusion_model": "model.gguf"})
    monkeypatch.setattr(krea2.gguf_backend, "is_available", lambda: True)
    assert adapter.validate_inputs(**{**base, "diffusion_model": "model.gguf"}) == []
    assert adapter.validate_inputs(**{**base, "text_encoder": "clip_gguf/qwen.gguf"}) == []

    warnings = adapter.validate_inputs(**{**base, "negative_prompt": "ignored"})
    assert warnings == [
        "Krea 2 profile does not use negative prompts by default; "
        "negative_prompt was ignored."
    ]
    assert adapter.validate_inputs(
        **{
            **base,
            "negative_prompt": "used",
            "settings": {"family": "krea2", "use_zero_negative_conditioning": False},
        }
    ) == []


def test_ideogram4_resolve_settings_uses_presets_and_workflow_scheduler():
    default_settings = Ideogram4Adapter().resolve_settings(
        model_settings={
            "family": "ideogram4",
            "preset_steps": 20,
            "dual_cfg": 7.0,
            "scheduler": "ideogram4",
        },
        width=1024,
        height=1024,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )
    workflow_settings = Ideogram4Adapter().resolve_settings(
        model_settings={
            "family": "ideogram4",
            "preset_steps": 28,
            "dual_cfg": 7.0,
            "scheduler": "simple",
            "schedule_mode": "basic",
        },
        width=1024,
        height=1024,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    assert default_settings["steps"] == 20
    assert default_settings["cfg"] == 7.0
    assert default_settings["sampler"] == "euler"
    assert default_settings["scheduler"] == "ideogram4"
    assert workflow_settings["steps"] == 28
    assert workflow_settings["scheduler"] == "simple"


def test_flux2_allows_exact_dimensions_when_multiple_value_is_none(monkeypatch):
    adapter = Flux2Klein9BAdapter()
    monkeypatch.setattr(flux2_klein_9b.gguf_backend, "is_available", lambda: True)

    warnings = adapter.validate_inputs(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1025,
        height=1024,
        settings={"multiple_value": "none"},
    )

    assert warnings == []


def test_z_image_validation_accepts_negative_prompt_when_zero_negative_disabled():
    adapter = ZImageTurboAdapter()

    warnings = adapter.validate_inputs(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="used",
        width=1024,
        height=1024,
        settings={"family": "z_image_turbo", "use_zero_negative_conditioning": False},
    )

    assert warnings == []


def test_z_image_negative_prompt_warning_is_in_run_info(monkeypatch):
    def fake_generate(**kwargs):
        return "image", {"samples": "latent"}, "positive", "negative", "vae"

    monkeypatch.setattr(z_image_turbo.pipeline, "generate_z_image_turbo_t2i", fake_generate)

    image, latent, run_info, model_info, _pid_info, output_width, output_height, _inpaint_info, _image_original = AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="should be ignored",
        seed=123,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
        model_settings={
            "family": "z_image_turbo",
            "force_steps": 8,
            "ignore_negative_prompt": True,
        },
        **{"size mode": "use aspect ratio", "max side": 1024, "aspect ratio": "9:16"},
    )

    parsed = json.loads(run_info)
    assert image == "image"
    assert latent == {"samples": "latent"}
    assert model_info["positive"] == "positive"
    assert model_info["negative"] == "negative"
    assert parsed["width"] == 576
    assert parsed["height"] == 1024
    assert (output_width, output_height) == (576, 1024)
    assert parsed["warnings"] == [
        "Z-Image Turbo profile does not use negative prompts by default; "
        "negative_prompt was ignored."
    ]


def test_flux2_distilled_defaults_to_four_steps():
    settings = Flux2Klein9BAdapter().resolve_settings(
        model_settings={"family": "flux2_klein_9b", "variant": "distilled"},
        width=1024,
        height=1024,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    assert settings["steps"] == 4


def test_flux2_base_defaults_to_fifty_steps():
    settings = Flux2Klein9BAdapter().resolve_settings(
        model_settings={"family": "flux2_klein_9b", "variant": "base"},
        width=1024,
        height=1024,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    assert settings["steps"] == 50


def test_flux2_explicit_steps_override_variant_default():
    settings = Flux2Klein9BAdapter().resolve_settings(
        model_settings={"family": "flux2_klein_9b", "variant": "base"},
        width=1024,
        height=1024,
        steps=17,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    assert settings["steps"] == 17


def test_krea2_resolve_settings_uses_workflow_defaults():
    settings = Krea2Adapter().resolve_settings(
        model_settings={"family": "krea2"},
        width=1344,
        height=2048,
        steps=0,
        cfg=0.0,
        sampler="auto",
        scheduler="auto",
    )

    assert settings["steps"] == 8
    assert settings["cfg"] == 1.0
    assert settings["sampler"] == "er_sde"
    assert settings["scheduler"] == "simple"
    assert settings["width"] == 1344
    assert settings["height"] == 2048
    assert settings["max_length"] == 4096
