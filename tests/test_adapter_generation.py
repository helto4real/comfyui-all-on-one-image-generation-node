import json

from adapters import flux2_klein_9b, z_image_turbo
from adapters.flux2_klein_9b import Flux2Klein9BAdapter
from adapters.z_image_turbo import ZImageTurboAdapter
from nodes.aio_generate import AIOImageGenerate
from services.reference_inputs import ReferenceInputs


def test_z_image_adapter_calls_real_generation_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}

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

    image, latent = adapter.generate(
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
    assert calls["steps"] == 8
    assert calls["positive_prompt"] == "prompt"
    assert calls["loaded_model"] == "patched_model"
    assert calls["loaded_clip"] == "patched_clip"


def test_flux2_adapter_calls_real_generation_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}

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

    image, latent = adapter.generate(
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
    assert calls["steps"] == 4
    assert calls["negative_prompt"] == "negative"
    assert calls["loaded_model"] == "patched_model"
    assert calls["loaded_clip"] == "patched_clip"


def test_flux2_adapter_passes_reference_inputs_to_pipeline(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return "image", {"samples": "latent"}

    monkeypatch.setattr(flux2_klein_9b.pipeline, "generate_flux2_klein_t2i", fake_generate)
    adapter = Flux2Klein9BAdapter()
    reference_inputs = ReferenceInputs(images=("first", "second"))

    image, latent = adapter.generate(
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
    assert calls["reference_inputs"] is reference_inputs


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


def test_z_image_negative_prompt_warning_is_in_run_info(monkeypatch):
    def fake_generate(**kwargs):
        return "image", {"samples": "latent"}

    monkeypatch.setattr(z_image_turbo.pipeline, "generate_z_image_turbo_t2i", fake_generate)

    image, latent, run_info = AIOImageGenerate().generate(
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
    assert parsed["width"] == 576
    assert parsed["height"] == 1024
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
