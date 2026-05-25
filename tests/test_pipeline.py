import sys
from types import SimpleNamespace

import pytest

from loaders import gguf_backend
from services import pipeline


def test_gguf_diffusion_loader_uses_comfy_node_mapping(monkeypatch):
    calls = {}

    class FakeUnetLoaderGGUF:
        def load_unet(self, unet_name):
            calls["unet_name"] = unet_name
            return ("model",)

    fake_nodes = SimpleNamespace(
        NODE_CLASS_MAPPINGS={"UnetLoaderGGUF": FakeUnetLoaderGGUF}
    )
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)

    model = pipeline.load_diffusion_model(
        diffusion_model="unet_gguf/model.gguf",
    )

    assert model == "model"
    assert calls["unet_name"] == "model.gguf"


def test_gguf_text_encoder_loader_uses_comfy_node_mapping(monkeypatch):
    calls = {}

    class FakeCLIPLoaderGGUF:
        def load_clip(self, clip_name, type):
            calls["clip_name"] = clip_name
            calls["type"] = type
            return ("clip",)

    fake_nodes = SimpleNamespace(
        NODE_CLASS_MAPPINGS={"CLIPLoaderGGUF": FakeCLIPLoaderGGUF}
    )
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)

    clip = pipeline.load_text_encoder(
        text_encoder="clip_gguf/text.gguf",
        clip_type="flux2",
    )

    assert clip == "clip"
    assert calls == {"clip_name": "text.gguf", "type": "flux2"}


def test_missing_gguf_node_mapping_raises_install_message(monkeypatch):
    fake_nodes = SimpleNamespace(NODE_CLASS_MAPPINGS={})
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)

    with pytest.raises(ValueError, match="GGUF support requires"):
        pipeline.load_diffusion_model(
            diffusion_model="model.gguf",
        )


def test_missing_gguf_node_mapping_uses_backend_message(monkeypatch):
    fake_nodes = SimpleNamespace(NODE_CLASS_MAPPINGS={})
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes)

    with pytest.raises(ValueError) as exc_info:
        pipeline.load_text_encoder(
            text_encoder="text.gguf",
            clip_type="stable_diffusion",
        )

    assert str(exc_info.value) == gguf_backend.explain_missing()


def test_pipeline_applies_loras_before_prompt_encoding(monkeypatch):
    events = []

    monkeypatch.setattr(
        pipeline,
        "load_diffusion_model",
        lambda **kwargs: events.append("load_model") or "model",
    )
    monkeypatch.setattr(
        pipeline,
        "load_text_encoder",
        lambda **kwargs: events.append("load_clip") or "clip",
    )
    monkeypatch.setattr(
        pipeline,
        "apply_lora_config",
        lambda **kwargs: events.append("apply_loras") or ("model+lora", "clip+lora", []),
    )
    monkeypatch.setattr(
        pipeline,
        "load_vae",
        lambda **kwargs: events.append("load_vae") or "vae",
    )
    monkeypatch.setattr(
        pipeline,
        "encode_z_image_prompt",
        lambda **kwargs: events.append("encode_prompt") or "conditioning",
    )
    monkeypatch.setattr(pipeline, "make_empty_z_image_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "image")

    image, latent = pipeline.generate_z_image_turbo_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=1024,
        height=1024,
        seed=0,
        steps=8,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={},
        lora_config={"lora_1": {"on": True, "lora": "style", "strength": 1}},
    )

    assert image == "image"
    assert latent == {"samples": "sampled"}
    assert events[:5] == [
        "load_model",
        "load_clip",
        "apply_loras",
        "load_vae",
        "encode_prompt",
    ]


def test_flux2_pipeline_encodes_reference_images_before_sampling(monkeypatch):
    events = []
    captured = {}

    monkeypatch.setattr(
        pipeline,
        "load_diffusion_model",
        lambda **kwargs: events.append("load_model") or "model",
    )
    monkeypatch.setattr(
        pipeline,
        "load_text_encoder",
        lambda **kwargs: events.append("load_clip") or "clip",
    )
    monkeypatch.setattr(
        pipeline,
        "apply_lora_config",
        lambda **kwargs: events.append("apply_loras") or ("model", "clip", []),
    )
    monkeypatch.setattr(
        pipeline,
        "load_vae",
        lambda **kwargs: events.append("load_vae") or "vae",
    )
    monkeypatch.setattr(
        pipeline,
        "encode_flux2_prompt",
        lambda **kwargs: events.append(f"encode:{kwargs['prompt']}") or kwargs["prompt"],
    )
    monkeypatch.setattr(
        pipeline,
        "scale_image_to_total_pixels",
        lambda **kwargs: events.append(f"scale:{kwargs['image']}") or f"scaled:{kwargs['image']}",
    )
    monkeypatch.setattr(
        pipeline,
        "encode_image_to_latent",
        lambda **kwargs: events.append(f"vae:{kwargs['image']}") or {"samples": f"latent:{kwargs['image']}"},
    )

    def fake_apply_references(**kwargs):
        events.append("apply_references")
        captured["reference_latents"] = kwargs["reference_latents"]
        return "positive+refs", "negative+refs"

    def fake_sample(**kwargs):
        events.append("sample")
        captured["positive"] = kwargs["positive"]
        captured["negative"] = kwargs["negative"]
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", fake_apply_references)
    monkeypatch.setattr(pipeline, "make_empty_flux2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline, "sample_with_sigmas", fake_sample)
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: events.append("decode") or "image",
    )

    reference_inputs = SimpleNamespace(images=("first", "second"))
    image, latent = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        seed=0,
        steps=4,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={},
        reference_inputs=reference_inputs,
    )

    assert image == "image"
    assert latent == {"samples": "sampled"}
    assert captured["reference_latents"] == [
        {"samples": "latent:scaled:first"},
        {"samples": "latent:scaled:second"},
    ]
    assert captured["positive"] == "positive+refs"
    assert captured["negative"] == "negative+refs"
    assert events.index("scale:first") < events.index("apply_references")
    assert events.index("apply_references") < events.index("sample")
