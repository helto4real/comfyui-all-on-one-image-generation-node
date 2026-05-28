import sys
from types import ModuleType
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

    image, latent, positive, negative, loaded_vae = pipeline.generate_z_image_turbo_t2i(
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
    assert positive == "conditioning"
    assert negative == "conditioning"
    assert loaded_vae == "vae"
    assert events[:4] == [
        "load_model",
        "load_clip",
        "apply_loras",
        "encode_prompt",
    ]


def test_z_image_pipeline_uses_connected_post_lora_model_and_clip(monkeypatch):
    events = []
    captured = {}

    monkeypatch.setattr(
        pipeline,
        "load_diffusion_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model should be connected")),
    )
    monkeypatch.setattr(
        pipeline,
        "load_text_encoder",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("clip should be connected")),
    )

    def fake_apply_loras(**kwargs):
        raise AssertionError("loras should already be applied to connected model and clip")

    def fake_encode_prompt(**kwargs):
        events.append(f"encode_prompt:{kwargs['clip']}")
        return "conditioning"

    def fake_sample(**kwargs):
        captured["sample_model"] = kwargs["model"]
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "apply_lora_config", fake_apply_loras)
    monkeypatch.setattr(
        pipeline,
        "load_vae",
        lambda **kwargs: events.append("load_vae") or "vae",
    )
    monkeypatch.setattr(pipeline, "encode_z_image_prompt", fake_encode_prompt)
    monkeypatch.setattr(pipeline, "make_empty_z_image_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "image")

    image, latent, positive, negative, loaded_vae = pipeline.generate_z_image_turbo_t2i(
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
        lora_config={"loras": [{"enabled": True, "name": "style"}]},
        loaded_model="post_lora_patched_model",
        loaded_clip="post_lora_clip",
    )

    assert image == "image"
    assert latent == {"samples": "sampled"}
    assert positive == "conditioning"
    assert negative == "conditioning"
    assert loaded_vae == "vae"
    assert events[:3] == [
        "encode_prompt:post_lora_clip",
        "encode_prompt:post_lora_clip",
        "load_vae",
    ]
    assert captured["sample_model"] == "post_lora_patched_model"


def test_z_image_pipeline_skips_vae_when_image_decode_disabled(monkeypatch):
    events = []

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(
        pipeline,
        "load_vae",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("vae should not load")),
    )
    monkeypatch.setattr(
        pipeline,
        "encode_z_image_prompt",
        lambda **kwargs: events.append(f"encode:{kwargs['prompt']}") or f"conditioning:{kwargs['prompt']}",
    )
    monkeypatch.setattr(pipeline, "make_empty_z_image_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("latent should not decode")),
    )

    image, latent, positive, negative, loaded_vae = pipeline.generate_z_image_turbo_t2i(
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
        decode_image=False,
    )

    assert image is None
    assert latent == {"samples": "sampled"}
    assert positive == "conditioning:prompt"
    assert negative == "conditioning:"
    assert loaded_vae is None


def test_z_image_pipeline_returns_vae_without_decoding_when_requested(monkeypatch):
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_z_image_prompt", lambda **kwargs: f"conditioning:{kwargs['prompt']}")
    monkeypatch.setattr(pipeline, "make_empty_z_image_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("latent should not decode")),
    )

    image, latent, positive, negative, loaded_vae = pipeline.generate_z_image_turbo_t2i(
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
        decode_image=False,
        return_vae=True,
    )

    assert image is None
    assert latent == {"samples": "sampled"}
    assert positive == "conditioning:prompt"
    assert negative == "conditioning:"
    assert loaded_vae == "vae"


def test_flux2_pipeline_uses_connected_post_lora_model_and_clip(monkeypatch):
    events = []
    captured = {}

    monkeypatch.setattr(
        pipeline,
        "load_diffusion_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model should be connected")),
    )
    monkeypatch.setattr(
        pipeline,
        "load_text_encoder",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("clip should be connected")),
    )
    monkeypatch.setattr(
        pipeline,
        "apply_lora_config",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("loras should already be applied")),
    )
    monkeypatch.setattr(
        pipeline,
        "load_vae",
        lambda **kwargs: events.append("load_vae") or "vae",
    )
    monkeypatch.setattr(
        pipeline,
        "encode_flux2_prompt",
        lambda **kwargs: events.append(f"encode:{kwargs['clip']}") or "conditioning",
    )
    monkeypatch.setattr(pipeline, "make_empty_flux2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: "sigmas")
    def fake_sample_with_sigmas(**kwargs):
        captured["model"] = kwargs["model"]
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "sample_with_sigmas", fake_sample_with_sigmas)
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "zeroed")
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "image")

    image, latent, positive, negative, loaded_vae = pipeline.generate_flux2_klein_t2i(
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
        loaded_model="post_lora_patched_model",
        loaded_clip="post_lora_clip",
    )

    assert image == "image"
    assert latent == {"samples": "sampled"}
    assert positive == "conditioning"
    assert negative == "zeroed"
    assert loaded_vae == "vae"
    assert events[:2] == [
        "encode:post_lora_clip",
        "load_vae",
    ]
    assert captured["model"] == "post_lora_patched_model"


def test_flux2_pipeline_skips_vae_when_image_decode_disabled_without_references(monkeypatch):
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(
        pipeline,
        "load_vae",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("vae should not load")),
    )
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: kwargs["prompt"])
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: f"zeroed:{conditioning}")
    monkeypatch.setattr(pipeline, "make_empty_flux2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline, "sample_with_sigmas", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("latent should not decode")),
    )

    image, latent, positive, negative, loaded_vae = pipeline.generate_flux2_klein_t2i(
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
        decode_image=False,
    )

    assert image is None
    assert latent == {"samples": "sampled"}
    assert positive == "prompt"
    assert negative == "zeroed:prompt"
    assert loaded_vae is None


def test_flux2_pipeline_returns_vae_without_decoding_when_requested(monkeypatch):
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: kwargs["prompt"])
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: f"zeroed:{conditioning}")
    monkeypatch.setattr(pipeline, "make_empty_flux2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline, "sample_with_sigmas", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("latent should not decode")),
    )

    image, latent, positive, negative, loaded_vae = pipeline.generate_flux2_klein_t2i(
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
        decode_image=False,
        return_vae=True,
    )

    assert image is None
    assert latent == {"samples": "sampled"}
    assert positive == "prompt"
    assert negative == "zeroed:prompt"
    assert loaded_vae == "vae"


def test_flux2_pipeline_loads_vae_for_references_when_image_decode_disabled(monkeypatch):
    events = []

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: events.append("load_vae") or "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: kwargs["prompt"])
    monkeypatch.setattr(
        pipeline,
        "scale_image_to_total_pixels",
        lambda **kwargs: events.append(f"scale:{kwargs['image']}") or f"scaled:{kwargs['image']}",
    )
    monkeypatch.setattr(
        pipeline,
        "encode_image_to_latent",
        lambda **kwargs: events.append(f"vae:{kwargs['vae']}:{kwargs['image']}") or {"samples": "ref"},
    )
    monkeypatch.setattr(
        pipeline,
        "apply_reference_latents_to_conditioning",
        lambda **kwargs: ("positive+refs", "negative+refs"),
    )
    monkeypatch.setattr(pipeline, "make_empty_flux2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("latent should not decode")),
    )

    image, latent, positive, negative, loaded_vae = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        seed=0,
        steps=4,
        cfg=1.5,
        sampler="auto",
        scheduler="normal",
        settings={},
        reference_inputs=SimpleNamespace(images=("first",)),
        decode_image=False,
    )

    assert image is None
    assert latent == {"samples": "sampled"}
    assert positive == "positive+refs"
    assert negative == "negative+refs"
    assert loaded_vae == "vae"
    assert events == ["load_vae", "scale:first", "vae:vae:scaled:first"]


def test_pipeline_applies_loras_when_only_model_is_connected(monkeypatch):
    events = []
    captured = {}

    monkeypatch.setattr(
        pipeline,
        "load_diffusion_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model should be connected")),
    )
    monkeypatch.setattr(
        pipeline,
        "load_text_encoder",
        lambda **kwargs: events.append("load_clip") or "clip",
    )
    monkeypatch.setattr(
        pipeline,
        "apply_lora_config",
        lambda **kwargs: events.append(f"apply_loras:{kwargs['model']}:{kwargs['clip']}") or ("model+lora", "clip+lora", []),
    )
    monkeypatch.setattr(
        pipeline,
        "load_vae",
        lambda **kwargs: events.append("load_vae") or "vae",
    )
    monkeypatch.setattr(
        pipeline,
        "encode_z_image_prompt",
        lambda **kwargs: events.append(f"encode:{kwargs['clip']}") or "conditioning",
    )
    monkeypatch.setattr(pipeline, "make_empty_z_image_latent", lambda **kwargs: {"samples": "empty"})
    def fake_sample(**kwargs):
        captured["model"] = kwargs["model"]
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "image")

    pipeline.generate_z_image_turbo_t2i(
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
        lora_config={"loras": [{"enabled": True, "name": "style"}]},
        loaded_model="patched_model",
    )

    assert events[:3] == [
        "load_clip",
        "apply_loras:patched_model:clip",
        "encode:clip+lora",
    ]
    assert captured["model"] == "model+lora"


def test_flux2_pipeline_zeroes_negative_conditioning_after_references_when_cfg_one(monkeypatch):
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
        captured["reference_negative_input"] = kwargs["negative"]
        return "positive+refs", "discarded-negative+refs"

    def fake_zero_out(conditioning):
        events.append(f"zero_out:{conditioning}")
        return f"zeroed:{conditioning}"

    def fake_sample(**kwargs):
        events.append("sample")
        captured["positive"] = kwargs["positive"]
        captured["negative"] = kwargs["negative"]
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", fake_apply_references)
    monkeypatch.setattr(pipeline, "zero_out_conditioning", fake_zero_out)
    monkeypatch.setattr(pipeline, "make_empty_flux2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline, "sample_with_sigmas", fake_sample)
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: events.append("decode") or "image",
    )

    reference_inputs = SimpleNamespace(images=("first", "second"))
    image, latent, positive, negative, loaded_vae = pipeline.generate_flux2_klein_t2i(
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
    assert positive == "positive+refs"
    assert negative == "zeroed:positive+refs"
    assert captured["reference_latents"] == [
        {"samples": "latent:scaled:first"},
        {"samples": "latent:scaled:second"},
    ]
    assert captured["reference_negative_input"] == "prompt"
    assert captured["positive"] == "positive+refs"
    assert captured["negative"] == "zeroed:positive+refs"
    assert "encode:negative" not in events
    assert events.index("scale:first") < events.index("apply_references")
    assert events.index("apply_references") < events.index("zero_out:positive+refs")
    assert events.index("zero_out:positive+refs") < events.index("sample")


def test_flux2_pipeline_keeps_negative_prompt_conditioning_when_cfg_not_one(monkeypatch):
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
        captured["reference_negative_input"] = kwargs["negative"]
        return "positive+refs", "negative+refs"

    def fail_zero_out(conditioning):
        raise AssertionError("zero_out_conditioning should not be called when cfg is not 1.0")

    def fake_sample(**kwargs):
        events.append("sample")
        captured["positive"] = kwargs["positive"]
        captured["negative"] = kwargs["negative"]
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", fake_apply_references)
    monkeypatch.setattr(pipeline, "zero_out_conditioning", fail_zero_out)
    monkeypatch.setattr(pipeline, "make_empty_flux2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: events.append("decode") or "image",
    )

    reference_inputs = SimpleNamespace(images=("first",))
    image, latent, positive, negative, loaded_vae = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        seed=0,
        steps=4,
        cfg=1.5,
        sampler="auto",
        scheduler="normal",
        settings={},
        reference_inputs=reference_inputs,
    )

    assert image == "image"
    assert latent == {"samples": "sampled"}
    assert positive == "positive+refs"
    assert negative == "negative+refs"
    assert captured["reference_latents"] == [{"samples": "latent:scaled:first"}]
    assert captured["reference_negative_input"] == "negative"
    assert captured["positive"] == "positive+refs"
    assert captured["negative"] == "negative+refs"
    assert "encode:negative" in events
    assert events.index("encode:negative") < events.index("apply_references")
    assert events.index("apply_references") < events.index("sample")


def test_reference_scaling_calculates_target_dimensions(monkeypatch):
    calls = {}

    class FakeImage:
        shape = (1, 768, 512, 3)

        def movedim(self, *args):
            calls["input_movedim"] = args
            return FakeSamples()

    class FakeSamples:
        shape = (1, 3, 768, 512)

    class FakeResized:
        def movedim(self, *args):
            calls["output_movedim"] = args
            return "resized-image"

    fake_comfy = ModuleType("comfy")
    fake_utils = ModuleType("comfy.utils")

    def fake_common_upscale(samples, width, height, upscale_method, crop):
        calls["samples"] = samples
        calls["width"] = width
        calls["height"] = height
        calls["upscale_method"] = upscale_method
        calls["crop"] = crop
        return FakeResized()

    fake_utils.common_upscale = fake_common_upscale
    fake_comfy.utils = fake_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)

    image = pipeline.scale_image_to_total_pixels(
        image=FakeImage(),
        megapixels=1.0,
        upscale_method="area",
        resolution_steps=1,
        multiple_value="16",
    )

    assert image == "resized-image"
    assert calls["width"] == 832
    assert calls["height"] == 1248
    assert calls["upscale_method"] == "area"
    assert calls["crop"] == "center"


def test_reference_scaling_uses_exact_dimensions_for_none_multiple(monkeypatch):
    calls = {}

    class FakeImage:
        shape = (1, 768, 512, 3)

        def movedim(self, *args):
            return FakeSamples()

    class FakeSamples:
        shape = (1, 3, 768, 512)

    class FakeResized:
        def movedim(self, *args):
            return "resized-image"

    fake_comfy = ModuleType("comfy")
    fake_utils = ModuleType("comfy.utils")

    def fake_common_upscale(samples, width, height, upscale_method, crop):
        calls["width"] = width
        calls["height"] = height
        return FakeResized()

    fake_utils.common_upscale = fake_common_upscale
    fake_comfy.utils = fake_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)

    pipeline.scale_image_to_total_pixels(
        image=FakeImage(),
        megapixels=1.0,
        upscale_method="area",
        resolution_steps=1,
        multiple_value="none",
    )

    assert calls["width"] == 836
    assert calls["height"] == 1254


def test_pid_target_dimensions_parse_model_name_and_preserve_aspect_ratio():
    square = pipeline.resolve_pid_target_dimensions(
        pid_diffusion_model="diffusion_models/pid/pid_flux1_1024_to_4096_4step_bf16.safetensors",
        source_width=1024,
        source_height=1024,
    )
    portrait = pipeline.resolve_pid_target_dimensions(
        pid_diffusion_model="pid/pid_flux1_512_to_2048_4step_bf16.safetensors",
        source_width=768,
        source_height=1024,
    )

    assert square == {
        "input_size": 1024,
        "output_size": 4096,
        "width": 4096,
        "height": 4096,
    }
    assert portrait == {
        "input_size": 512,
        "output_size": 2048,
        "width": 1536,
        "height": 2048,
    }


def test_pid_target_dimensions_requires_size_pattern():
    with pytest.raises(ValueError, match="512_to_2048"):
        pipeline.resolve_pid_target_dimensions(
            pid_diffusion_model="pid/model_without_size.safetensors",
            source_width=1024,
            source_height=1024,
        )


def test_generate_pid_upscale_uses_prompt_latent_and_vram_purge(monkeypatch):
    events = []
    captured = {}

    monkeypatch.setattr(
        pipeline,
        "purge_vram_and_cache",
        lambda: events.append("purge"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_diffusion_model",
        lambda **kwargs: events.append(f"load_model:{kwargs['diffusion_model']}") or "pid_model",
    )
    monkeypatch.setattr(
        pipeline,
        "load_text_encoder",
        lambda **kwargs: events.append(f"load_clip:{kwargs['clip_type']}") or "pid_clip",
    )
    monkeypatch.setattr(
        pipeline,
        "encode_pid_prompt",
        lambda **kwargs: captured.update({"prompt": kwargs["prompt"]}) or "pid_positive",
    )

    def fake_apply_pid_conditioning(**kwargs):
        captured["source_latent"] = kwargs["latent"]
        captured["latent_format"] = kwargs["latent_format"]
        return "pid_conditioning"

    def fake_make_latent(**kwargs):
        captured["target_width"] = kwargs["width"]
        captured["target_height"] = kwargs["height"]
        return {"samples": "target"}

    monkeypatch.setattr(pipeline, "apply_pid_conditioning", fake_apply_pid_conditioning)
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: f"zeroed:{conditioning}")
    monkeypatch.setattr(pipeline, "make_empty_chroma_radiance_latent", fake_make_latent)
    monkeypatch.setattr(pipeline, "pid_sampler", lambda **kwargs: events.append("sampler") or "sampler")
    monkeypatch.setattr(pipeline, "pid_sigmas", lambda **kwargs: events.append("sigmas") or "sigmas")

    def fake_sample(**kwargs):
        captured["seed"] = kwargs["seed"]
        captured["negative"] = kwargs["negative"]
        return {"samples": "pid_latent"}

    monkeypatch.setattr(pipeline, "sample_pid_custom", fake_sample)
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: events.append(f"load_vae:{kwargs['vae']}") or "pid_vae")
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "pid_image")

    image, metadata = pipeline.generate_pid_upscale(
        pid_diffusion_model="pid/pid_flux1_1024_to_4096_4step_bf16.safetensors",
        pid_text_encoder="pid/gemma_2_2b_it_elm_bf16.safetensors",
        pid_vae="pixel_space",
        positive_prompt="same prompt",
        source_latent={"samples": "generated"},
        source_width=1024,
        source_height=576,
        seed=123,
        latent_format="flux",
        save_vram=True,
    )

    assert image == "pid_image"
    assert events[0] == "purge"
    assert events[-1] == "purge"
    assert "load_clip:pixeldit" in events
    assert captured["prompt"] == "same prompt"
    assert captured["source_latent"] == {"samples": "generated"}
    assert captured["latent_format"] == "flux"
    assert captured["target_width"] == 4096
    assert captured["target_height"] == 2304
    assert captured["seed"] == 123
    assert captured["negative"] == "zeroed:pid_conditioning"
    assert metadata["target_width"] == 4096
    assert metadata["target_height"] == 2304
