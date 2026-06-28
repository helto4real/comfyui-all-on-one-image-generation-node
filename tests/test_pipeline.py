import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from loaders import gguf_backend
from services import pipeline


class FakeSecondPassImage:
    shape = (1, 1024, 1024, 3)

    def movedim(self, *args):
        return FakeSecondPassSamples(args)


class FakeSecondPassSamples:
    shape = (1, 3, 1024, 1024)

    def __init__(self, args=()):
        self.args = args


class FakeSecondPassResized:
    def __init__(self, calls):
        self.calls = calls

    def movedim(self, *args):
        self.calls["resized_movedim"] = args
        return "upscaled_image"


def _install_fake_common_upscale(monkeypatch, calls):
    fake_comfy = ModuleType("comfy")
    fake_utils = ModuleType("comfy.utils")

    def fake_common_upscale(samples, width, height, upscale_method, crop):
        calls["upscale"] = {
            "samples": samples,
            "width": width,
            "height": height,
            "upscale_method": upscale_method,
            "crop": crop,
        }
        return FakeSecondPassResized(calls)

    fake_utils.common_upscale = fake_common_upscale
    fake_comfy.utils = fake_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)


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


def test_pipeline_applies_performance_after_loras_by_default(monkeypatch):
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
        "apply_model_performance",
        lambda **kwargs: events.append(f"performance:{kwargs['model']}") or f"{kwargs['model']}+perf",
    )
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_z_image_prompt", lambda **kwargs: "conditioning")
    monkeypatch.setattr(pipeline, "make_empty_z_image_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", lambda **kwargs: {"samples": "sampled"})
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
        settings={"attention_mode": "off", "performance_apply_timing": "after_loras"},
        lora_config={"loras": [{"enabled": True, "name": "style"}]},
    )

    assert events[:4] == [
        "load_model",
        "load_clip",
        "apply_loras",
        "performance:model+lora",
    ]


def test_pipeline_can_apply_performance_before_loras(monkeypatch):
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
        "apply_model_performance",
        lambda **kwargs: events.append(f"performance:{kwargs['model']}") or f"{kwargs['model']}+perf",
    )
    monkeypatch.setattr(
        pipeline,
        "apply_lora_config",
        lambda **kwargs: events.append(f"apply_loras:{kwargs['model']}") or ("model+perf+lora", "clip+lora", []),
    )
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_z_image_prompt", lambda **kwargs: "conditioning")
    monkeypatch.setattr(pipeline, "make_empty_z_image_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", lambda **kwargs: {"samples": "sampled"})
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
        settings={"attention_mode": "off", "performance_apply_timing": "before_loras"},
        lora_config={"loras": [{"enabled": True, "name": "style"}]},
    )

    assert events[:4] == [
        "load_model",
        "load_clip",
        "performance:model",
        "apply_loras:model+perf",
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


def test_krea2_pipeline_applies_enhancer_model_after_loras_and_performance(monkeypatch):
    events = []
    captured = {}

    def fake_load_model(**kwargs):
        captured["load_model"] = kwargs
        events.append("load_model")
        return "model"

    def fake_load_clip(**kwargs):
        captured["load_clip"] = kwargs
        events.append("load_clip")
        return "clip"

    def fake_apply_loras(**kwargs):
        events.append("apply_loras")
        return "model+lora", "clip+lora", [{"name": "style"}]

    def fake_performance(**kwargs):
        captured["performance_settings"] = dict(kwargs["settings"])
        events.append(f"performance:{kwargs['model']}")
        return f"{kwargs['model']}+perf"

    def fake_encode(**kwargs):
        captured["encode"] = kwargs
        events.append(f"encode:{kwargs['clip']}")
        return "positive"

    def fake_zero(conditioning):
        captured["zero"] = conditioning
        events.append("zero_negative")
        return "zeroed_negative"

    def fake_enhancer(model, *, enabled, strength):
        captured["enhancer"] = {
            "model": model,
            "enabled": enabled,
            "strength": strength,
        }
        events.append("apply_enhancer")
        return f"{model}+enhancer"

    def fake_sample(**kwargs):
        captured["sample"] = kwargs
        events.append("sample")
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "load_diffusion_model", fake_load_model)
    monkeypatch.setattr(pipeline, "load_text_encoder", fake_load_clip)
    monkeypatch.setattr(pipeline, "apply_lora_config", fake_apply_loras)
    monkeypatch.setattr(pipeline, "apply_model_performance", fake_performance)
    monkeypatch.setattr(pipeline, "encode_krea2_prompt", fake_encode)
    monkeypatch.setattr(pipeline, "zero_out_conditioning", fake_zero)
    monkeypatch.setattr(pipeline.krea2_enhancer, "apply_krea2_enhancer", fake_enhancer)
    monkeypatch.setattr(pipeline, "make_empty_krea2_latent", lambda **kwargs: events.append("latent") or {"samples": "empty"})
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: events.append("load_vae") or "vae")
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: events.append("decode") or "image")

    image, latent, positive, negative, loaded_vae = pipeline.generate_krea2_t2i(
        diffusion_model="krea/krea2_turbo_fp8.safetensors",
        text_encoder="qwen3vl_4b_fp8_scaled.safetensors",
        vae="qwen_image_vae.safetensors",
        positive_prompt="prompt",
        width=1344,
        height=2048,
        seed=0,
        steps=8,
        cfg=1.0,
        sampler="er_sde",
        scheduler="simple",
        settings={
            "precision_policy": "fp8",
            "attention_mode": "off",
            "fp16_accumulation_enabled": True,
            "enhancer_enabled": True,
            "enhancer_strength": 0.75,
        },
        lora_config={"loras": [{"enabled": True, "name": "style"}]},
    )

    assert image == "image"
    assert latent == {"samples": "sampled"}
    assert positive == "positive"
    assert negative == "zeroed_negative"
    assert loaded_vae == "vae"
    assert events == [
        "load_model",
        "load_clip",
        "apply_loras",
        "performance:model+lora",
        "apply_enhancer",
        "encode:clip+lora",
        "zero_negative",
        "latent",
        "sample",
        "load_vae",
        "decode",
    ]
    assert captured["load_model"] == {
        "diffusion_model": "krea/krea2_turbo_fp8.safetensors",
        "precision_policy": "fp8",
    }
    assert captured["load_clip"] == {
        "text_encoder": "qwen3vl_4b_fp8_scaled.safetensors",
        "clip_type": "krea2",
    }
    assert captured["performance_settings"]["fp16_accumulation_enabled"] is True
    assert captured["zero"] == "positive"
    assert captured["enhancer"] == {
        "model": "model+lora+perf",
        "enabled": True,
        "strength": 0.75,
    }
    assert captured["sample"]["model"] == "model+lora+perf+enhancer"
    assert captured["sample"]["positive"] == "positive"
    assert captured["sample"]["negative"] == "zeroed_negative"
    assert captured["sample"]["sampler"] == "er_sde"
    assert captured["sample"]["scheduler"] == "simple"


def test_krea2_pipeline_can_disable_enhancer_and_decode(monkeypatch):
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "encode_krea2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "zeroed_negative")
    monkeypatch.setattr(pipeline, "make_empty_krea2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(
        pipeline,
        "load_vae",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("vae should not load")),
    )
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("latent should not decode")),
    )

    image, latent, positive, negative, loaded_vae = pipeline.generate_krea2_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=1344,
        height=2048,
        seed=0,
        steps=8,
        cfg=1.0,
        sampler="er_sde",
        scheduler="simple",
        settings={"enhancer_enabled": False},
        decode_image=False,
    )

    assert image is None
    assert latent == {"samples": "sampled"}
    assert positive == "positive"
    assert negative == "zeroed_negative"
    assert loaded_vae is None


def test_krea2_pipeline_uses_inpaint_source_latent_and_blends_output(monkeypatch):
    calls = {}
    previews = {
        pipeline.INPAINT_PREVIEW_REQUESTED: {
            pipeline.INPAINT_PREVIEW_SOURCE: True,
            pipeline.INPAINT_PREVIEW_SAMPLE: True,
            pipeline.INPAINT_PREVIEW_MASK: True,
        },
        pipeline.INPAINT_PREVIEW_SOURCE: None,
        pipeline.INPAINT_PREVIEW_SAMPLE: None,
        pipeline.INPAINT_PREVIEW_MASK: None,
    }

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "encode_krea2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline,
        "make_empty_krea2_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("empty latent should not be used")),
    )
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="source_image",
            mask="source_mask",
            noise_mask="noise_mask",
        ),
    )

    def fake_encode_source(**kwargs):
        calls["encode_source"] = kwargs
        return {"samples": "inpaint", "noise_mask": "mask"}

    def fake_sample(**kwargs):
        calls["sample"] = kwargs
        return {"samples": "sampled", "noise_mask": kwargs["latent"]["noise_mask"]}

    def fake_blend(**kwargs):
        calls["blend"] = kwargs
        return "blended_image"

    monkeypatch.setattr(pipeline.inpaint_service, "encode_inpaint_source_latent", fake_encode_source)
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "decoded_image")
    monkeypatch.setattr(pipeline.inpaint_service, "blend_inpaint_image", fake_blend)
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "stitch_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback path should not stitch")),
    )

    image, latent, positive, negative, loaded_vae = pipeline.generate_krea2_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=512,
        height=768,
        seed=123,
        steps=8,
        cfg=1.0,
        sampler="er_sde",
        scheduler="simple",
        settings={"enhancer_enabled": False},
        inpaint_config={
            "image": "image",
            "mask": "mask",
            "denoise": 0.35,
            "steps": 5,
            "mask_feather": 12,
            "final_blend": True,
        },
        inpaint_previews=previews,
    )

    assert image == "blended_image"
    assert latent == {"samples": "sampled", "noise_mask": "mask"}
    assert positive == "positive"
    assert negative == "negative"
    assert loaded_vae == "vae"
    assert calls["encode_source"]["source"].image == "source_image"
    assert calls["encode_source"]["source"].sampling_mask == "noise_mask"
    assert calls["sample"]["latent"] == {"samples": "inpaint", "noise_mask": "mask"}
    assert calls["sample"]["denoise"] == 0.35
    assert calls["sample"]["steps"] == 5
    assert calls["blend"] == {
        "source_image": "source_image",
        "generated_image": "decoded_image",
        "mask": "source_mask",
        "feather": 12,
    }
    assert previews[pipeline.INPAINT_PREVIEW_SOURCE] == "source_image"
    assert previews[pipeline.INPAINT_PREVIEW_MASK] == "noise_mask"
    assert previews[pipeline.INPAINT_PREVIEW_SAMPLE] == "decoded_image"


def test_krea2_pipeline_stitches_crop_inpaint_output(monkeypatch):
    calls = {}

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "encode_krea2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="crop",
            mask="crop_mask",
            stitcher="stitcher",
        ),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "encode_inpaint_source_latent",
        lambda **kwargs: {"samples": "inpaint", "noise_mask": "mask"},
    )
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "decoded_image")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "blend_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("crop path should stitch, not blend")),
    )

    def fake_stitch(**kwargs):
        calls["stitch"] = kwargs
        return "stitched_image"

    monkeypatch.setattr(pipeline.inpaint_service, "stitch_inpaint_image", fake_stitch)

    image, *_ = pipeline.generate_krea2_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=512,
        height=768,
        seed=123,
        steps=8,
        cfg=1.0,
        sampler="er_sde",
        scheduler="simple",
        settings={"enhancer_enabled": False},
        inpaint_config={"image": "image", "mask": "mask", "denoise": 1.0},
    )

    assert image == "stitched_image"
    assert calls["stitch"] == {
        "stitcher": "stitcher",
        "inpainted_image": "decoded_image",
    }


def test_krea2_pipeline_skips_sampling_for_zero_denoise_inpaint(monkeypatch):
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "encode_krea2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(image="source", mask="mask"),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "encode_inpaint_source_latent",
        lambda **kwargs: {"samples": "inpaint", "noise_mask": "mask"},
    )
    monkeypatch.setattr(
        pipeline,
        "sample_with_comfy_ksampler",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("sample should not run")),
    )
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("decode should not run")),
    )

    image, latent, _, _, loaded_vae = pipeline.generate_krea2_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=512,
        height=768,
        seed=123,
        steps=8,
        cfg=1.0,
        sampler="er_sde",
        scheduler="simple",
        settings={"enhancer_enabled": False},
        inpaint_config={"image": "image", "mask": "mask", "denoise": 0.0},
        decode_image=False,
    )

    assert image is None
    assert latent == {"samples": "inpaint", "noise_mask": "mask"}
    assert loaded_vae == "vae"


def test_ideogram4_pipeline_uses_dual_model_flow_and_ideogram_sigmas(monkeypatch):
    calls = {"models": []}

    def fake_load_model(**kwargs):
        calls["models"].append(kwargs)
        return f"model:{kwargs['diffusion_model']}"

    def fake_load_clip(**kwargs):
        calls["clip"] = kwargs
        return "clip"

    def fake_aura(**kwargs):
        calls["aura"] = kwargs
        return "conditional+aura"

    def fake_loras(**kwargs):
        calls["loras"] = kwargs
        return "conditional+aura+lora", [{"name": "style"}]

    def fake_cfg_override(**kwargs):
        calls["cfg_override"] = kwargs
        return "conditional+aura+lora+cfg"

    def fake_encode(**kwargs):
        calls["encode"] = kwargs
        return "positive"

    def fake_guider(**kwargs):
        calls["guider"] = kwargs
        return "guider"

    def fake_sigmas(**kwargs):
        calls["sigmas"] = kwargs
        return "ideogram_sigmas"

    def fake_sample(**kwargs):
        calls["sample"] = kwargs
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "load_diffusion_model", fake_load_model)
    monkeypatch.setattr(pipeline, "load_text_encoder", fake_load_clip)
    monkeypatch.setattr(pipeline, "apply_model_sampling_aura", fake_aura)
    monkeypatch.setattr(pipeline, "apply_lora_config_model_only", fake_loras)
    monkeypatch.setattr(pipeline, "apply_cfg_override", fake_cfg_override)
    monkeypatch.setattr(pipeline, "encode_ideogram4_prompt", fake_encode)
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "make_empty_ideogram4_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "build_dual_model_guider", fake_guider)
    monkeypatch.setattr(pipeline, "ideogram4_sigmas", fake_sigmas)
    monkeypatch.setattr(
        pipeline,
        "basic_sigmas",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("basic scheduler should not be used")),
    )
    monkeypatch.setattr(pipeline, "sample_with_custom_guider", fake_sample)
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "image")

    image, latent, positive, negative, loaded_vae = pipeline.generate_ideogram4_t2i(
        diffusion_model="ideogram4_fp8_scaled.safetensors",
        unconditional_model="ideogram4_unconditional_fp8_scaled.safetensors",
        text_encoder="qwen3vl_8b_fp8_scaled.safetensors",
        vae="flux2-vae.safetensors",
        positive_prompt="prompt",
        width=1024,
        height=1024,
        seed=123,
        steps=20,
        sampler="euler",
        scheduler="ideogram4",
        settings={
            "precision_policy": "bf16",
            "sampling_shift": 5.0,
            "cfg_override_enabled": True,
            "cfg_override": 3.0,
            "cfg_override_start_percent": 0.7,
            "cfg_override_end_percent": 1.0,
            "dual_cfg": 7.0,
            "schedule_mode": "ideogram4",
            "mu": 0.0,
            "std": 1.75,
        },
        lora_config={"loras": [{"enabled": True, "name": "style"}]},
    )

    assert image == "image"
    assert latent == {"samples": "sampled"}
    assert positive == "positive"
    assert negative == "negative"
    assert loaded_vae == "vae"
    assert calls["models"] == [
        {"diffusion_model": "ideogram4_fp8_scaled.safetensors", "precision_policy": "bf16"},
        {"diffusion_model": "ideogram4_unconditional_fp8_scaled.safetensors", "precision_policy": "bf16"},
    ]
    assert calls["clip"] == {
        "text_encoder": "qwen3vl_8b_fp8_scaled.safetensors",
        "clip_type": "ideogram4",
    }
    assert calls["aura"] == {"model": "model:ideogram4_fp8_scaled.safetensors", "shift": 5.0}
    assert calls["loras"]["model"] == "conditional+aura"
    assert calls["cfg_override"] == {
        "model": "conditional+aura+lora",
        "cfg": 3.0,
        "start_percent": 0.7,
        "end_percent": 1.0,
    }
    assert calls["guider"] == {
        "model": "conditional+aura+lora+cfg",
        "model_negative": "model:ideogram4_unconditional_fp8_scaled.safetensors",
        "positive": "positive",
        "negative": "negative",
        "cfg": 7.0,
    }
    assert calls["sigmas"] == {
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "mu": 0.0,
        "std": 1.75,
    }
    assert calls["sample"]["guider"] == "guider"
    assert calls["sample"]["sampler"] == "euler"
    assert calls["sample"]["sigmas"] == "ideogram_sigmas"


def test_ideogram4_pipeline_uses_crop_inpaint_latent_and_stitches_output(monkeypatch):
    calls = {}
    previews = {
        pipeline.INPAINT_PREVIEW_REQUESTED: {
            pipeline.INPAINT_PREVIEW_SOURCE: True,
            pipeline.INPAINT_PREVIEW_SAMPLE: True,
            pipeline.INPAINT_PREVIEW_MASK: True,
        },
        pipeline.INPAINT_PREVIEW_SOURCE: None,
        pipeline.INPAINT_PREVIEW_SAMPLE: None,
        pipeline.INPAINT_PREVIEW_MASK: None,
    }

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: f"model:{kwargs['diffusion_model']}")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_model_sampling_aura", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "apply_lora_config_model_only", lambda **kwargs: (kwargs["model"], []))
    monkeypatch.setattr(pipeline, "apply_cfg_override", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "encode_ideogram4_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline,
        "make_empty_ideogram4_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("empty latent should not be used")),
    )
    monkeypatch.setattr(pipeline, "build_dual_model_guider", lambda **kwargs: "guider")
    def fake_sigmas(**kwargs):
        calls["sigmas"] = kwargs
        return "sigmas"

    monkeypatch.setattr(pipeline, "ideogram4_sigmas", fake_sigmas)
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="cropped_image",
            mask="cropped_mask",
            noise_mask="cropped_noise_mask",
            stitcher="stitcher",
            used_crop=True,
        ),
    )

    def fake_encode_source(**kwargs):
        calls["encode_source"] = kwargs
        return {"samples": "inpaint", "noise_mask": "mask"}

    monkeypatch.setattr(pipeline.inpaint_service, "encode_inpaint_source_latent", fake_encode_source)
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_model_conditioning",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Ideogram should not use Flux inpaint conditioning")),
    )

    def fake_denoise(sigmas, denoise, *, steps=None):
        calls["denoise"] = {"sigmas": sigmas, "denoise": denoise, "steps": steps}
        return "trimmed_sigmas"

    def fake_sample(**kwargs):
        calls["sample"] = kwargs
        return {"samples": "sampled", "noise_mask": kwargs["latent"]["noise_mask"]}

    def fake_stitch(**kwargs):
        calls["stitch"] = kwargs
        return "stitched_image"

    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", fake_denoise)
    monkeypatch.setattr(pipeline, "sample_with_custom_guider", fake_sample)
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "decoded_image")
    monkeypatch.setattr(pipeline.inpaint_service, "stitch_inpaint_image", fake_stitch)
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "blend_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("crop path should stitch, not blend")),
    )

    image, latent, positive, negative, loaded_vae = pipeline.generate_ideogram4_t2i(
        diffusion_model="conditional.safetensors",
        unconditional_model="unconditional.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=512,
        height=768,
        seed=123,
        steps=20,
        sampler="euler",
        scheduler="ideogram4",
        settings={
            "sampling_shift": 5.0,
            "cfg_override_enabled": True,
            "dual_cfg": 7.0,
            "schedule_mode": "ideogram4",
        },
        inpaint_config={
            "image": "image",
            "mask": "mask",
            "denoise": 0.75,
            "steps": 8,
            "final_blend": False,
            "mask_feather": 24,
        },
        inpaint_previews=previews,
    )

    assert image == "stitched_image"
    assert latent == {"samples": "sampled", "noise_mask": "mask"}
    assert positive == "positive"
    assert negative == "negative"
    assert loaded_vae == "vae"
    assert calls["encode_source"]["source"].image == "cropped_image"
    assert calls["encode_source"]["source"].sampling_mask == "cropped_noise_mask"
    assert calls["sigmas"]["steps"] == 10
    assert calls["denoise"] == {"sigmas": "sigmas", "denoise": 0.75, "steps": 8}
    assert calls["sample"]["latent"] == {"samples": "inpaint", "noise_mask": "mask"}
    assert calls["sample"]["sigmas"] == "trimmed_sigmas"
    assert calls["stitch"] == {
        "stitcher": "stitcher",
        "inpainted_image": "decoded_image",
    }
    assert previews[pipeline.INPAINT_PREVIEW_SOURCE] == "cropped_image"
    assert previews[pipeline.INPAINT_PREVIEW_SAMPLE] == "decoded_image"
    assert previews[pipeline.INPAINT_PREVIEW_MASK] == "cropped_noise_mask"


def test_ideogram4_pipeline_skips_final_blend_when_disabled(monkeypatch):
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_model_sampling_aura", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "apply_lora_config_model_only", lambda **kwargs: (kwargs["model"], []))
    monkeypatch.setattr(pipeline, "apply_cfg_override", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "encode_ideogram4_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "build_dual_model_guider", lambda **kwargs: "guider")
    monkeypatch.setattr(pipeline, "ideogram4_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="source_image",
            mask="prepared_mask",
            noise_mask="noise_mask",
        ),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "encode_inpaint_source_latent",
        lambda **kwargs: {"samples": "inpaint", "noise_mask": "mask"},
    )
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", lambda sigmas, denoise, **kwargs: sigmas)
    monkeypatch.setattr(pipeline, "sample_with_custom_guider", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "decoded_image")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "blend_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("blend should not run")),
    )

    image, *_ = pipeline.generate_ideogram4_t2i(
        diffusion_model="conditional.safetensors",
        unconditional_model="unconditional.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=512,
        height=768,
        seed=123,
        steps=20,
        sampler="euler",
        scheduler="ideogram4",
        settings={"sampling_shift": 5.0, "cfg_override_enabled": True, "dual_cfg": 7.0},
        inpaint_config={
            "image": "image",
            "mask": "mask",
            "denoise": 1.0,
            "final_blend": False,
        },
    )

    assert image == "decoded_image"


def test_ideogram4_pipeline_fallback_final_blend_when_enabled(monkeypatch):
    calls = {}

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_model_sampling_aura", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "apply_lora_config_model_only", lambda **kwargs: (kwargs["model"], []))
    monkeypatch.setattr(pipeline, "apply_cfg_override", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "encode_ideogram4_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "build_dual_model_guider", lambda **kwargs: "guider")
    monkeypatch.setattr(pipeline, "ideogram4_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="source_image",
            mask="prepared_mask",
            noise_mask="noise_mask",
        ),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "encode_inpaint_source_latent",
        lambda **kwargs: {"samples": "inpaint", "noise_mask": "mask"},
    )
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", lambda sigmas, denoise, **kwargs: sigmas)
    monkeypatch.setattr(pipeline, "sample_with_custom_guider", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "decoded_image")

    def fake_blend(**kwargs):
        calls["blend"] = kwargs
        return "blended_image"

    monkeypatch.setattr(pipeline.inpaint_service, "blend_inpaint_image", fake_blend)
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "stitch_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback path should not stitch")),
    )

    image, *_ = pipeline.generate_ideogram4_t2i(
        diffusion_model="conditional.safetensors",
        unconditional_model="unconditional.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=512,
        height=768,
        seed=123,
        steps=20,
        sampler="euler",
        scheduler="ideogram4",
        settings={"sampling_shift": 5.0, "cfg_override_enabled": True, "dual_cfg": 7.0},
        inpaint_config={
            "image": "image",
            "mask": "mask",
            "denoise": 1.0,
            "final_blend": True,
            "mask_feather": 24,
        },
    )

    assert image == "blended_image"
    assert calls["blend"] == {
        "source_image": "source_image",
        "generated_image": "decoded_image",
        "mask": "prepared_mask",
        "feather": 24,
    }


def test_ideogram4_pipeline_skips_decode_and_blend_for_latent_only_inpaint(monkeypatch):
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_model_sampling_aura", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "apply_lora_config_model_only", lambda **kwargs: (kwargs["model"], []))
    monkeypatch.setattr(pipeline, "apply_cfg_override", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "encode_ideogram4_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "build_dual_model_guider", lambda **kwargs: "guider")
    monkeypatch.setattr(pipeline, "ideogram4_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="crop",
            mask="mask",
            noise_mask="noise_mask",
            stitcher="stitcher",
        ),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "encode_inpaint_source_latent",
        lambda **kwargs: {"samples": "inpaint", "noise_mask": "mask"},
    )
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", lambda sigmas, denoise, **kwargs: sigmas)
    monkeypatch.setattr(pipeline, "sample_with_custom_guider", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("decode should not run")),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "blend_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("blend should not run")),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "stitch_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("stitch should not run")),
    )

    image, latent, _, _, loaded_vae = pipeline.generate_ideogram4_t2i(
        diffusion_model="conditional.safetensors",
        unconditional_model="unconditional.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=512,
        height=768,
        seed=123,
        steps=20,
        sampler="euler",
        scheduler="ideogram4",
        settings={"sampling_shift": 5.0, "cfg_override_enabled": True, "dual_cfg": 7.0},
        inpaint_config={"image": "image", "mask": "mask", "denoise": 1.0},
        decode_image=False,
    )

    assert image is None
    assert latent == {"samples": "sampled"}
    assert loaded_vae == "vae"


def test_ideogram4_pipeline_skips_sampling_when_inpaint_denoise_is_zero(monkeypatch):
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_model_sampling_aura", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "apply_lora_config_model_only", lambda **kwargs: (kwargs["model"], []))
    monkeypatch.setattr(pipeline, "apply_cfg_override", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "encode_ideogram4_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "build_dual_model_guider", lambda **kwargs: "guider")
    monkeypatch.setattr(pipeline, "ideogram4_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="crop",
            mask="mask",
            noise_mask="noise_mask",
            stitcher="stitcher",
        ),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "encode_inpaint_source_latent",
        lambda **kwargs: {"samples": "inpaint", "noise_mask": "mask"},
    )
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", lambda sigmas, denoise, **kwargs: sigmas)
    monkeypatch.setattr(
        pipeline,
        "sample_with_custom_guider",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("sampling should not run")),
    )
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "decoded_image")
    monkeypatch.setattr(pipeline.inpaint_service, "stitch_inpaint_image", lambda **kwargs: "stitched_image")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "blend_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("crop path should stitch, not blend")),
    )

    image, latent, _, _, loaded_vae = pipeline.generate_ideogram4_t2i(
        diffusion_model="conditional.safetensors",
        unconditional_model="unconditional.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=512,
        height=768,
        seed=123,
        steps=20,
        sampler="euler",
        scheduler="ideogram4",
        settings={"sampling_shift": 5.0, "cfg_override_enabled": True, "dual_cfg": 7.0},
        inpaint_config={"image": "image", "mask": "mask", "denoise": 0.0},
    )

    assert image == "stitched_image"
    assert latent == {"samples": "inpaint", "noise_mask": "mask"}
    assert loaded_vae == "vae"


def test_ideogram4_pipeline_can_skip_unconditional_model(monkeypatch):
    calls = {"models": [], "performance": []}

    def fake_load_model(**kwargs):
        calls["models"].append(kwargs)
        return f"model:{kwargs['diffusion_model']}"

    def fake_performance(**kwargs):
        calls["performance"].append(kwargs["model"])
        return f"{kwargs['model']}+perf"

    def fake_guider(**kwargs):
        calls["guider"] = kwargs
        return "guider"

    monkeypatch.setattr(pipeline, "load_diffusion_model", fake_load_model)
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_model_sampling_aura", lambda **kwargs: f"{kwargs['model']}+aura")
    monkeypatch.setattr(pipeline, "apply_model_performance", fake_performance)
    monkeypatch.setattr(
        pipeline,
        "apply_lora_config_model_only",
        lambda **kwargs: (f"{kwargs['model']}+lora", []),
    )
    monkeypatch.setattr(pipeline, "apply_cfg_override", lambda **kwargs: f"{kwargs['model']}+cfg")
    monkeypatch.setattr(pipeline, "encode_ideogram4_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "make_empty_ideogram4_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "build_dual_model_guider", fake_guider)
    monkeypatch.setattr(pipeline, "ideogram4_sigmas", lambda **kwargs: "ideogram_sigmas")
    monkeypatch.setattr(pipeline, "sample_with_custom_guider", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "image")

    pipeline.generate_ideogram4_t2i(
        diffusion_model="ideogram4_fp8_scaled.safetensors",
        unconditional_model="",
        text_encoder="qwen3vl_8b_fp8_scaled.safetensors",
        vae="flux2-vae.safetensors",
        positive_prompt="prompt",
        width=1024,
        height=1024,
        seed=123,
        steps=20,
        sampler="euler",
        scheduler="ideogram4",
        settings={
            "run_unconditional_model": False,
            "sampling_shift": 5.0,
            "cfg_override_enabled": True,
            "dual_cfg": 7.0,
            "schedule_mode": "ideogram4",
            "performance_apply_timing": "before_loras",
        },
    )

    assert calls["models"] == [
        {"diffusion_model": "ideogram4_fp8_scaled.safetensors", "precision_policy": None},
    ]
    assert calls["performance"] == ["model:ideogram4_fp8_scaled.safetensors+aura"]
    assert calls["guider"]["model_negative"] is None
    assert calls["guider"]["model"] == "model:ideogram4_fp8_scaled.safetensors+aura+perf+lora+cfg"


def test_ideogram4_pipeline_workflow_preset_uses_basic_scheduler(monkeypatch):
    calls = {}

    def fake_basic_sigmas(**kwargs):
        calls["basic_sigmas"] = kwargs
        return "basic_sigmas"

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: f"model:{kwargs['diffusion_model']}")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_model_sampling_aura", lambda **kwargs: "conditional+aura")
    monkeypatch.setattr(
        pipeline,
        "apply_lora_config_model_only",
        lambda **kwargs: (kwargs["model"], []),
    )
    monkeypatch.setattr(pipeline, "apply_cfg_override", lambda **kwargs: kwargs["model"])
    monkeypatch.setattr(pipeline, "encode_ideogram4_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "make_empty_ideogram4_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "build_dual_model_guider", lambda **kwargs: "guider")
    monkeypatch.setattr(
        pipeline,
        "ideogram4_sigmas",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("ideogram sigmas should not be used")),
    )
    monkeypatch.setattr(pipeline, "basic_sigmas", fake_basic_sigmas)
    monkeypatch.setattr(pipeline, "sample_with_custom_guider", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "image")

    pipeline.generate_ideogram4_t2i(
        diffusion_model="conditional.safetensors",
        unconditional_model="unconditional.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=1024,
        height=1024,
        seed=123,
        steps=28,
        sampler="euler",
        scheduler="simple",
        settings={
            "sampling_shift": 5.0,
            "cfg_override_enabled": True,
            "dual_cfg": 7.0,
            "schedule_mode": "basic",
        },
    )

    assert calls["basic_sigmas"] == {
        "model": "conditional+aura",
        "scheduler": "simple",
        "steps": 28,
    }


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


def test_flux2_pipeline_uses_inpaint_latent_and_final_blend(monkeypatch):
    calls = {}
    previews = {
        pipeline.INPAINT_PREVIEW_REQUESTED: {
            pipeline.INPAINT_PREVIEW_SOURCE: True,
            pipeline.INPAINT_PREVIEW_SAMPLE: True,
            pipeline.INPAINT_PREVIEW_MASK: True,
        },
        pipeline.INPAINT_PREVIEW_SOURCE: None,
        pipeline.INPAINT_PREVIEW_SAMPLE: None,
        pipeline.INPAINT_PREVIEW_MASK: None,
    }

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline,
        "make_empty_flux2_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("empty latent should not be used")),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="cropped_image",
            mask="cropped_mask",
            noise_mask="cropped_noise_mask",
            stitcher="stitcher",
            used_crop=True,
            width=1024,
            height=1024,
        ),
    )
    monkeypatch.setattr(pipeline, "encode_image_to_latent", lambda **kwargs: {"samples": "source_ref"})

    def fake_apply_references(**kwargs):
        calls["reference_latents"] = kwargs["reference_latents"]
        return "positive_with_ref", "unused"

    def fake_inpaint_conditioning(**kwargs):
        calls["conditioning"] = kwargs
        return "inpaint_positive", "inpaint_negative", {"samples": "inpaint", "noise_mask": "mask"}

    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", fake_apply_references)
    monkeypatch.setattr(pipeline.inpaint_service, "apply_inpaint_model_conditioning", fake_inpaint_conditioning)
    def fake_flux2_sigmas(**kwargs):
        calls["sigmas"] = kwargs
        return "sigmas"

    monkeypatch.setattr(pipeline, "flux2_sigmas", fake_flux2_sigmas)

    def fake_denoise(sigmas, denoise, *, steps=None):
        calls["denoise"] = {"sigmas": sigmas, "denoise": denoise, "steps": steps}
        return "trimmed_sigmas"

    def fake_sample(**kwargs):
        calls["sample"] = kwargs
        return {"samples": "sampled", "noise_mask": kwargs["latent"]["noise_mask"]}

    def fake_stitch(**kwargs):
        calls["stitch"] = kwargs
        return "stitched_image"

    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", fake_denoise)
    monkeypatch.setattr(pipeline, "sample_with_sigmas", fake_sample)
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "decoded_image")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_color_match",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("color match should be opt-in")),
    )
    monkeypatch.setattr(pipeline.inpaint_service, "stitch_inpaint_image", fake_stitch)
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "blend_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("crop path should stitch, not blend")),
    )

    image, latent, positive, negative, loaded_vae = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=512,
        height=768,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={},
        inpaint_config={
            "image": "image",
            "mask": "mask",
            "denoise": 0.75,
            "steps": 8,
            "final_blend": True,
            "mask_feather": 24,
        },
        inpaint_previews=previews,
    )

    assert image == "stitched_image"
    assert latent == {"samples": "sampled", "noise_mask": "mask"}
    assert positive == "inpaint_positive"
    assert negative == "inpaint_negative"
    assert loaded_vae == "vae"
    assert calls["reference_latents"] == [{"samples": "source_ref"}]
    assert calls["conditioning"]["image"] == "cropped_image"
    assert calls["conditioning"]["mask"] == "cropped_noise_mask"
    assert calls["sigmas"] == {"steps": 10, "width": 1024, "height": 1024}
    assert calls["denoise"] == {"sigmas": "sigmas", "denoise": 0.75, "steps": 8}
    assert calls["sample"]["latent"] == {"samples": "inpaint", "noise_mask": "mask"}
    assert calls["sample"]["steps"] == 8
    assert calls["sample"]["sigmas"] == "trimmed_sigmas"
    assert calls["stitch"] == {
        "stitcher": "stitcher",
        "inpainted_image": "decoded_image",
    }
    assert previews[pipeline.INPAINT_PREVIEW_SOURCE] == "cropped_image"
    assert previews[pipeline.INPAINT_PREVIEW_SAMPLE] == "decoded_image"
    assert previews[pipeline.INPAINT_PREVIEW_MASK] == "cropped_noise_mask"


def test_flux2_pipeline_applies_color_match_before_stitch(monkeypatch):
    calls = {}
    events = []

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="cropped_image",
            mask="cropped_mask",
            noise_mask="cropped_noise_mask",
            stitcher="stitcher",
            width=1024,
            height=1024,
        ),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_model_conditioning",
        lambda **kwargs: ("inpaint_positive", "inpaint_negative", {"samples": "inpaint", "noise_mask": "mask"}),
    )
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", lambda sigmas, denoise, **kwargs: sigmas)
    monkeypatch.setattr(pipeline, "sample_with_sigmas", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: events.append("decode") or "decoded_image")

    def fake_color_match(**kwargs):
        events.append("color_match")
        calls["color_match"] = kwargs
        return "matched_image"

    def fake_stitch(**kwargs):
        events.append("stitch")
        calls["stitch"] = kwargs
        return "stitched_image"

    monkeypatch.setattr(pipeline.inpaint_service, "apply_inpaint_color_match", fake_color_match)
    monkeypatch.setattr(pipeline.inpaint_service, "stitch_inpaint_image", fake_stitch)

    image, *_ = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=512,
        height=768,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={},
        inpaint_config={
            "image": "image",
            "mask": "mask",
            "denoise": 0.75,
            "color_match_strength": 0.25,
            "crop_source_reference": False,
        },
    )

    assert image == "stitched_image"
    assert calls["color_match"] == {
        "target_image": "decoded_image",
        "reference_image": "cropped_image",
        "exclude_mask": "cropped_noise_mask",
        "strength": 0.25,
    }
    assert calls["stitch"] == {
        "stitcher": "stitcher",
        "inpainted_image": "matched_image",
    }
    assert events == ["decode", "color_match", "stitch"]


def test_flux2_pipeline_color_match_missing_node_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="source_image",
            mask="source_mask",
            noise_mask="sampling_mask",
        ),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_model_conditioning",
        lambda **kwargs: ("inpaint_positive", "inpaint_negative", {"samples": "inpaint", "noise_mask": "mask"}),
    )
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", lambda sigmas, denoise, **kwargs: sigmas)
    monkeypatch.setattr(pipeline, "sample_with_sigmas", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "decoded_image")

    with pytest.raises(ValueError, match="INPAINT_ColorMatch"):
        pipeline.generate_flux2_klein_t2i(
            diffusion_model="model.safetensors",
            text_encoder="text.safetensors",
            vae="vae.safetensors",
            positive_prompt="prompt",
            negative_prompt="negative",
            width=512,
            height=768,
            seed=123,
            steps=4,
            cfg=1.0,
            sampler="auto",
            scheduler="auto",
            settings={},
            inpaint_config={
                "image": "image",
                "mask": "mask",
                "denoise": 0.75,
                "color_match_strength": 0.25,
                "crop_source_reference": False,
            },
        )


def test_flux2_pipeline_skips_duplicate_inpaint_source_reference(monkeypatch):
    source_image = object()
    encoded_images = []
    settings = {}

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(image="crop", mask="mask", stitcher="stitcher"),
    )
    monkeypatch.setattr(
        pipeline,
        "scale_image_to_total_pixels",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("duplicate full source should not scale")),
    )

    def fake_encode_image_to_latent(**kwargs):
        encoded_images.append(kwargs["image"])
        return {"samples": f"ref:{kwargs['image']}"}

    monkeypatch.setattr(pipeline, "encode_image_to_latent", fake_encode_image_to_latent)
    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", lambda **kwargs: ("positive+ref", "unused"))
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_model_conditioning",
        lambda **kwargs: ("inpaint_positive", "inpaint_negative", {"samples": "inpaint", "noise_mask": "mask"}),
    )
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("decode should not run")),
    )

    _, latent, _, _, _ = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=512,
        height=768,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        settings=settings,
        reference_inputs=SimpleNamespace(images=(source_image,)),
        inpaint_config={"image": source_image, "mask": "mask", "denoise": 1.0},
        decode_image=False,
    )

    assert latent == {"samples": "sampled"}
    assert encoded_images == ["crop"]
    assert settings["duplicate_inpaint_reference_skipped"] is True
    assert settings["duplicate_inpaint_reference_count"] == 1


def test_flux2_pipeline_preserves_distinct_references_with_inpaint(monkeypatch):
    source_image = object()
    reference_image = object()
    events = []
    settings = {
        "reference_megapixels": 2.0,
        "reference_upscale_method": "lanczos",
        "reference_resolution_steps": 4,
        "multiple_value": "16",
    }

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(
            image="crop",
            mask="mask",
            stitcher="stitcher",
            width=1024,
            height=1024,
        ),
    )

    def fake_scale(**kwargs):
        events.append(
            (
                "scale",
                kwargs["image"],
                kwargs["megapixels"],
                kwargs["upscale_method"],
                kwargs["resolution_steps"],
                kwargs["multiple_value"],
            )
        )
        return "scaled_reference"

    def fake_encode_image_to_latent(**kwargs):
        events.append(("encode", kwargs["image"]))
        return {"samples": f"ref:{kwargs['image']}"}

    monkeypatch.setattr(pipeline, "scale_image_to_total_pixels", fake_scale)
    monkeypatch.setattr(pipeline, "encode_image_to_latent", fake_encode_image_to_latent)
    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", lambda **kwargs: ("positive+ref", "unused"))
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_model_conditioning",
        lambda **kwargs: ("inpaint_positive", "inpaint_negative", {"samples": "inpaint", "noise_mask": "mask"}),
    )
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("decode should not run")),
    )

    _, latent, _, _, _ = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=512,
        height=768,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        settings=settings,
        reference_inputs=SimpleNamespace(images=(reference_image,)),
        inpaint_config={"image": source_image, "mask": "mask", "denoise": 1.0},
        decode_image=False,
    )

    assert latent == {"samples": "sampled"}
    assert events == [
        ("scale", reference_image, 2.0, "lanczos", 4, "16"),
        ("encode", "scaled_reference"),
        ("encode", "crop"),
    ]
    assert "duplicate_inpaint_reference_skipped" not in settings
    assert settings["reference_megapixels"] == 2.0


def test_flux2_pipeline_applies_memory_policy_before_sampling(monkeypatch):
    events = []
    settings = {"memory_policy": "balanced"}

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "make_empty_flux2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "apply_memory_policy_before_sampling", lambda value: events.append(("memory", value)))

    def fake_sample(**kwargs):
        events.append(("sample", kwargs["latent"]))
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("decode should not run")),
    )

    _, latent, _, _, _ = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=512,
        height=768,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        settings=settings,
        decode_image=False,
    )

    assert latent == {"samples": "sampled"}
    assert events == [("memory", settings), ("sample", {"samples": "empty"})]


def test_flux2_pipeline_skips_decode_and_blend_for_latent_only_inpaint(monkeypatch):
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(image="crop", mask="mask", stitcher="stitcher"),
    )
    monkeypatch.setattr(pipeline, "encode_image_to_latent", lambda **kwargs: {"samples": "source_ref"})
    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", lambda **kwargs: ("positive+ref", "unused"))
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_model_conditioning",
        lambda **kwargs: ("inpaint_positive", "inpaint_negative", {"samples": "inpaint", "noise_mask": "mask"}),
    )
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", lambda sigmas, denoise, **kwargs: sigmas)
    monkeypatch.setattr(pipeline, "sample_with_sigmas", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("decode should not run")),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "blend_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("blend should not run")),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "stitch_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("stitch should not run")),
    )

    image, latent, _, _, loaded_vae = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=512,
        height=768,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={},
        inpaint_config={"image": "image", "mask": "mask", "denoise": 1.0},
        decode_image=False,
    )

    assert image is None
    assert latent == {"samples": "sampled"}
    assert loaded_vae == "vae"


def test_flux2_pipeline_decodes_inpaint_sample_preview_without_final_image(monkeypatch):
    previews = {
        pipeline.INPAINT_PREVIEW_REQUESTED: {
            pipeline.INPAINT_PREVIEW_SOURCE: False,
            pipeline.INPAINT_PREVIEW_SAMPLE: True,
            pipeline.INPAINT_PREVIEW_MASK: False,
        },
        pipeline.INPAINT_PREVIEW_SOURCE: None,
        pipeline.INPAINT_PREVIEW_SAMPLE: None,
        pipeline.INPAINT_PREVIEW_MASK: None,
    }

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(image="crop", mask="mask", stitcher="stitcher"),
    )
    monkeypatch.setattr(pipeline, "encode_image_to_latent", lambda **kwargs: {"samples": "source_ref"})
    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", lambda **kwargs: ("positive+ref", "unused"))
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_model_conditioning",
        lambda **kwargs: ("inpaint_positive", "inpaint_negative", {"samples": "inpaint", "noise_mask": "mask"}),
    )
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", lambda sigmas, denoise, **kwargs: sigmas)
    monkeypatch.setattr(pipeline, "sample_with_sigmas", lambda **kwargs: {"samples": "sampled"})
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "decoded_preview")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "blend_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("blend should not run")),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "stitch_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("stitch should not run")),
    )

    image, latent, _, _, loaded_vae = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=512,
        height=768,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={},
        inpaint_config={"image": "image", "mask": "mask", "denoise": 1.0},
        inpaint_previews=previews,
        decode_image=False,
    )

    assert image is None
    assert latent == {"samples": "sampled"}
    assert loaded_vae == "vae"
    assert previews[pipeline.INPAINT_PREVIEW_SAMPLE] == "decoded_preview"


def test_flux2_pipeline_passes_inpaint_denoise_to_ksampler_scheduler(monkeypatch):
    calls = {}

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(image="crop", mask="mask", stitcher="stitcher"),
    )
    monkeypatch.setattr(pipeline, "encode_image_to_latent", lambda **kwargs: {"samples": "source_ref"})
    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", lambda **kwargs: ("positive+ref", "unused"))
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_model_conditioning",
        lambda **kwargs: ("inpaint_positive", "inpaint_negative", {"samples": "inpaint", "noise_mask": "mask"}),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "resolve_denoise_schedule_steps",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("non-auto schedulers should delegate denoise schedule expansion to ComfyUI KSampler")
        ),
    )

    def fake_sample(**kwargs):
        calls["sample"] = kwargs
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("decode should not run")),
    )

    _, latent, _, _, _ = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=512,
        height=768,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        settings={},
        inpaint_config={"image": "image", "mask": "mask", "denoise": 0.35, "steps": 3},
        decode_image=False,
    )

    assert latent == {"samples": "sampled"}
    assert calls["sample"]["latent"] == {"samples": "inpaint", "noise_mask": "mask"}
    assert calls["sample"]["denoise"] == 0.35
    assert calls["sample"]["steps"] == 3


def test_flux2_pipeline_downscales_no_crop_inpaint_before_sampling(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = {}
    image = torch.rand((1, 1024, 1024, 3))
    mask = torch.zeros((1, 1024, 1024))
    mask[:, 256:768, 256:768] = 1.0

    fake_comfy = ModuleType("comfy")
    fake_utils = ModuleType("comfy.utils")

    def fake_common_upscale(samples, width, height, upscale_method, crop):
        calls["resize"] = {
            "width": width,
            "height": height,
            "upscale_method": upscale_method,
            "crop": crop,
        }
        return torch.zeros((samples.shape[0], samples.shape[1], int(height), int(width)))

    fake_utils.common_upscale = fake_common_upscale
    fake_comfy.utils = fake_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS={}))
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")

    def fake_encode_image_to_latent(**kwargs):
        calls["source_reference_shape"] = tuple(kwargs["image"].shape)
        return {"samples": "source_ref"}

    def fake_apply_references(**kwargs):
        calls["reference_latents"] = kwargs["reference_latents"]
        return "positive+ref", "unused"

    def fake_inpaint_conditioning(**kwargs):
        calls["conditioning_image_shape"] = tuple(kwargs["image"].shape)
        calls["conditioning_mask_shape"] = tuple(kwargs["mask"].shape)
        noise_mask = kwargs["mask"].reshape((-1, 1, kwargs["mask"].shape[-2], kwargs["mask"].shape[-1]))
        return "inpaint_positive", "inpaint_negative", {"samples": "inpaint", "noise_mask": noise_mask}

    def fake_sample(**kwargs):
        calls["sample"] = kwargs
        return {"samples": "sampled"}

    monkeypatch.setattr(pipeline, "encode_image_to_latent", fake_encode_image_to_latent)
    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", fake_apply_references)
    monkeypatch.setattr(pipeline.inpaint_service, "apply_inpaint_model_conditioning", fake_inpaint_conditioning)
    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("decode should not run")),
    )

    _, latent, _, _, _ = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=1024,
        height=1024,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        settings={},
        inpaint_config={
            "image": image,
            "mask": mask,
            "denoise": 1.0,
            "max_full_frame_megapixels": 0.25,
            "max_full_frame_side": 512,
        },
        decode_image=False,
    )

    assert latent == {"samples": "sampled"}
    assert calls["resize"] == {
        "width": 512,
        "height": 512,
        "upscale_method": "bilinear",
        "crop": "center",
    }
    assert calls["source_reference_shape"] == (1, 512, 512, 3)
    assert calls["conditioning_image_shape"] == (1, 512, 512, 3)
    assert calls["conditioning_mask_shape"] == (1, 512, 512)
    assert calls["sample"]["latent"]["noise_mask"].shape == (1, 1, 512, 512)


def test_flux2_pipeline_skips_sampling_when_inpaint_denoise_is_zero(monkeypatch):
    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: pipeline.inpaint_service.InpaintSource(image="crop", mask="mask", stitcher="stitcher"),
    )
    monkeypatch.setattr(pipeline, "encode_image_to_latent", lambda **kwargs: {"samples": "source_ref"})
    monkeypatch.setattr(pipeline, "apply_reference_latents_to_conditioning", lambda **kwargs: ("positive+ref", "unused"))
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_model_conditioning",
        lambda **kwargs: ("inpaint_positive", "inpaint_negative", {"samples": "inpaint", "noise_mask": "mask"}),
    )
    monkeypatch.setattr(
        pipeline,
        "sample_with_sigmas",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("sampling should not run")),
    )
    monkeypatch.setattr(
        pipeline,
        "sample_with_comfy_ksampler",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("sampling should not run")),
    )
    monkeypatch.setattr(pipeline, "decode_latent", lambda **kwargs: "decoded_image")
    monkeypatch.setattr(pipeline.inpaint_service, "stitch_inpaint_image", lambda **kwargs: "stitched_image")
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "blend_inpaint_image",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("crop path should stitch, not blend")),
    )

    image, latent, _, _, loaded_vae = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="negative",
        width=512,
        height=768,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={},
        inpaint_config={"image": "image", "mask": "mask", "denoise": 0.0},
    )

    assert image == "stitched_image"
    assert latent == {"samples": "inpaint", "noise_mask": "mask"}
    assert loaded_vae == "vae"


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


def test_flux2_pipeline_shares_vae_for_references_and_inpaint(monkeypatch):
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
        lambda **kwargs: events.append(f"ref:{kwargs['vae']}:{kwargs['image']}") or {"samples": "ref"},
    )
    monkeypatch.setattr(
        pipeline,
        "apply_reference_latents_to_conditioning",
        lambda **kwargs: ("positive+refs", "negative+refs"),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "prepare_inpaint_source",
        lambda **kwargs: events.append("crop") or pipeline.inpaint_service.InpaintSource(
            image="crop",
            mask="mask",
            stitcher="stitcher",
        ),
    )
    monkeypatch.setattr(
        pipeline.inpaint_service,
        "apply_inpaint_model_conditioning",
        lambda **kwargs: events.append(f"inpaint:{kwargs['vae']}") or (
            "positive+inpaint",
            "negative+inpaint",
            {"samples": "inpaint", "noise_mask": "mask"},
        ),
    )
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: "sigmas")
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", lambda sigmas, denoise, **kwargs: sigmas)
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
        positive_prompt="positive",
        negative_prompt="negative",
        width=1024,
        height=1024,
        seed=0,
        steps=4,
        cfg=1.5,
        sampler="auto",
        scheduler="auto",
        settings={},
        reference_inputs=SimpleNamespace(images=("first",)),
        inpaint_config={"image": "image", "mask": "mask", "denoise": 1.0},
        decode_image=False,
    )

    assert image is None
    assert latent == {"samples": "sampled"}
    assert positive == "positive+inpaint"
    assert negative == "negative+inpaint"
    assert loaded_vae == "vae"
    assert events == [
        "load_vae",
        "crop",
        "scale:first",
        "ref:vae:scaled:first",
        "ref:vae:crop",
        "inpaint:vae",
    ]


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


def test_second_pass_steps_zero_reuses_main_steps():
    config = pipeline.normalize_second_pass_config({"enabled": True})

    assert config["steps_input"] == 0
    assert pipeline.resolve_second_pass_steps(config, 11) == 11
    assert pipeline.second_pass_status(config, main_steps=11)["steps"] == 11


def test_second_pass_helper_upscales_encodes_samples_and_decodes(monkeypatch):
    calls = {}
    _install_fake_common_upscale(monkeypatch, calls)
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "loaded_vae")

    def fake_encode(**kwargs):
        calls["encode"] = kwargs
        return {"samples": "encoded_upscale"}

    def fake_decode(**kwargs):
        calls["decode"] = kwargs
        return "decoded_second_pass"

    def fake_sample(latent, width, height, denoise, steps):
        calls["sample"] = {
            "latent": latent,
            "width": width,
            "height": height,
            "denoise": denoise,
            "steps": steps,
        }
        return {"samples": "sampled_second_pass"}

    monkeypatch.setattr(pipeline, "encode_image_to_latent", fake_encode)
    monkeypatch.setattr(pipeline, "decode_latent", fake_decode)

    image, latent, loaded_vae = pipeline.apply_second_sampler_pass(
        config={"enabled": True, "steps_input": 5},
        image=FakeSecondPassImage(),
        latent={
            "samples": "first_pass",
            pipeline.PID_CAPTURE_KEY: {"latent": "pid", "sigma": 0.1},
        },
        vae="vae.safetensors",
        loaded_vae=None,
        sample_latent=fake_sample,
        dimension_multiple=16,
        main_steps=9,
    )

    assert image == "decoded_second_pass"
    assert loaded_vae == "loaded_vae"
    assert calls["upscale"]["width"] == 1536
    assert calls["upscale"]["height"] == 1536
    assert calls["upscale"]["upscale_method"] == "lanczos"
    assert calls["upscale"]["crop"] == "disabled"
    assert calls["encode"]["image"] == "upscaled_image"
    assert calls["sample"] == {
        "latent": {"samples": "encoded_upscale"},
        "width": 1536,
        "height": 1536,
        "denoise": 0.15,
        "steps": 5,
    }
    assert calls["decode"]["latent"]["samples"] == "sampled_second_pass"
    assert latent[pipeline.PID_CAPTURE_KEY] == {"latent": "pid", "sigma": 0.1}
    assert latent[pipeline.SECOND_PASS_INFO_KEY] == {
        "enabled": True,
        "applied": True,
        "denoise": 0.15,
        "steps_input": 5,
        "steps": 5,
        "upscale_ratio": 1.5,
        "upscale_method": "lanczos",
        "first_pass_size": {"width": 1024, "height": 1024},
        "final_size": {"width": 1536, "height": 1536},
    }


def test_z_image_pipeline_second_pass_reuses_ksampler_without_final_decode(monkeypatch):
    sample_calls = []
    decode_calls = []

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model+lora", "clip+lora", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_z_image_prompt", lambda **kwargs: "conditioning")
    monkeypatch.setattr(pipeline, "make_empty_z_image_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "upscale_image_by_ratio", lambda **kwargs: ("upscaled_image", 1536, 1536))
    monkeypatch.setattr(pipeline, "encode_image_to_latent", lambda **kwargs: {"samples": "upscaled_latent"})

    def fake_sample(**kwargs):
        sample_calls.append(kwargs)
        return {"samples": f"sampled_{len(sample_calls)}"}

    def fake_decode(**kwargs):
        decode_calls.append(kwargs)
        if len(decode_calls) == 1:
            return FakeSecondPassImage()
        raise AssertionError("final second-pass image should not decode")

    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)
    monkeypatch.setattr(pipeline, "decode_latent", fake_decode)

    image, latent, _, _, loaded_vae = pipeline.generate_z_image_turbo_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=1024,
        height=1024,
        seed=123,
        steps=8,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        settings={},
        second_pass_config={"enabled": True, "decode_image": False, "return_image_original": True},
    )

    assert image is None
    assert loaded_vae == "vae"
    assert len(sample_calls) == 2
    assert sample_calls[1]["model"] == "model+lora"
    assert sample_calls[1]["positive"] == "conditioning"
    assert sample_calls[1]["latent"] == {"samples": "upscaled_latent"}
    assert sample_calls[1]["denoise"] == 0.15
    assert sample_calls[1]["steps"] == 8
    assert latent["samples"] == "sampled_2"
    assert latent[pipeline.SECOND_PASS_ORIGINAL_IMAGE_KEY].shape == (1, 1024, 1024, 3)
    assert latent[pipeline.SECOND_PASS_INFO_KEY]["steps_input"] == 0
    assert latent[pipeline.SECOND_PASS_INFO_KEY]["steps"] == 8
    assert latent[pipeline.SECOND_PASS_INFO_KEY]["final_size"] == {"width": 1536, "height": 1536}


def test_krea2_pipeline_second_pass_reuses_enhanced_model(monkeypatch):
    sample_calls = []

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model+lora", "clip+lora", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_krea2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline.krea2_enhancer, "apply_krea2_enhancer", lambda *args, **kwargs: "enhanced_model")
    monkeypatch.setattr(pipeline, "make_empty_krea2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "upscale_image_by_ratio", lambda **kwargs: ("upscaled_image", 1536, 1536))
    monkeypatch.setattr(pipeline, "encode_image_to_latent", lambda **kwargs: {"samples": "upscaled_latent"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: FakeSecondPassImage() if len(sample_calls) < 2 else "second_pass_image",
    )

    def fake_sample(**kwargs):
        sample_calls.append(kwargs)
        return {"samples": f"sampled_{len(sample_calls)}"}

    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)

    image, latent, positive, _, _ = pipeline.generate_krea2_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=1024,
        height=1024,
        seed=123,
        steps=8,
        cfg=1.0,
        sampler="er_sde",
        scheduler="simple",
        settings={"enhancer_enabled": True, "enhancer_strength": 0.5},
        second_pass_config={"enabled": True, "decode_image": True, "steps_input": 6},
    )

    assert image == "second_pass_image"
    assert positive == "positive"
    assert sample_calls[0]["model"] == "enhanced_model"
    assert sample_calls[1]["model"] == "enhanced_model"
    assert sample_calls[1]["positive"] == "positive"
    assert sample_calls[1]["negative"] == "negative"
    assert sample_calls[1]["denoise"] == 0.15
    assert sample_calls[1]["steps"] == 6
    assert latent["samples"] == "sampled_2"
    assert latent[pipeline.SECOND_PASS_INFO_KEY]["steps"] == 6


def test_flux2_pipeline_second_pass_uses_upscaled_flux_sigmas(monkeypatch):
    sigmas_calls = []
    denoise_calls = []
    sample_calls = []

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "make_empty_flux2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "upscale_image_by_ratio", lambda **kwargs: ("upscaled_image", 1536, 1536))
    monkeypatch.setattr(pipeline, "encode_image_to_latent", lambda **kwargs: {"samples": "upscaled_latent"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: FakeSecondPassImage() if len(sample_calls) < 2 else "second_pass_image",
    )

    def fake_flux2_sigmas(**kwargs):
        sigmas_calls.append(kwargs)
        return f"sigmas_{kwargs['width']}x{kwargs['height']}"

    def fake_denoise(sigmas, denoise):
        denoise_calls.append({"sigmas": sigmas, "denoise": denoise})
        return f"denoised_{sigmas}_{denoise}"

    def fake_sample(**kwargs):
        sample_calls.append(kwargs)
        return {"samples": f"sampled_{len(sample_calls)}"}

    monkeypatch.setattr(pipeline, "flux2_sigmas", fake_flux2_sigmas)
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", fake_denoise)
    monkeypatch.setattr(pipeline, "sample_with_sigmas", fake_sample)

    image, latent, _, _, _ = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={},
        second_pass_config={"enabled": True, "decode_image": True, "steps_input": 6},
    )

    assert image == "second_pass_image"
    assert sigmas_calls == [
        {"steps": 4, "width": 1024, "height": 1024},
        {"steps": 6, "width": 1536, "height": 1536},
    ]
    assert denoise_calls == [{"sigmas": "sigmas_1536x1536", "denoise": 0.15}]
    assert sample_calls[1]["sigmas"] == "denoised_sigmas_1536x1536_0.15"
    assert sample_calls[1]["latent"] == {"samples": "upscaled_latent"}
    assert sample_calls[1]["steps"] == 6
    assert latent["samples"] == "sampled_2"


def test_flux2_pipeline_second_pass_non_auto_uses_ksampler_steps(monkeypatch):
    sigmas_calls = []
    sample_calls = []

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_lora_config", lambda **kwargs: ("model", "clip", []))
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_flux2_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "make_empty_flux2_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "upscale_image_by_ratio", lambda **kwargs: ("upscaled_image", 1536, 1536))
    monkeypatch.setattr(pipeline, "encode_image_to_latent", lambda **kwargs: {"samples": "upscaled_latent"})
    monkeypatch.setattr(pipeline, "flux2_sigmas", lambda **kwargs: sigmas_calls.append(kwargs) or "sigmas")
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: FakeSecondPassImage() if len(sample_calls) < 2 else "second_pass_image",
    )

    def fake_sample(**kwargs):
        sample_calls.append(kwargs)
        return {"samples": f"sampled_{len(sample_calls)}"}

    monkeypatch.setattr(pipeline, "sample_with_comfy_ksampler", fake_sample)

    image, latent, _, _, _ = pipeline.generate_flux2_klein_t2i(
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        settings={},
        second_pass_config={"enabled": True, "decode_image": True, "steps_input": 7},
    )

    assert image == "second_pass_image"
    assert sigmas_calls == []
    assert sample_calls[0]["steps"] == 4
    assert sample_calls[1]["steps"] == 7
    assert sample_calls[1]["latent"] == {"samples": "upscaled_latent"}
    assert sample_calls[1]["denoise"] == 0.15
    assert latent["samples"] == "sampled_2"


def test_ideogram4_pipeline_second_pass_uses_custom_guider_sigmas(monkeypatch):
    sigmas_calls = []
    denoise_calls = []
    sample_calls = []

    monkeypatch.setattr(pipeline, "load_diffusion_model", lambda **kwargs: f"model:{kwargs['diffusion_model']}")
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(pipeline, "apply_model_sampling_aura", lambda **kwargs: "conditional+aura")
    monkeypatch.setattr(pipeline, "apply_lora_config_model_only", lambda **kwargs: ("conditional+lora", []))
    monkeypatch.setattr(pipeline, "apply_cfg_override", lambda **kwargs: "conditional+cfg")
    monkeypatch.setattr(pipeline, "load_vae", lambda **kwargs: "vae")
    monkeypatch.setattr(pipeline, "encode_ideogram4_prompt", lambda **kwargs: "positive")
    monkeypatch.setattr(pipeline, "zero_out_conditioning", lambda conditioning: "negative")
    monkeypatch.setattr(pipeline, "make_empty_ideogram4_latent", lambda **kwargs: {"samples": "empty"})
    monkeypatch.setattr(pipeline, "build_dual_model_guider", lambda **kwargs: "guider")
    monkeypatch.setattr(pipeline, "upscale_image_by_ratio", lambda **kwargs: ("upscaled_image", 1536, 1536))
    monkeypatch.setattr(pipeline, "encode_image_to_latent", lambda **kwargs: {"samples": "upscaled_latent"})
    monkeypatch.setattr(
        pipeline,
        "decode_latent",
        lambda **kwargs: FakeSecondPassImage() if len(sample_calls) < 2 else "second_pass_image",
    )

    def fake_sigmas(**kwargs):
        sigmas_calls.append(kwargs)
        return f"ideogram_sigmas_{kwargs['width']}x{kwargs['height']}"

    def fake_denoise(sigmas, denoise):
        denoise_calls.append({"sigmas": sigmas, "denoise": denoise})
        return f"denoised_{sigmas}_{denoise}"

    def fake_sample(**kwargs):
        sample_calls.append(kwargs)
        return {"samples": f"sampled_{len(sample_calls)}"}

    monkeypatch.setattr(pipeline, "ideogram4_sigmas", fake_sigmas)
    monkeypatch.setattr(pipeline.inpaint_service, "apply_denoise_to_sigmas", fake_denoise)
    monkeypatch.setattr(pipeline, "sample_with_custom_guider", fake_sample)

    image, latent, _, _, _ = pipeline.generate_ideogram4_t2i(
        diffusion_model="model.safetensors",
        unconditional_model="unconditional.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="prompt",
        width=1024,
        height=1024,
        seed=123,
        steps=20,
        sampler="euler",
        scheduler="ideogram4",
        settings={"mu": 0.0, "std": 1.75},
        second_pass_config={"enabled": True, "decode_image": True, "steps_input": 12},
    )

    assert image == "second_pass_image"
    assert sigmas_calls == [
        {"steps": 20, "width": 1024, "height": 1024, "mu": 0.0, "std": 1.75},
        {"steps": 12, "width": 1536, "height": 1536, "mu": 0.0, "std": 1.75},
    ]
    assert denoise_calls == [
        {"sigmas": "ideogram_sigmas_1536x1536", "denoise": 0.15},
    ]
    assert sample_calls[1]["guider"] == "guider"
    assert sample_calls[1]["sigmas"] == "denoised_ideogram_sigmas_1536x1536_0.15"
    assert sample_calls[1]["latent"] == {"samples": "upscaled_latent"}
    assert latent["samples"] == "sampled_2"
    assert latent[pipeline.SECOND_PASS_INFO_KEY]["steps"] == 12


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


def test_sample_with_sigmas_forwards_noise_mask(monkeypatch):
    calls = {}

    comfy_module = ModuleType("comfy")
    sample_module = ModuleType("comfy.sample")
    utils_module = ModuleType("comfy.utils")
    latent_preview_module = ModuleType("latent_preview")
    sigmas = SimpleNamespace(shape=(5,))
    model = SimpleNamespace(load_device="cuda", model_options={})

    def fake_sample(*args, **kwargs):
        calls["sample_args"] = args
        calls["sample"] = kwargs
        return "sampled_samples"

    sample_module.fix_empty_latent_channels = lambda model, latent_image, downscale_ratio_spacial, downscale_ratio_temporal: latent_image
    sample_module.prepare_noise = lambda latent_image, seed, batch_inds: "noise"
    sample_module.sample = fake_sample
    utils_module.PROGRESS_BAR_ENABLED = True
    latent_preview_module.prepare_callback = lambda model, steps: (lambda *args: None)
    comfy_module.sample = sample_module
    comfy_module.utils = utils_module

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sample", sample_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils_module)
    monkeypatch.setitem(sys.modules, "latent_preview", latent_preview_module)

    out = pipeline.sample_with_sigmas(
        model=model,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        positive="positive",
        negative="negative",
        latent={"samples": "latent_samples", "noise_mask": "noise_mask"},
        sigmas=sigmas,
    )

    assert out == {"samples": "sampled_samples", "noise_mask": "noise_mask"}
    assert calls["sample"]["noise_mask"] == "noise_mask"
    assert calls["sample"]["sigmas"] is sigmas


def test_sample_with_comfy_ksampler_uses_progress_callback(monkeypatch):
    calls = {}

    class FakeSigmas:
        shape = (5,)

        def __getitem__(self, index):
            return [1.0, 0.75, 0.5, 0.25, 0.0][index]

    class FakeKSampler:
        def __init__(self, *args, **kwargs):
            calls["ksampler"] = {"args": args, "kwargs": kwargs}
            self.sigmas = FakeSigmas()

    class FakeProgress:
        def prepare_sampling_callback(self, model, steps, x0_output_dict=None):
            calls["progress"] = {
                "model": model,
                "steps": steps,
                "x0_output_dict": x0_output_dict,
            }

            def callback(step, x0, x, total_steps):
                calls["progress_callback"] = (step, x0, x, total_steps)

            return callback

    comfy_module = ModuleType("comfy")
    sample_module = ModuleType("comfy.sample")
    samplers_module = ModuleType("comfy.samplers")
    utils_module = ModuleType("comfy.utils")
    model = SimpleNamespace(load_device="cuda", model_options={})

    def fake_sample(*args, **kwargs):
        calls["sample"] = {"args": args, "kwargs": kwargs}
        kwargs["callback"](0, "x0", "x", 4)
        return "sampled_samples"

    sample_module.fix_empty_latent_channels = lambda model, latent_image, downscale_ratio_spacial, downscale_ratio_temporal: latent_image
    sample_module.prepare_noise = lambda latent_image, seed, batch_inds: "noise"
    sample_module.sample = fake_sample
    samplers_module.KSampler = FakeKSampler
    utils_module.PROGRESS_BAR_ENABLED = True
    comfy_module.sample = sample_module
    comfy_module.samplers = samplers_module
    comfy_module.utils = utils_module

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sample", sample_module)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils_module)

    out = pipeline.sample_with_comfy_ksampler(
        model=model,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        positive="positive",
        negative="negative",
        latent={"samples": "latent_samples"},
        progress=FakeProgress(),
    )

    assert out == {"samples": "sampled_samples"}
    assert calls["progress"]["model"] is model
    assert calls["progress"]["steps"] == 4
    assert calls["progress_callback"] == (0, "x0", "x", 4)


def test_sample_with_sigmas_uses_progress_callback_and_pid_capture(monkeypatch):
    calls = {}

    class FakeTensor:
        def detach(self):
            return self

        def to(self, device):
            calls["capture_device"] = device
            return self

        def contiguous(self):
            return "captured_samples"

    class FakeSigmas:
        shape = (5,)

        def __getitem__(self, index):
            return [1.0, 0.75, 0.5, 0.25, 0.0][index]

    class FakeProgress:
        def prepare_sampling_callback(self, model, steps, x0_output_dict=None):
            calls["progress"] = {
                "model": model,
                "steps": steps,
                "x0_output_dict": x0_output_dict,
            }

            def callback(step, x0, x, total_steps):
                calls["progress_callback"] = (step, x0, x, total_steps)

            return callback

    comfy_module = ModuleType("comfy")
    sample_module = ModuleType("comfy.sample")
    utils_module = ModuleType("comfy.utils")
    latent_preview_module = ModuleType("latent_preview")
    sigmas = FakeSigmas()
    model = SimpleNamespace(load_device="cuda", model_options={})

    def fail_prepare_callback(*args, **kwargs):
        raise AssertionError("progress callback should be used before latent_preview fallback")

    def fake_sample(*args, **kwargs):
        calls["sample"] = {"args": args, "kwargs": kwargs}
        kwargs["callback"](1, "x0", FakeTensor(), 4)
        return "sampled_samples"

    sample_module.fix_empty_latent_channels = lambda model, latent_image, downscale_ratio_spacial, downscale_ratio_temporal: latent_image
    sample_module.prepare_noise = lambda latent_image, seed, batch_inds: "noise"
    sample_module.sample = fake_sample
    utils_module.PROGRESS_BAR_ENABLED = True
    latent_preview_module.prepare_callback = fail_prepare_callback
    comfy_module.sample = sample_module
    comfy_module.utils = utils_module

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sample", sample_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils_module)
    monkeypatch.setitem(sys.modules, "latent_preview", latent_preview_module)

    out = pipeline.sample_with_sigmas(
        model=model,
        seed=123,
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        positive="positive",
        negative="negative",
        latent={"samples": "latent_samples"},
        sigmas=sigmas,
        pid_capture_step=2,
        progress=FakeProgress(),
    )

    assert out["samples"] == "sampled_samples"
    assert calls["progress"]["steps"] == 4
    assert calls["progress_callback"][0:2] == (1, "x0")
    assert calls["capture_device"] == "cpu"
    pid = out[pipeline.PID_CAPTURE_KEY]
    assert pid["sigma"] == 0.75
    assert pid["step"] == 2
    assert pid["latent"]["samples"] == "captured_samples"
    assert pid["latent"]["pid_sigma"] == 0.75
    assert pid["latent"]["pid_capture_step"] == 2


def test_sample_with_custom_guider_uses_progress_callback(monkeypatch):
    calls = {}

    class FakeSigmas:
        shape = (3,)

        def __getitem__(self, index):
            return [1.0, 0.5, 0.0][index]

    class FakeNoise:
        def __init__(self, seed):
            self.seed = seed

        def generate_noise(self, latent):
            calls["noise_latent"] = latent
            return "noise"

    class FakeRandomNoise:
        @staticmethod
        def execute(noise_seed):
            calls["noise_seed"] = noise_seed
            return (FakeNoise(noise_seed),)

    class FakeKSamplerSelect:
        @staticmethod
        def execute(sampler_name):
            calls["sampler_name"] = sampler_name
            return ("sampler_obj",)

    class FakeProgress:
        def prepare_sampling_callback(self, model, steps, x0_output_dict=None):
            calls["progress"] = {
                "model": model,
                "steps": steps,
                "x0_output_dict": x0_output_dict,
            }

            def callback(step, x0, x, total_steps):
                calls["progress_callback"] = (step, x0, x, total_steps)

            return callback

    class FakeGuider:
        def __init__(self):
            self.model_patcher = SimpleNamespace(
                load_device="cuda",
                model=SimpleNamespace(latent_format="flux"),
            )

        def sample(
            self,
            noise,
            latent_image,
            sampler,
            sigmas,
            *,
            denoise_mask=None,
            callback=None,
            disable_pbar=None,
            seed=None,
        ):
            calls["guider_sample"] = {
                "noise": noise,
                "latent_image": latent_image,
                "sampler": sampler,
                "sigmas": sigmas,
                "denoise_mask": denoise_mask,
                "disable_pbar": disable_pbar,
                "seed": seed,
            }
            callback(0, "x0", "x", 2)
            return "sampled_samples"

    comfy_module = ModuleType("comfy")
    sample_module = ModuleType("comfy.sample")
    utils_module = ModuleType("comfy.utils")
    model_management_module = ModuleType("comfy.model_management")
    comfy_extras_module = ModuleType("comfy_extras")
    custom_sampler_module = ModuleType("comfy_extras.nodes_custom_sampler")
    sigmas = FakeSigmas()
    guider = FakeGuider()

    sample_module.fix_empty_latent_channels = lambda model, latent_image, downscale_ratio_spacial, downscale_ratio_temporal: latent_image
    utils_module.PROGRESS_BAR_ENABLED = False
    model_management_module.intermediate_device = lambda: "intermediate"
    custom_sampler_module.RandomNoise = FakeRandomNoise
    custom_sampler_module.KSamplerSelect = FakeKSamplerSelect
    comfy_module.sample = sample_module
    comfy_module.utils = utils_module
    comfy_module.model_management = model_management_module

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sample", sample_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils_module)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management_module)
    monkeypatch.setitem(sys.modules, "comfy_extras", comfy_extras_module)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", custom_sampler_module)

    out = pipeline.sample_with_custom_guider(
        guider=guider,
        seed=321,
        sampler="euler",
        sigmas=sigmas,
        latent={"samples": "latent_samples", "noise_mask": "mask"},
        progress=FakeProgress(),
    )

    assert out == {"samples": "sampled_samples", "noise_mask": "mask"}
    assert calls["noise_seed"] == 321
    assert calls["sampler_name"] == "euler"
    assert calls["progress"]["model"] is guider.model_patcher
    assert calls["progress"]["steps"] == 2
    assert calls["progress_callback"] == (0, "x0", "x", 2)
    assert calls["guider_sample"]["denoise_mask"] == "mask"
    assert calls["guider_sample"]["disable_pbar"] is True
    assert calls["guider_sample"]["seed"] == 321


def test_pid_capture_step_defaults_near_end_and_clamps():
    assert pipeline.resolve_pid_capture_step(None, 50) is None
    assert pipeline.resolve_pid_capture_step(0, 50) == 46
    assert pipeline.resolve_pid_capture_step(0, 4) == 4
    assert pipeline.resolve_pid_capture_step(99, 8) == 8


def test_pid_capture_sidecar_attaches_latent_sigma_and_step():
    source = {"samples": "source", "downscale_ratio_spacial": 16}
    final = {"samples": "final"}
    captured = {"samples": "captured", "sigma": 0.342}

    out = pipeline._attach_pid_capture(
        latent=final,
        source_latent=source,
        captured=captured,
        fallback_samples="fallback",
        target_step=46,
    )

    pid = out[pipeline.PID_CAPTURE_KEY]
    assert out["samples"] == "final"
    assert pid["sigma"] == 0.342
    assert pid["step"] == 46
    assert pid["latent"] == {
        "samples": "captured",
        "pid_sigma": 0.342,
        "pid_capture_step": 46,
    }
