import pytest

from nodes.pipeline_models import AIOLoadPipelineModels


def test_pipeline_models_node_returns_model_and_clip():
    assert AIOLoadPipelineModels.RETURN_TYPES == ("MODEL", "CLIP")
    assert AIOLoadPipelineModels.RETURN_NAMES == ("model", "clip")
    assert AIOLoadPipelineModels.CATEGORY == "AIO/Image"


def test_pipeline_models_node_loads_with_model_family_clip_type(monkeypatch):
    from nodes import pipeline_models

    calls = {}

    def fake_load_model(**kwargs):
        calls["model"] = kwargs
        return "model"

    def fake_load_clip(**kwargs):
        calls["clip"] = kwargs
        return "clip"

    monkeypatch.setattr(pipeline_models.pipeline, "load_diffusion_model", fake_load_model)
    monkeypatch.setattr(pipeline_models.pipeline, "load_text_encoder", fake_load_clip)
    monkeypatch.setattr(
        pipeline_models.pipeline,
        "apply_lora_config",
        lambda **kwargs: ("model+lora", "clip+lora", [{"name": "style"}]),
    )

    model, clip = AIOLoadPipelineModels().load(
        model_type="flux2_klein_9b",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        model_settings={"family": "flux2_klein_9b", "precision_policy": "bf16"},
        lora_config={"loras": [{"enabled": True, "name": "style"}]},
    )

    assert model == "model+lora"
    assert clip == "clip+lora"
    assert calls["model"] == {
        "diffusion_model": "model.safetensors",
        "precision_policy": "bf16",
    }
    assert calls["clip"] == {
        "text_encoder": "text.safetensors",
        "clip_type": "flux2",
    }


def test_pipeline_models_node_uses_ideogram4_clip_type_and_model_only_loras(monkeypatch):
    from nodes import pipeline_models

    calls = {}

    def fake_load_clip(**kwargs):
        calls["clip"] = kwargs
        return "clip"

    def fake_model_only_loras(**kwargs):
        calls["loras"] = kwargs
        return "model+lora", [{"name": "style"}]

    monkeypatch.setattr(pipeline_models.pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline_models.pipeline, "load_text_encoder", fake_load_clip)
    monkeypatch.setattr(
        pipeline_models.pipeline,
        "apply_lora_config",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Ideogram 4 should use model-only LoRAs")),
    )
    monkeypatch.setattr(pipeline_models.pipeline, "apply_lora_config_model_only", fake_model_only_loras)

    model, clip = AIOLoadPipelineModels().load(
        model_type="ideogram4",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        model_settings={"family": "ideogram4"},
        lora_config={"loras": [{"enabled": True, "name": "style"}]},
    )

    assert model == "model+lora"
    assert clip == "clip"
    assert calls["clip"] == {
        "text_encoder": "text.safetensors",
        "clip_type": "ideogram4",
    }
    assert calls["loras"]["model"] == "model"


def test_pipeline_models_node_uses_krea2_clip_type_and_performance(monkeypatch):
    from nodes import pipeline_models

    calls = {}

    def fake_load_model(**kwargs):
        calls["model"] = kwargs
        return "model"

    def fake_load_clip(**kwargs):
        calls["clip"] = kwargs
        return "clip"

    def fake_apply_loras(**kwargs):
        calls["loras"] = kwargs
        return "model+lora", "clip+lora", [{"name": "style"}]

    def fake_performance(**kwargs):
        calls["performance"] = kwargs
        return f"{kwargs['model']}+perf"

    monkeypatch.setattr(pipeline_models.pipeline, "load_diffusion_model", fake_load_model)
    monkeypatch.setattr(pipeline_models.pipeline, "load_text_encoder", fake_load_clip)
    monkeypatch.setattr(pipeline_models.pipeline, "apply_lora_config", fake_apply_loras)
    monkeypatch.setattr(pipeline_models.pipeline, "apply_model_performance", fake_performance)

    model, clip = AIOLoadPipelineModels().load(
        model_type="krea2",
        diffusion_model="krea/krea2_turbo_fp8.safetensors",
        text_encoder="qwen3vl_4b_fp8_scaled.safetensors",
        model_settings={
            "family": "krea2",
            "precision_policy": "fp8",
            "attention_mode": "off",
            "fp16_accumulation_enabled": True,
        },
        lora_config={"loras": [{"enabled": True, "name": "style"}]},
    )

    assert model == "model+lora+perf"
    assert clip == "clip+lora"
    assert calls["model"] == {
        "diffusion_model": "krea/krea2_turbo_fp8.safetensors",
        "precision_policy": "fp8",
    }
    assert calls["clip"] == {
        "text_encoder": "qwen3vl_4b_fp8_scaled.safetensors",
        "clip_type": "krea2",
    }
    assert calls["loras"]["model"] == "model"
    assert calls["loras"]["clip"] == "clip"
    assert calls["performance"]["model"] == "model+lora"
    assert calls["performance"]["settings"]["fp16_accumulation_enabled"] is True


def test_pipeline_models_node_applies_performance_after_loras(monkeypatch):
    from nodes import pipeline_models

    events = []

    monkeypatch.setattr(pipeline_models.pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline_models.pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(
        pipeline_models.pipeline,
        "apply_lora_config",
        lambda **kwargs: events.append("loras") or ("model+lora", "clip+lora", []),
    )
    monkeypatch.setattr(
        pipeline_models.pipeline,
        "apply_model_performance",
        lambda **kwargs: events.append(f"performance:{kwargs['model']}") or f"{kwargs['model']}+perf",
    )

    model, clip = AIOLoadPipelineModels().load(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        model_settings={"family": "z_image_turbo", "attention_mode": "off", "performance_apply_timing": "after_loras"},
    )

    assert model == "model+lora+perf"
    assert clip == "clip+lora"
    assert events == ["loras", "performance:model+lora"]


def test_pipeline_models_node_applies_performance_before_loras(monkeypatch):
    from nodes import pipeline_models

    events = []

    monkeypatch.setattr(pipeline_models.pipeline, "load_diffusion_model", lambda **kwargs: "model")
    monkeypatch.setattr(pipeline_models.pipeline, "load_text_encoder", lambda **kwargs: "clip")
    monkeypatch.setattr(
        pipeline_models.pipeline,
        "apply_model_performance",
        lambda **kwargs: events.append(f"performance:{kwargs['model']}") or f"{kwargs['model']}+perf",
    )
    monkeypatch.setattr(
        pipeline_models.pipeline,
        "apply_lora_config",
        lambda **kwargs: events.append(f"loras:{kwargs['model']}") or ("model+perf+lora", "clip+lora", []),
    )

    model, clip = AIOLoadPipelineModels().load(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        model_settings={"family": "z_image_turbo", "attention_mode": "off", "performance_apply_timing": "before_loras"},
    )

    assert model == "model+perf+lora"
    assert clip == "clip+lora"
    assert events == ["performance:model", "loras:model+perf"]


def test_pipeline_models_node_applies_empty_lora_config(monkeypatch):
    from nodes import pipeline_models

    calls = {}

    monkeypatch.setattr(
        pipeline_models.pipeline,
        "load_diffusion_model",
        lambda **kwargs: "model",
    )
    monkeypatch.setattr(
        pipeline_models.pipeline,
        "load_text_encoder",
        lambda **kwargs: "clip",
    )

    def fake_apply_loras(**kwargs):
        calls["lora_config"] = kwargs["lora_config"]
        return kwargs["model"], kwargs["clip"], []

    monkeypatch.setattr(pipeline_models.pipeline, "apply_lora_config", fake_apply_loras)

    model, clip = AIOLoadPipelineModels().load(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
    )

    assert model == "model"
    assert clip == "clip"
    assert calls["lora_config"] == {"version": 1, "loras": [], "ui": {"show_strengths": "single", "match": ""}}


def test_pipeline_models_node_rejects_settings_family_mismatch():
    with pytest.raises(ValueError, match="Selected settings are for flux2_klein_9b"):
        AIOLoadPipelineModels().load(
            model_type="z_image_turbo",
            diffusion_model="model.safetensors",
            text_encoder="text.safetensors",
            model_settings={"family": "flux2_klein_9b"},
        )
