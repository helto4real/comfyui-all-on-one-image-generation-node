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
