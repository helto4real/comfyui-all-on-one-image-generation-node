import json

from services.run_info import build_run_info, to_json


def test_run_info_json_serializable_and_contains_core_fields():
    run_info = build_run_info(
        model_type="z_image_turbo",
        display_name="Z-Image Turbo",
        diffusion_model="diffusion_models/model.safetensors",
        diffusion_model_format="safetensors",
        text_encoder="text_encoders/text.safetensors",
        text_encoder_format="safetensors",
        vae="vae/vae.safetensors",
        vae_format="safetensors",
        width=1024,
        height=1024,
        seed=123,
        steps=8,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={
            "family": "z_image_turbo",
            "steps": 8,
            "attention_mode": "auto",
            "resolved_attention_mode": "sage",
            "torch_compile_mode": "on",
            "torch_compile_backend": "inductor",
            "resolved_torch_compile_mode": "on",
            "resolved_torch_compile_backend": "inductor",
            "performance_apply_timing": "after_loras",
        },
        warnings=["warning"],
        adapter_version="0.1.0",
        loras=[{"name": "style.safetensors", "strength_model": 0.8, "strength_clip": 0.8}],
    )

    serialized = to_json(run_info)
    parsed = json.loads(serialized)
    assert "\n" in serialized
    assert '  "adapter_version"' in serialized
    assert parsed["model_type"] == "z_image_turbo"
    assert parsed["diffusion_model_format"] == "safetensors"
    assert parsed["text_encoder_format"] == "safetensors"
    assert parsed["vae_format"] == "safetensors"
    assert parsed["seed"] == 123
    assert parsed["steps"] == 8
    assert parsed["warnings"] == ["warning"]
    assert parsed["loras"][0]["name"] == "style.safetensors"
    assert parsed["settings"]["resolved_attention_mode"] == "sage"
    assert parsed["settings"]["resolved_torch_compile_mode"] == "on"
    assert parsed["settings"]["performance_apply_timing"] == "after_loras"
    assert parsed["performance"] == {
        "configured": True,
        "attention_mode": "auto",
        "resolved_attention_mode": "sage",
        "torch_compile_mode": "on",
        "torch_compile_backend": "inductor",
        "resolved_torch_compile_mode": "on",
        "resolved_torch_compile_backend": "inductor",
        "performance_apply_timing": "after_loras",
    }


def test_run_info_reports_performance_defaults_when_settings_are_absent():
    run_info = build_run_info(
        model_type="z_image_turbo",
        display_name="Z-Image Turbo",
        diffusion_model="diffusion_models/model.safetensors",
        diffusion_model_format="safetensors",
        text_encoder="text_encoders/text.safetensors",
        text_encoder_format="safetensors",
        vae="vae/vae.safetensors",
        vae_format="safetensors",
        width=1024,
        height=1024,
        seed=123,
        steps=8,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={"family": "z_image_turbo", "steps": 8},
        warnings=[],
        adapter_version="0.1.0",
    )

    parsed = json.loads(to_json(run_info))
    assert parsed["performance"] == {
        "configured": False,
        "attention_mode": "off",
        "resolved_attention_mode": "off",
        "torch_compile_mode": "off",
        "torch_compile_backend": "inductor",
        "resolved_torch_compile_mode": "off",
        "resolved_torch_compile_backend": "off",
        "performance_apply_timing": "after_loras",
    }
