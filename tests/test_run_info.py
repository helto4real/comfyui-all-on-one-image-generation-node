import json

import pytest
from helto_privacy import initialize_keystore

from services import privacy
from services.run_info import build_run_info, to_json

PASSWORD = "correct horse battery"


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
            "fp16_accumulation_enabled": True,
            "resolved_fp16_accumulation_enabled": True,
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
    assert parsed["batch"] == {"count": 1, "seeds": [123], "seed_mode": "increment"}
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
        "fp16_accumulation_enabled": True,
        "resolved_fp16_accumulation_enabled": True,
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


def test_run_info_reports_explicit_batch_metadata():
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
        seed=9223372036854775807,
        steps=8,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        settings={"family": "z_image_turbo", "steps": 8},
        warnings=[],
        adapter_version="0.1.0",
        batch={
            "count": 3,
            "seeds": [9223372036854775807, 0, 1],
            "seed_mode": "increment",
        },
    )

    parsed = json.loads(to_json(run_info))

    assert parsed["seed"] == 9223372036854775807
    assert parsed["batch"] == {
        "count": 3,
        "seeds": [9223372036854775807, 0, 1],
        "seed_mode": "increment",
    }


def test_run_info_reports_memory_policy_and_reference_dedupe():
    run_info = build_run_info(
        model_type="flux2_klein_9b",
        display_name="FLUX.2 Klein 9B",
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
        sampler="euler",
        scheduler="beta",
        settings={
            "family": "flux2_klein_9b",
            "memory_policy": "auto",
            "resolved_memory_policy": "auto",
            "memory_cleanup_applied": True,
            "memory_reserved_vram_gb": 0.6,
            "duplicate_inpaint_reference_skipped": True,
            "duplicate_inpaint_reference_count": 1,
        },
        warnings=[],
        adapter_version="0.1.0",
    )

    parsed = json.loads(to_json(run_info))

    assert parsed["performance"]["memory_policy"] == "auto"
    assert parsed["performance"]["resolved_memory_policy"] == "auto"
    assert parsed["performance"]["memory_cleanup_applied"] is True
    assert parsed["performance"]["memory_reserved_vram_gb"] == 0.6
    assert parsed["performance"]["duplicate_inpaint_reference_skipped"] is True
    assert parsed["performance"]["duplicate_inpaint_reference_count"] == 1


@pytest.mark.skipif(not privacy.CRYPTO_AVAILABLE, reason="cryptography is not installed")
def test_run_info_encrypts_prompt_override_when_private(monkeypatch, tmp_path):
    monkeypatch.setattr(privacy, "config_dir", lambda: tmp_path)
    initialize_keystore(PASSWORD)
    settings = {
        "family": "ideogram4",
        "steps": 20,
        "positive_prompt_source": "ideogram4_prompt_builder",
        "positive_prompt_override": '{"secret":"private prompt"}',
    }

    run_info = build_run_info(
        model_type="ideogram4",
        display_name="Ideogram 4",
        diffusion_model="ideogram4/ideogram4_fp8_scaled.safetensors",
        diffusion_model_format="safetensors",
        text_encoder="qwen3vl_8b_fp8_scaled.safetensors",
        text_encoder_format="safetensors",
        vae="flux.2/flux2-vae.safetensors",
        vae_format="safetensors",
        width=1024,
        height=1024,
        seed=123,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="ideogram4",
        settings=settings,
        warnings=[],
        adapter_version="0.1.0",
        privacy_mode=True,
    )

    dumped = to_json(run_info)
    parsed = json.loads(dumped)
    encrypted = parsed["settings"]["positive_prompt_override"]

    assert "private prompt" not in dumped
    assert privacy.is_encrypted_payload(encrypted)
    assert privacy.decrypt_text_if_encrypted(encrypted) == '{"secret":"private prompt"}'
    assert settings["positive_prompt_override"] == '{"secret":"private prompt"}'
