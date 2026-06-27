import pytest

from nodes.flux2_klein_settings import AIOFlux2Klein9BSettings
from nodes.ideogram4_settings import AIOIdeogram4Settings, DEFAULT_UNCONDITIONAL_MODEL
from nodes.krea2_settings import AIOKrea2Settings
from nodes.z_image_settings import AIOZImageTurboSettings
from services import privacy
import sys


def test_z_settings_returns_family_dict():
    settings = AIOZImageTurboSettings().build_settings(
        "default", 8, "off", True, "auto"
    )[0]

    assert settings["family"] == "z_image_turbo"
    assert settings["force_steps"] == 8
    assert settings["attention_mode"] == "auto"
    assert settings["torch_compile_mode"] == "off"
    assert settings["torch_compile_backend"] == "inductor"
    assert settings["performance_apply_timing"] == "after_loras"


def test_ideogram4_settings_can_disable_unconditional_model():
    settings = AIOIdeogram4Settings().build_settings(
        "Default",
        DEFAULT_UNCONDITIONAL_MODEL,
        7.0,
        True,
        3.0,
        0.7,
        1.0,
        5.0,
        "auto",
        run_unconditional_model=False,
    )[0]

    assert settings["run_unconditional_model"] is False


def test_flux_settings_returns_family_dict():
    settings = AIOFlux2Klein9BSettings().build_settings(
        "distilled", 1.0, "auto", "balanced", 0.5, 1.15
    )[0]

    assert settings["family"] == "flux2_klein_9b"
    assert settings["variant"] == "distilled"
    assert "edit_mode" not in AIOFlux2Klein9BSettings.INPUT_TYPES()["required"]
    assert "edit_mode" not in settings
    assert "output_size_mode" not in settings
    assert "reference_strength" not in AIOFlux2Klein9BSettings.INPUT_TYPES()["required"]
    assert "reference_strength" not in settings
    assert settings["reference_megapixels"] == 1.0
    assert settings["reference_upscale_method"] == "area"
    assert settings["reference_resolution_steps"] == 1
    assert settings["attention_mode"] == "auto"
    assert settings["torch_compile_mode"] == "off"
    assert settings["torch_compile_backend"] == "inductor"
    assert settings["performance_apply_timing"] == "after_loras"


def test_krea2_settings_returns_workflow_defaults():
    settings = AIOKrea2Settings().build_settings(
        True,
        1.0,
        "auto",
    )[0]

    assert settings["family"] == "krea2"
    assert settings["enhancer_enabled"] is True
    assert settings["enhancer_strength"] == 1.0
    assert settings["precision_policy"] == "auto"
    assert settings["attention_mode"] == "auto"
    assert settings["torch_compile_mode"] == "off"
    assert settings["torch_compile_backend"] == "inductor"
    assert settings["performance_apply_timing"] == "after_loras"
    assert settings["fp16_accumulation_enabled"] is True


def test_ideogram4_settings_returns_default_family_dict():
    settings = AIOIdeogram4Settings().build_settings(
        "Default",
        DEFAULT_UNCONDITIONAL_MODEL,
        7.0,
        True,
        3.0,
        0.7,
        1.0,
        5.0,
        "auto",
    )[0]

    assert settings["family"] == "ideogram4"
    assert settings["preset"] == "Default"
    assert settings["unconditional_model"] == DEFAULT_UNCONDITIONAL_MODEL
    assert settings["preset_steps"] == 20
    assert settings["mu"] == 0.0
    assert settings["std"] == 1.75
    assert settings["schedule_mode"] == "ideogram4"
    assert settings["scheduler"] == "ideogram4"
    assert settings["dual_cfg"] == 7.0
    assert settings["cfg_override_enabled"] is True
    assert settings["cfg_override"] == 3.0
    assert settings["cfg_override_start_percent"] == 0.7
    assert settings["cfg_override_end_percent"] == 1.0
    assert settings["sampling_shift"] == 5.0
    assert settings["precision_policy"] == "auto"
    assert settings["attention_mode"] == "auto"
    assert settings["torch_compile_mode"] == "off"
    assert settings["torch_compile_backend"] == "inductor"
    assert settings["performance_apply_timing"] == "after_loras"
    assert settings["run_unconditional_model"] is True


def test_ideogram4_settings_accepts_prompt_builder_payload():
    settings = AIOIdeogram4Settings().build_settings(
        "Default",
        DEFAULT_UNCONDITIONAL_MODEL,
        7.0,
        True,
        3.0,
        0.7,
        1.0,
        5.0,
        "auto",
        prompt_builder={
            "family": "ideogram4",
            "prompt": '{"compositional_deconstruction":{"background":"Room","elements":[]}}',
            "width": 1088,
            "height": 608,
            "max_side": 1088,
            "aspect_ratio": "16:9",
            "multiple_value": "16",
        },
    )[0]

    assert settings["positive_prompt_source"] == "ideogram4_prompt_builder"
    assert settings["positive_prompt_override"] == '{"compositional_deconstruction":{"background":"Room","elements":[]}}'
    assert settings["prompt_builder_width"] == 1088
    assert settings["prompt_builder_height"] == 608
    assert settings["prompt_builder_max_side"] == 1088
    assert settings["prompt_builder_aspect_ratio"] == "16:9"
    assert settings["prompt_builder_multiple_value"] == "16"


@pytest.mark.skipif(not privacy.CRYPTO_AVAILABLE, reason="cryptography is not installed")
def test_ideogram4_settings_encrypts_private_prompt_builder_override(monkeypatch, tmp_path):
    monkeypatch.setattr(privacy, "config_dir", lambda: tmp_path)
    prompt = '{"compositional_deconstruction":{"background":"Private room","elements":[]}}'

    settings = AIOIdeogram4Settings().build_settings(
        "Default",
        DEFAULT_UNCONDITIONAL_MODEL,
        7.0,
        True,
        3.0,
        0.7,
        1.0,
        5.0,
        "auto",
        prompt_builder={
            "family": "ideogram4",
            "prompt": prompt,
            "privacy_mode": True,
        },
    )[0]

    dumped = str(settings["positive_prompt_override"])
    assert "Private room" not in dumped
    assert privacy.is_encrypted_payload(settings["positive_prompt_override"])
    assert privacy.decrypt_text_if_encrypted(settings["positive_prompt_override"]) == prompt
    assert settings["prompt_builder_privacy_mode"] is True


def test_ideogram4_unconditional_model_choices_are_category_prefixed(monkeypatch):
    class FakeFolderPaths:
        @staticmethod
        def get_filename_list(category):
            if category == "diffusion_models":
                return ["ideogram4/ideogram4_unconditional_fp8_scaled.safetensors"]
            if category == "unet":
                return ["other_unet.safetensors"]
            return []

    monkeypatch.setitem(sys.modules, "folder_paths", FakeFolderPaths)
    inputs = AIOIdeogram4Settings.INPUT_TYPES()
    choices, config = inputs["required"]["unconditional_model"]

    assert config["default"] == DEFAULT_UNCONDITIONAL_MODEL
    assert DEFAULT_UNCONDITIONAL_MODEL in choices
    assert "unet/other_unet.safetensors" in choices
    assert "ideogram4/ideogram4_unconditional_fp8_scaled.safetensors" not in choices


def test_ideogram4_settings_presets():
    quality = AIOIdeogram4Settings().build_settings(
        "Quality", DEFAULT_UNCONDITIONAL_MODEL, 7.0, True, 3.0, 0.7, 1.0, 5.0, "auto"
    )[0]
    turbo = AIOIdeogram4Settings().build_settings(
        "Turbo", DEFAULT_UNCONDITIONAL_MODEL, 7.0, True, 3.0, 0.7, 1.0, 5.0, "auto"
    )[0]
    workflow = AIOIdeogram4Settings().build_settings(
        "Workflow Compatible", DEFAULT_UNCONDITIONAL_MODEL, 7.0, True, 3.0, 0.7, 1.0, 5.0, "auto"
    )[0]

    assert quality["preset_steps"] == 48
    assert quality["mu"] == 0.0
    assert quality["std"] == 1.5
    assert quality["schedule_mode"] == "ideogram4"
    assert turbo["preset_steps"] == 12
    assert turbo["mu"] == 0.5
    assert turbo["std"] == 1.75
    assert workflow["preset_steps"] == 28
    assert workflow["schedule_mode"] == "basic"
    assert workflow["scheduler"] == "simple"
    assert workflow["cfg_override_start_percent"] == 0.9


def test_flux_settings_can_override_reference_scaling():
    settings = AIOFlux2Klein9BSettings().build_settings(
        "distilled",
        1.0,
        "auto",
        "balanced",
        0.5,
        1.15,
        2.0,
        "lanczos",
        8,
    )[0]

    assert settings["reference_megapixels"] == 2.0
    assert settings["reference_upscale_method"] == "lanczos"
    assert settings["reference_resolution_steps"] == 8
    assert settings["attention_mode"] == "auto"
    assert settings["torch_compile_mode"] == "off"
    assert settings["torch_compile_backend"] == "inductor"
    assert settings["performance_apply_timing"] == "after_loras"


def test_settings_can_override_performance_controls():
    z_settings = AIOZImageTurboSettings().build_settings(
        "default",
        8,
        "off",
        True,
        "auto",
        "sage",
        "on",
        "cudagraphs",
        "before_loras",
    )[0]
    flux_settings = AIOFlux2Klein9BSettings().build_settings(
        "distilled",
        1.0,
        "auto",
        "balanced",
        0.5,
        1.15,
        1.0,
        "area",
        1,
        "pytorch",
        "auto",
        "inductor",
        "before_loras",
    )[0]
    krea_settings = AIOKrea2Settings().build_settings(
        False,
        0.35,
        "bf16",
        "pytorch",
        "auto",
        "cudagraphs",
        "before_loras",
        False,
    )[0]

    assert z_settings["attention_mode"] == "sage"
    assert z_settings["torch_compile_mode"] == "on"
    assert z_settings["torch_compile_backend"] == "cudagraphs"
    assert z_settings["performance_apply_timing"] == "before_loras"
    assert flux_settings["attention_mode"] == "pytorch"
    assert flux_settings["torch_compile_mode"] == "auto"
    assert flux_settings["torch_compile_backend"] == "inductor"
    assert flux_settings["performance_apply_timing"] == "before_loras"
    assert krea_settings["enhancer_enabled"] is False
    assert krea_settings["enhancer_strength"] == 0.35
    assert krea_settings["precision_policy"] == "bf16"
    assert krea_settings["attention_mode"] == "pytorch"
    assert krea_settings["torch_compile_mode"] == "auto"
    assert krea_settings["torch_compile_backend"] == "cudagraphs"
    assert krea_settings["performance_apply_timing"] == "before_loras"
    assert krea_settings["fp16_accumulation_enabled"] is False


def test_flux_settings_ignores_legacy_reference_strength_argument():
    settings = AIOFlux2Klein9BSettings().build_settings(
        "distilled",
        1.0,
        0.75,
        "auto",
        "balanced",
        0.5,
        1.15,
        2.0,
        "lanczos",
        8,
    )[0]

    assert "reference_strength" not in settings
    assert settings["precision_policy"] == "auto"
    assert settings["memory_policy"] == "balanced"
    assert settings["base_shift"] == 0.5
    assert settings["max_shift"] == 1.15
    assert settings["reference_megapixels"] == 2.0
    assert settings["reference_upscale_method"] == "lanczos"
    assert settings["reference_resolution_steps"] == 8
    assert settings["attention_mode"] == "auto"
    assert settings["torch_compile_mode"] == "off"
    assert settings["torch_compile_backend"] == "inductor"
    assert settings["performance_apply_timing"] == "after_loras"


def test_flux_settings_ignores_legacy_edit_mode_argument():
    settings = AIOFlux2Klein9BSettings().build_settings(
        "distilled",
        1.0,
        "single_reference",
        0.75,
        "auto",
        "balanced",
        0.5,
        1.15,
        2.0,
        "lanczos",
        8,
    )[0]

    assert "edit_mode" not in settings
    assert "reference_strength" not in settings
    assert settings["precision_policy"] == "auto"
    assert settings["memory_policy"] == "balanced"
    assert settings["base_shift"] == 0.5
    assert settings["max_shift"] == 1.15
    assert settings["reference_megapixels"] == 2.0
    assert settings["reference_upscale_method"] == "lanczos"
    assert settings["reference_resolution_steps"] == 8
