from nodes.flux2_klein_settings import AIOFlux2Klein9BSettings
from nodes.z_image_settings import AIOZImageTurboSettings


def test_z_settings_returns_family_dict():
    settings = AIOZImageTurboSettings().build_settings(
        "default", 8, "off", True, "auto"
    )[0]

    assert settings["family"] == "z_image_turbo"
    assert settings["force_steps"] == 8


def test_flux_settings_returns_family_dict():
    settings = AIOFlux2Klein9BSettings().build_settings(
        "distilled", 1.0, 0.75, "auto", "balanced", 0.5, 1.15
    )[0]

    assert settings["family"] == "flux2_klein_9b"
    assert settings["variant"] == "distilled"
    assert "edit_mode" not in AIOFlux2Klein9BSettings.INPUT_TYPES()["required"]
    assert "edit_mode" not in settings
    assert "output_size_mode" not in settings
    assert settings["reference_megapixels"] == 1.0
    assert settings["reference_upscale_method"] == "area"
    assert settings["reference_resolution_steps"] == 1


def test_flux_settings_can_override_reference_scaling():
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

    assert settings["reference_megapixels"] == 2.0
    assert settings["reference_upscale_method"] == "lanczos"
    assert settings["reference_resolution_steps"] == 8


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
    assert settings["reference_strength"] == 0.75
    assert settings["precision_policy"] == "auto"
    assert settings["memory_policy"] == "balanced"
    assert settings["base_shift"] == 0.5
    assert settings["max_shift"] == 1.15
    assert settings["reference_megapixels"] == 2.0
    assert settings["reference_upscale_method"] == "lanczos"
    assert settings["reference_resolution_steps"] == 8
