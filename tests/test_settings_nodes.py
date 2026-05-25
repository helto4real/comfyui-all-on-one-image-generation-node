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
        "distilled", 1.0, "text_to_image", 0.75, "auto", "balanced", 0.5, 1.15
    )[0]

    assert settings["family"] == "flux2_klein_9b"
    assert settings["variant"] == "distilled"
    assert "output_size_mode" not in settings
