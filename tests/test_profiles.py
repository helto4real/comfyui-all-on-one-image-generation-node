from services.profiles import get_profile, list_profiles


def test_profile_defaults():
    z_profile = get_profile("z_image_turbo")
    flux_profile = get_profile("flux2_klein_9b")
    ideogram_profile = get_profile("ideogram4")

    assert z_profile.default_steps == 8
    assert flux_profile.default_steps == 4
    assert ideogram_profile.default_steps == 20
    assert ideogram_profile.default_cfg == 7.0
    assert ideogram_profile.supports_inpaint is True
    assert z_profile.supports_inpaint is False
    assert flux_profile.supports_inpaint is False
    assert "diffusion_model" in z_profile.required_components
    assert "text_encoder" in flux_profile.required_components
    assert "vae" in flux_profile.required_components


def test_list_profiles_contains_initial_families():
    keys = {profile.key for profile in list_profiles()}

    assert keys == {"z_image_turbo", "flux2_klein_9b", "ideogram4"}
