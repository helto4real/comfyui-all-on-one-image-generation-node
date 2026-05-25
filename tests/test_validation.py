import pytest

from services.profiles import get_profile
from services.validation import (
    dimension_multiple_from_settings,
    validate_dimensions,
    validate_gguf_available_for_models,
    validate_reference_inputs,
    validate_settings_family,
)


def test_mismatched_settings_family_rejected():
    with pytest.raises(
        ValueError,
        match="Selected settings are for flux2_klein_9b, but model_type is z_image_turbo.",
    ):
        validate_settings_family("z_image_turbo", {"family": "flux2_klein_9b"})


def test_gguf_file_without_backend_rejected():
    with pytest.raises(ValueError, match="A GGUF model file was selected"):
        validate_gguf_available_for_models(False, "text.gguf")


def test_reference_image_rejected_for_unsupported_adapter_mode():
    profile = get_profile("z_image_turbo")

    with pytest.raises(ValueError, match="reference_image was connected"):
        validate_reference_inputs(profile, reference_image=object())


def test_invalid_dimensions_rejected():
    with pytest.raises(ValueError, match="multiples of 16"):
        validate_dimensions(1025, 1024, 16)


def test_dimensions_allow_exact_values_when_multiple_is_none():
    validate_dimensions(1025, 1024, None)


def test_dimension_multiple_comes_from_settings():
    assert dimension_multiple_from_settings({"multiple_value": "none"}, 16) is None
    assert dimension_multiple_from_settings({"multiple_value": "8"}, 16) == 8
    assert dimension_multiple_from_settings({}, 16) == 16
