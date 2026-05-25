import pytest

from services.profiles import get_profile
from services.validation import (
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
