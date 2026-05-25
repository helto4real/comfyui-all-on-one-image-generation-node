import pytest

import adapters  # noqa: F401
from services.registry import get_adapter, get_profile, list_model_types


def test_known_model_types_resolve():
    assert list_model_types() == ["flux2_klein_9b", "z_image_turbo"]
    assert get_profile("z_image_turbo").display_name == "Z-Image Turbo"
    assert get_adapter("flux2_klein_9b").model_type == "flux2_klein_9b"


def test_unknown_model_type_raises_useful_error():
    with pytest.raises(ValueError, match="Unsupported model_type 'unknown'"):
        get_adapter("unknown")
