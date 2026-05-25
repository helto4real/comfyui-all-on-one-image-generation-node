import pytest

from services.reference_inputs import normalize_reference_inputs


def test_reference_inputs_are_collected_in_socket_order():
    references = normalize_reference_inputs(
        {"image 1": "first", "image 2": "second", "image 4": None},
        mask="mask",
    )

    assert references.images == ("first", "second")
    assert references.mask == "mask"


def test_reference_inputs_reject_gaps():
    with pytest.raises(ValueError, match="image 2 was connected, but image 1 is empty"):
        normalize_reference_inputs({"image 2": "second"})


def test_reference_inputs_reject_mask_without_image_1():
    with pytest.raises(ValueError, match="mask can only be used"):
        normalize_reference_inputs(mask="mask")
