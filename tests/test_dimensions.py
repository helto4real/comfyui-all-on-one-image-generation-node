from services.dimensions import infer_nearest_aspect_ratio, resolve_dimensions_from_controls


class FakeImage:
    shape = (1, 768, 512, 3)


class FakeReferenceInputs:
    images = (FakeImage(),)


def test_dimensions_resolve_square_aspect_ratio():
    dimensions = resolve_dimensions_from_controls(
        max_side=1024,
        aspect_ratio="1:1",
    )

    assert (dimensions.width, dimensions.height) == (1024, 1024)


def test_dimensions_resolve_landscape_aspect_ratio():
    dimensions = resolve_dimensions_from_controls(
        max_side=1024,
        aspect_ratio="16:9",
    )

    assert (dimensions.width, dimensions.height) == (1024, 576)


def test_dimensions_resolve_portrait_aspect_ratio():
    dimensions = resolve_dimensions_from_controls(
        max_side=1024,
        aspect_ratio="9:16",
    )

    assert (dimensions.width, dimensions.height) == (576, 1024)


def test_dimensions_round_to_adapter_multiple():
    dimensions = resolve_dimensions_from_controls(
        max_side=1024,
        aspect_ratio="2:3",
        multiple_value="16",
    )

    assert (dimensions.width, dimensions.height) == (688, 1024)


def test_dimensions_use_nearest_integer_by_default():
    dimensions = resolve_dimensions_from_controls(
        max_side=1024,
        aspect_ratio="2:3",
    )

    assert (dimensions.width, dimensions.height) == (683, 1024)
    assert dimensions.multiple_value == "none"


def test_dimensions_none_multiple_allows_any_max_side_in_range():
    dimensions = resolve_dimensions_from_controls(
        max_side=1025,
        aspect_ratio="1:1",
    )

    assert (dimensions.width, dimensions.height) == (1025, 1025)
    assert dimensions.max_side == 1025


def test_dimensions_can_round_to_multiple_of_8():
    dimensions = resolve_dimensions_from_controls(
        max_side=1024,
        aspect_ratio="2:3",
        multiple_value="8",
    )

    assert (dimensions.width, dimensions.height) == (680, 1024)


def test_dimensions_can_round_to_multiple_of_32():
    dimensions = resolve_dimensions_from_controls(
        max_side=1024,
        aspect_ratio="16:9",
        multiple_value="32",
    )

    assert (dimensions.width, dimensions.height) == (1024, 576)


def test_dimensions_rejects_max_side_that_does_not_match_multiple_value():
    try:
        resolve_dimensions_from_controls(
            max_side=1025,
            aspect_ratio="1:1",
            multiple_value="16",
        )
    except ValueError as error:
        assert str(error) == "max side must be a multiple of 16 when multiple value is 16."
    else:
        raise AssertionError("Expected ValueError.")


def test_dimensions_rejects_max_side_outside_supported_range():
    try:
        resolve_dimensions_from_controls(
            max_side=4097,
            aspect_ratio="1:1",
        )
    except ValueError as error:
        assert str(error) == "max side must be between 256 and 4096."
    else:
        raise AssertionError("Expected ValueError.")


def test_dimensions_infer_legacy_aspect_ratio():
    dimensions = resolve_dimensions_from_controls(
        max_side=None,
        aspect_ratio=None,
        legacy_width=1024,
        legacy_height=768,
    )

    assert dimensions.max_side == 1024
    assert dimensions.aspect_ratio == "4:3"
    assert (dimensions.width, dimensions.height) == (1024, 768)


def test_dimensions_can_use_image_1_size():
    dimensions = resolve_dimensions_from_controls(
        size_mode="use image 1 size",
        max_side=1024,
        aspect_ratio="1:1",
        reference_inputs=FakeReferenceInputs(),
        multiple_value="16",
    )

    assert (dimensions.width, dimensions.height) == (512, 768)
    assert dimensions.max_side == 768
    assert dimensions.size_mode == "use image 1 size"


def test_dimensions_image_1_size_can_use_exact_dimensions():
    dimensions = resolve_dimensions_from_controls(
        size_mode="use image 1 size",
        max_side=1024,
        aspect_ratio="1:1",
        reference_inputs=FakeReferenceInputs(),
    )

    assert (dimensions.width, dimensions.height) == (512, 768)
    assert dimensions.multiple_value == "none"


def test_dimensions_require_image_1_for_image_1_size_mode():
    try:
        resolve_dimensions_from_controls(
            size_mode="use image 1 size",
            max_side=1024,
            aspect_ratio="1:1",
        )
    except ValueError as error:
        assert str(error) == "size mode 'use image 1 size' requires image 1."
    else:
        raise AssertionError("Expected ValueError.")


def test_infer_nearest_aspect_ratio_defaults_for_invalid_dimensions():
    assert infer_nearest_aspect_ratio(0, 1024) == "1:1"
