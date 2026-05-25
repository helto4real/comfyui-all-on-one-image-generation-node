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
    )

    assert (dimensions.width, dimensions.height) == (688, 1024)


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
    )

    assert (dimensions.width, dimensions.height) == (512, 768)
    assert dimensions.max_side == 768
    assert dimensions.size_mode == "use image 1 size"


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
