import pytest

from services import krea2_rebalance


@pytest.fixture
def torch_module():
    return pytest.importorskip("torch")


def test_parse_per_layer_weights_accepts_commas_and_semicolons():
    assert krea2_rebalance.parse_per_layer_weights("1; 2,3") == [1.0, 2.0, 3.0]


def test_parse_per_layer_weights_rejects_invalid_or_single_value():
    assert krea2_rebalance.parse_per_layer_weights("bad") is None
    assert krea2_rebalance.parse_per_layer_weights("1.0") is None


def test_rebalance_conditioning_scales_per_layer_and_preserves_extras(torch_module):
    torch = torch_module
    tensor = torch.arange(1, 9, dtype=torch.float16).reshape(1, 1, 8)
    extras = {"pooled_output": torch.ones(1, 2), "mask": "keep"}

    out = krea2_rebalance.rebalance_conditioning(
        [[tensor, extras]],
        multiplier=2.0,
        per_layer_weights="1.0,3.0",
    )

    assert out[0][0].dtype == torch.float16
    expected = torch.tensor([2, 4, 6, 8, 30, 36, 42, 48], dtype=torch.float16).reshape(1, 1, 8)
    assert torch.equal(out[0][0], expected)
    assert torch.equal(out[0][1]["pooled_output"], extras["pooled_output"])
    assert out[0][1]["mask"] == "keep"
    assert out[0][1] is not extras


def test_rebalance_conditioning_falls_back_to_multiplier_for_invalid_weights(torch_module):
    torch = torch_module
    tensor = torch.ones(1, 1, 4)

    out = krea2_rebalance.rebalance_conditioning(
        {"nested": [[tensor, {"x": 1}]]},
        multiplier=3.0,
        per_layer_weights="bad",
    )

    assert torch.equal(out["nested"][0][0], tensor * 3.0)
