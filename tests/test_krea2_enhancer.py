import sys
import types

from services import krea2_enhancer


class FakeModel:
    def __init__(self, name="model"):
        self.name = name
        self.model_options = {"transformer_options": {}}
        self.added_wrappers = []
        self.clone_result = None

    def clone(self):
        self.clone_result = FakeModel("clone")
        return self.clone_result

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.added_wrappers.append((wrapper_type, key, wrapper))


def install_fake_patcher(monkeypatch):
    comfy_module = types.ModuleType("comfy")
    patcher_module = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"

    def add_wrapper_with_key(wrapper_type, key, wrapper, transformer_options, is_model_options=False):
        if is_model_options:
            transformer_options = transformer_options.setdefault("transformer_options", {})
        wrappers = transformer_options.setdefault("wrappers", {})
        wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    patcher_module.WrappersMP = WrappersMP
    patcher_module.add_wrapper_with_key = add_wrapper_with_key
    comfy_module.patcher_extension = patcher_module
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", patcher_module)
    return patcher_module


def test_disabled_and_zero_strength_return_original_model():
    model = FakeModel()

    assert krea2_enhancer.apply_krea2_enhancer(model, enabled=False, strength=1.0) is model
    assert krea2_enhancer.apply_krea2_enhancer(model, enabled=True, strength=0.0) is model
    assert model.clone_result is None


def test_active_detection_clamps_strength():
    assert krea2_enhancer.normalize_strength(2.5) == 1.0
    assert krea2_enhancer.normalize_strength(-1.0) == 0.0
    assert krea2_enhancer.is_enhancer_active(enabled=True, strength=0.01) is True
    assert krea2_enhancer.is_enhancer_active(enabled=False, strength=1.0) is False
    assert krea2_enhancer.is_enhancer_active(enabled=True, strength=0.0) is False


def test_conditioning_enhancer_scales_krea_chunks_and_copies_metadata():
    import torch

    tensor = torch.ones((1, 2, krea2_enhancer.KREA2_CHUNK_COUNT * krea2_enhancer.KREA2_CHUNK_DIM))
    pooled = torch.tensor([[1.0]])
    metadata = {"pooled_output": pooled, "tag": "original"}
    conditioning = [[tensor, metadata]]

    enhanced = krea2_enhancer.enhance_krea2_conditioning(
        conditioning,
        enabled=True,
        strength=2.5,
    )

    assert enhanced is not conditioning
    assert enhanced[0] is not conditioning[0]
    assert enhanced[0][1] is not metadata
    assert enhanced[0][1]["tag"] == "original"
    assert enhanced[0][1]["pooled_output"] is pooled
    assert torch.allclose(conditioning[0][0], torch.ones_like(tensor))

    chunks = enhanced[0][0].reshape(1, 2, krea2_enhancer.KREA2_CHUNK_COUNT, krea2_enhancer.KREA2_CHUNK_DIM)
    assert torch.allclose(chunks[:, :, 0], torch.full_like(chunks[:, :, 0], 15.0))
    assert torch.allclose(chunks[:, :, 7], torch.full_like(chunks[:, :, 7], 37.5))
    assert torch.allclose(chunks[:, :, 8], torch.full_like(chunks[:, :, 8], 75.0))


def test_conditioning_enhancer_returns_original_when_inactive():
    conditioning = [["conditioning", {}]]

    assert krea2_enhancer.enhance_krea2_conditioning(conditioning, enabled=False, strength=1.0) is conditioning
    assert krea2_enhancer.enhance_krea2_conditioning(conditioning, enabled=True, strength=0.0) is conditioning


def test_active_enhancer_clones_model_clamps_strength_and_registers_wrappers(monkeypatch):
    patcher_module = install_fake_patcher(monkeypatch)
    model = FakeModel()

    patched = krea2_enhancer.apply_krea2_enhancer(model, enabled=True, strength=2.5)

    assert patched is model.clone_result
    assert patched is not model
    config = patched.model_options["transformer_options"][krea2_enhancer.WRAPPER_KEY]
    assert config == {
        "enabled": True,
        "strength": 1.0,
        "debug": False,
        "max_debug_prints": 8,
    }
    assert patched.added_wrappers == [
        (
            patcher_module.WrappersMP.DIFFUSION_MODEL,
            krea2_enhancer.WRAPPER_KEY,
            krea2_enhancer.krea2t_enhancer_wrapper,
        )
    ]
    wrappers = patched.model_options["transformer_options"]["wrappers"]
    assert wrappers[patcher_module.WrappersMP.DIFFUSION_MODEL][krea2_enhancer.WRAPPER_KEY] == [
        krea2_enhancer.krea2t_enhancer_wrapper
    ]


def test_wrapper_skips_non_krea2_diffusion_model():
    calls = {}

    class FakeExecutor:
        class_obj = object()

        def __call__(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return "base"

    transformer_options = {
        krea2_enhancer.WRAPPER_KEY: {
            "enabled": True,
            "strength": 1.0,
        }
    }

    result = krea2_enhancer.krea2t_enhancer_wrapper(
        FakeExecutor(),
        "x",
        "timesteps",
        "context",
        transformer_options=transformer_options,
    )

    assert result == "base"
    assert calls["args"] == ("x", "timesteps", "context", None, transformer_options)
    assert calls["kwargs"] == {}


def test_service_import_does_not_load_torch_or_comfy_at_module_scope():
    assert "torch" not in vars(krea2_enhancer)
    assert "comfy" not in vars(krea2_enhancer)
