import sys
from types import ModuleType
from types import SimpleNamespace

from services import performance


class FakeModel:
    def __init__(self, *, load_device="cuda", label="model", clones=None):
        self.load_device = SimpleNamespace(type=load_device)
        self.label = label
        self.model_options = {"transformer_options": {}}
        self.clones = clones if clones is not None else []

    def clone(self, disable_dynamic=False):
        cloned = FakeModel(load_device=self.load_device.type, label=f"{self.label}+clone", clones=self.clones)
        cloned.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
        self.clones.append(disable_dynamic)
        return cloned


def install_fake_attention(monkeypatch, available):
    fake_comfy = ModuleType("comfy")
    fake_ldm = ModuleType("comfy.ldm")
    fake_modules = ModuleType("comfy.ldm.modules")
    fake_attention = ModuleType("comfy.ldm.modules.attention")

    def get_attention_function(name, default=None):
        return available.get(name, default)

    fake_attention.get_attention_function = get_attention_function
    fake_modules.attention = fake_attention
    fake_ldm.modules = fake_modules
    fake_comfy.ldm = fake_ldm
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.ldm", fake_ldm)
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules", fake_modules)
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules.attention", fake_attention)


def test_auto_attention_selects_best_installed_unmasked(monkeypatch):
    funcs = {
        "sage3": lambda *args, **kwargs: "sage3",
        "pytorch": lambda *args, **kwargs: "pytorch",
    }
    install_fake_attention(monkeypatch, funcs)
    settings = {"attention_mode": "auto", "torch_compile_mode": "off"}
    model = FakeModel()

    patched = performance.apply_performance_settings(model=model, settings=settings)

    assert patched is not model
    assert settings["resolved_attention_mode"] == "sage3"
    assert patched.model_options["transformer_options"]["optimized_attention_override"](None) == "sage3"


def test_auto_attention_uses_reference_safe_priority(monkeypatch):
    funcs = {
        "sage3": lambda *args, **kwargs: "sage3",
        "flash": lambda *args, **kwargs: "flash",
        "sage": lambda *args, **kwargs: "sage",
    }
    install_fake_attention(monkeypatch, funcs)
    settings = {"attention_mode": "auto", "torch_compile_mode": "off"}
    model = FakeModel()

    performance.apply_performance_settings(
        model=model,
        settings=settings,
        has_mask_or_reference=True,
    )

    assert settings["resolved_attention_mode"] == "sage"


def test_forced_reference_incompatible_attention_falls_back(monkeypatch):
    install_fake_attention(monkeypatch, {"flash": lambda *args, **kwargs: "flash"})
    settings = {"attention_mode": "flash", "torch_compile_mode": "off"}
    model = FakeModel()

    patched = performance.apply_performance_settings(
        model=model,
        settings=settings,
        has_mask_or_reference=True,
    )

    assert patched is model
    assert settings["resolved_attention_mode"] == "off"
    assert "performance_warnings" in settings


def test_attention_reports_off_when_model_options_unavailable(monkeypatch):
    install_fake_attention(monkeypatch, {"pytorch": lambda *args, **kwargs: "pytorch"})
    settings = {"attention_mode": "pytorch", "torch_compile_mode": "off"}

    performance.apply_performance_settings(model="not-a-model-patcher", settings=settings)

    assert settings["resolved_attention_mode"] == "off"
    assert "performance_warnings" in settings


def test_compile_auto_stays_off_without_cuda(monkeypatch):
    fake_torch = SimpleNamespace(compile=lambda model, **kwargs: model, cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    settings = {"torch_compile_mode": "auto", "torch_compile_backend": "inductor"}
    model = FakeModel()

    patched = performance.apply_performance_settings(model=model, settings=settings)

    assert patched is model
    assert settings["resolved_torch_compile_mode"] == "off"
    assert settings["resolved_torch_compile_backend"] == "off"


def test_compile_on_uses_comfy_compile_wrapper(monkeypatch):
    calls = {}
    fake_torch = SimpleNamespace(compile=lambda model, **kwargs: model, cuda=SimpleNamespace(is_available=lambda: False))
    fake_comfy_api = ModuleType("comfy_api")
    fake_helpers = ModuleType("comfy_api.torch_helpers")

    def set_torch_compile_wrapper(**kwargs):
        calls.update(kwargs)

    fake_helpers.set_torch_compile_wrapper = set_torch_compile_wrapper
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "comfy_api", fake_comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.torch_helpers", fake_helpers)
    settings = {"torch_compile_mode": "on", "torch_compile_backend": "cudagraphs"}
    model = FakeModel()

    patched = performance.apply_performance_settings(model=model, settings=settings)

    assert patched is not model
    assert model.clones == [True]
    assert calls["model"] is patched
    assert calls["backend"] == "cudagraphs"
    assert settings["resolved_torch_compile_mode"] == "on"
    assert settings["resolved_torch_compile_backend"] == "cudagraphs"
