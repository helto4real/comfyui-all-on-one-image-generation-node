import sys
from types import SimpleNamespace

from loaders import safetensors_backend


def test_safetensors_backend_resolves_category_prefixed_subfolder(monkeypatch):
    calls = []

    def get_full_path_or_raise(category, name):
        calls.append((category, name))
        if category == "diffusion_models" and name == "krea/model.safetensors":
            return "/models/diffusion_models/krea/model.safetensors"
        raise RuntimeError("not found")

    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_full_path_or_raise=get_full_path_or_raise))

    path = safetensors_backend.diffusion_model_path("diffusion_models/krea/model.safetensors")

    assert str(path) == "/models/diffusion_models/krea/model.safetensors"
    assert calls == [("diffusion_models", "krea/model.safetensors")]


def test_safetensors_backend_treats_unknown_prefix_as_subfolder(monkeypatch):
    calls = []

    def get_full_path_or_raise(category, name):
        calls.append((category, name))
        if category == "diffusion_models" and name == "krea/model.safetensors":
            return "/models/diffusion_models/krea/model.safetensors"
        raise RuntimeError("not found")

    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_full_path_or_raise=get_full_path_or_raise))

    path = safetensors_backend.diffusion_model_path("krea/model.safetensors")

    assert str(path) == "/models/diffusion_models/krea/model.safetensors"
    assert calls[0] == ("diffusion_models", "krea/model.safetensors")
