from pathlib import Path
import sys
from types import SimpleNamespace

from loaders import gguf_backend


def test_gguf_backend_reports_missing_when_no_backend(monkeypatch):
    monkeypatch.setattr(gguf_backend.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(gguf_backend, "_custom_node_dirs", lambda: [])

    assert gguf_backend.is_available() is False
    assert "ComfyUI-GGUF" in gguf_backend.explain_missing()


def test_gguf_backend_detects_custom_node_dir(monkeypatch, tmp_path):
    custom_nodes = tmp_path / "custom_nodes"
    (custom_nodes / "ComfyUI-GGUF").mkdir(parents=True)
    monkeypatch.setattr(gguf_backend.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(gguf_backend, "_custom_node_dirs", lambda: [Path(custom_nodes)])

    assert gguf_backend.is_available() is True


def test_gguf_backend_resolves_gguf_text_encoder_category(monkeypatch):
    calls = []

    def get_full_path_or_raise(category, name):
        calls.append((category, name))
        if category == "unet_gguf" and name == "model.gguf":
            return "/models/diffusion_models/model.gguf"
        if category == "clip_gguf" and name == "encoder.gguf":
            return "/models/text_encoders/encoder.gguf"
        if category == "vae_gguf" and name == "vae.gguf":
            return "/models/vae/vae.gguf"
        raise RuntimeError("not found")

    fake_folder_paths = SimpleNamespace(get_full_path_or_raise=get_full_path_or_raise)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)

    paths = gguf_backend.resolve_paths("model.gguf", "encoder.gguf", "vae.gguf")

    assert str(paths.text_encoder) == "/models/text_encoders/encoder.gguf"
    assert ("clip_gguf", "encoder.gguf") in calls
