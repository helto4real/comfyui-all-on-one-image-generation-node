import sys
from types import SimpleNamespace

from services.lora_application import apply_lora_config


def test_lora_application_uses_comfy_lora_loader_in_order(monkeypatch):
    calls = []

    class FakeLoraLoader:
        def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
            calls.append((model, clip, lora_name, strength_model, strength_clip))
            return f"{model}+{lora_name}", f"{clip}+{lora_name}"

    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(LoraLoader=FakeLoraLoader))

    model, clip, applied = apply_lora_config(
        model="model",
        clip="clip",
        lora_config={
            "loras": [
                {
                    "enabled": True,
                    "name": "a.safetensors",
                    "strength_model": 1.0,
                    "strength_clip": 0.5,
                },
                {
                    "enabled": True,
                    "name": "b.safetensors",
                    "strength_model": 0.7,
                    "strength_clip": 0.7,
                },
            ]
        },
    )

    assert model == "model+a.safetensors+b.safetensors"
    assert clip == "clip+a.safetensors+b.safetensors"
    assert [row["name"] for row in applied] == ["a.safetensors", "b.safetensors"]
    assert calls == [
        ("model", "clip", "a.safetensors", 1.0, 0.5),
        ("model+a.safetensors", "clip+a.safetensors", "b.safetensors", 0.7, 0.7),
    ]
