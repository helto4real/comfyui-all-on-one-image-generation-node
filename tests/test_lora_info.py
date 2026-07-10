import sys
from pathlib import Path
from types import SimpleNamespace

from services import lora_info


def _install_folder_paths(monkeypatch, root: Path):
    def get_full_path(category, file):
        assert category == "loras"
        candidate = root / file
        return str(candidate) if candidate.exists() else None

    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(
            get_full_path=get_full_path,
            get_folder_paths=lambda category: [str(root)] if category == "loras" else [],
        ),
    )


def test_lora_info_merges_civitai_data():
    info = {"images": [], "raw": {}}
    lora_info._merge_civitai(
        info,
        {
            "_sha256": "abc",
            "_civitai_api": "https://civitai.example/api",
            "id": 456,
            "modelId": 123,
            "name": "Version",
            "baseModel": "Flux.1",
            "model": {"name": "Model", "type": "LORA"},
            "trainedWords": ["word_a, word_b"],
            "images": [{"url": "https://image.example/999.jpeg", "meta": {"seed": 7}}],
        },
    )

    assert info["name"] == "Model - Version"
    assert "https://civitai.com/models/123?modelVersionId=456" in info["links"]
    assert info["trainedWords"] == [
        {"word": "word_a", "civitai": True},
        {"word": "word_b", "civitai": True},
    ]
    assert info["images"][0]["seed"] == 7


def test_lora_info_reads_safetensors_metadata(tmp_path):
    metadata = b'{"__metadata__":{"ss_output_name":"Demo","ss_clip_skip":"2"}}'
    path = tmp_path / "demo.safetensors"
    path.write_bytes(len(metadata).to_bytes(8, "little") + metadata)

    parsed = lora_info._read_safetensors_metadata(Path(path))

    assert parsed["ss_output_name"] == "Demo"
    assert parsed["ss_clip_skip"] == "2"


def test_lora_path_rejects_absolute_file_outside_configured_roots(monkeypatch, tmp_path):
    root = tmp_path / "loras"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    _install_folder_paths(monkeypatch, root)

    assert lora_info._lora_path(str(outside)) is None


def test_lora_info_accepts_configured_file_without_exposing_server_path(monkeypatch, tmp_path):
    root = tmp_path / "loras"
    root.mkdir()
    model = root / "demo.safetensors"
    model.write_bytes(b"not-a-real-safetensors-file")
    _install_folder_paths(monkeypatch, root)

    info = lora_info.get_lora_info("demo.safetensors", light=True)

    assert info is not None
    assert info["file"] == "demo.safetensors"
    assert "path" not in info


def test_normal_lora_info_read_neither_contacts_civitai_nor_writes_sidecar(monkeypatch, tmp_path):
    root = tmp_path / "loras"
    root.mkdir()
    (root / "demo.safetensors").write_bytes(b"demo")
    _install_folder_paths(monkeypatch, root)
    monkeypatch.setattr(lora_info, "_fetch_civitai", lambda _hash: (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setattr(lora_info, "_write_json", lambda *_args: (_ for _ in ()).throw(AssertionError("write")))

    info = lora_info.get_lora_info("demo.safetensors")

    assert info is not None
    assert info["sha256"]
    assert "civitai" not in info["raw"]


def test_explicit_civitai_refresh_fetches_and_persists(monkeypatch, tmp_path):
    root = tmp_path / "loras"
    root.mkdir()
    (root / "demo.safetensors").write_bytes(b"demo")
    _install_folder_paths(monkeypatch, root)
    writes = []
    monkeypatch.setattr(lora_info, "_fetch_civitai", lambda file_hash: {"_sha256": file_hash, "error": "offline"})
    monkeypatch.setattr(lora_info, "_write_json", lambda path, data: writes.append((path, data)))

    info = lora_info.get_lora_info("demo.safetensors", fetch_civitai=True, persist=True)

    assert info is not None
    assert info["raw"]["civitai"]["error"] == "offline"
    assert len(writes) == 1


def test_partial_save_allows_only_user_editable_fields(monkeypatch, tmp_path):
    root = tmp_path / "loras"
    root.mkdir()
    (root / "demo.safetensors").write_bytes(b"demo")
    _install_folder_paths(monkeypatch, root)

    info = lora_info.save_lora_info_partial(
        "demo.safetensors",
        {
            "name": "Friendly",
            "strengthMin": -0.5,
            "strengthMax": 1.5,
            "userNote": "Local note",
            "path": "/etc/passwd",
            "raw": {"injected": True},
            "images": [{"url": "javascript:alert(1)"}],
        },
    )

    assert info is not None
    assert info["name"] == "Friendly"
    assert info["strengthMin"] == -0.5
    assert info["strengthMax"] == 1.5
    assert info["userNote"] == "Local note"
    assert "path" not in info
    assert info["images"] == []
    assert info["raw"] == {}


def test_preview_image_rejects_symlink_that_escapes_lora_root(monkeypatch, tmp_path):
    root = tmp_path / "loras"
    root.mkdir()
    model = root / "demo.safetensors"
    model.write_bytes(b"demo")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    (root / "demo.png").symlink_to(outside)
    _install_folder_paths(monkeypatch, root)

    assert lora_info._preview_image(model) is None
