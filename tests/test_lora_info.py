from pathlib import Path

from services import lora_info


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
