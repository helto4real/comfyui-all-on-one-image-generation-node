import json

import pytest
from helto_privacy import initialize_keystore
from helto_privacy import lock_keystore

from services import privacy
from services import ideogram4_prompt_library as library

PASSWORD = "correct horse battery"


def sample_payload(prompt='{"compositional_deconstruction":{"background":"studio","elements":[]}}'):
    return {
        "family": "ideogram4",
        "version": 1,
        "state": {
            "version": 1,
            "widgets": {
                "max side": 1024,
                "aspect ratio": "1:1",
                "privacy_mode": False,
                "background": "studio",
                "output_format": "compact",
            },
            "elements": [],
            "style_palette": [],
            "output_format": "compact",
            "bg_brightness": 25,
            "active": -1,
        },
        "prompt": prompt,
    }


def test_ideogram_prompt_library_loads_empty(tmp_path):
    assert library.load_library(tmp_path) == {
        "schema_version": library.LIBRARY_SCHEMA_VERSION,
        "version": library.LIBRARY_VERSION,
        "prompts": [],
    }
    assert library.list_items(tmp_path)["prompts"] == []


def test_ideogram_prompt_library_create_list_use_public(tmp_path):
    payload = sample_payload()

    created = library.create_prompt(
        payload,
        metadata={"id": "public-prompt", "name": "Studio", "description": "public desc", "tags": ["set"]},
        base_dir=tmp_path,
    )
    listed = library.list_items(tmp_path)["prompts"][0]
    used = library.use_prompt("public-prompt", base_dir=tmp_path)

    assert created["id"] == "public-prompt"
    assert created["payload"] == payload
    assert listed["description"] == "public desc"
    assert listed["prompt_preview"].startswith('{"compositional_deconstruction"')
    assert listed["summary"]["element_count"] == 0
    assert used["payload"] == payload
    assert library.load_library(tmp_path)["prompts"][0]["last_used_at"]


def test_ideogram_prompt_library_replace_patch_duplicate_delete(tmp_path):
    library.create_prompt(sample_payload(), metadata={"id": "prompt-a", "name": "A"}, base_dir=tmp_path)
    replacement = sample_payload('{"new":"prompt"}')

    replaced = library.replace_prompt(
        "prompt-a",
        replacement,
        metadata={"name": "B", "description": "updated", "tags": ["x"]},
        base_dir=tmp_path,
    )
    patched = library.patch_prompt("prompt-a", metadata={"name": "C"}, base_dir=tmp_path)
    duplicate = library.duplicate_prompt("prompt-a", metadata={"id": "prompt-copy"}, base_dir=tmp_path)
    deleted = library.delete_prompt("prompt-copy", base_dir=tmp_path)

    assert replaced["name"] == "B"
    assert replaced["payload"] == replacement
    assert patched["name"] == "C"
    assert patched["description"] == "updated"
    assert duplicate["id"] == "prompt-copy"
    assert duplicate["payload"] == replacement
    assert deleted == {"id": "prompt-copy", "kind": "prompt"}
    assert [item["id"] for item in library.list_items(tmp_path)["prompts"]] == ["prompt-a"]


@pytest.mark.skipif(not privacy.CRYPTO_AVAILABLE, reason="cryptography is not installed")
def test_private_ideogram_prompt_encrypts_without_cleartext_leak(tmp_path):
    initialize_keystore(PASSWORD)
    payload = sample_payload('{"secret":"private room"}')
    payload["state"]["widgets"]["background"] = "private room"
    payload["state"]["elements"] = [{"desc": "secret subject", "x": 0, "y": 0, "w": 1, "h": 1}]

    created = library.create_prompt(
        payload,
        metadata={
            "id": "private-prompt",
            "name": "secret title",
            "description": "secret description",
            "tags": ["secret tag"],
            "private": True,
        },
        base_dir=tmp_path,
    )

    stored_text = library.library_path(tmp_path).read_text(encoding="utf-8")
    listed_text = json.dumps(library.list_items(tmp_path), ensure_ascii=False)
    assert "private room" not in stored_text
    assert "secret subject" not in stored_text
    assert "secret title" not in stored_text
    assert "secret description" not in stored_text
    assert "secret tag" not in stored_text
    assert '"encrypted_payload"' in stored_text
    assert '"payload"' not in json.dumps(library.load_library(tmp_path)["prompts"][0])
    assert "private room" not in listed_text
    assert "secret subject" not in listed_text
    assert "secret title" not in listed_text
    assert "secret description" not in listed_text
    assert "secret tag" not in listed_text
    assert library.list_items(tmp_path)["prompts"][0]["name"] == library.PRIVATE_ITEM_NAME
    assert library.list_items(tmp_path)["prompts"][0]["description"] == ""
    assert library.list_items(tmp_path)["prompts"][0]["tags"] == []
    assert library.list_items(tmp_path)["prompts"][0]["summary"] == {"is_private": True}
    assert created["name"] == "secret title"
    assert created["payload"] == payload
    assert created["description"] == "secret description"
    assert created["tags"] == ["secret tag"]

    used = library.use_prompt("private-prompt", base_dir=tmp_path)
    assert used["payload"] == payload
    assert used["name"] == "secret title"
    assert used["tags"] == ["secret tag"]


@pytest.mark.skipif(not privacy.CRYPTO_AVAILABLE, reason="cryptography is not installed")
def test_private_ideogram_prompt_locked_keystore_raises_readable_error(tmp_path):
    initialize_keystore(PASSWORD)
    library.create_prompt(
        sample_payload('{"secret":"locked"}'),
        metadata={"id": "private-prompt", "private": True},
        base_dir=tmp_path,
    )
    lock_keystore()

    with pytest.raises(privacy.PrivacyError, match="PRIVACY_LOCKED"):
        library.use_prompt("private-prompt", base_dir=tmp_path)


@pytest.mark.skipif(not privacy.CRYPTO_AVAILABLE, reason="cryptography is not installed")
def test_private_ideogram_prompt_unrecoverable_payload_can_still_be_deleted(tmp_path):
    initialize_keystore(PASSWORD)
    library.create_prompt(
        sample_payload('{"secret":"lost"}'),
        metadata={"id": "private-prompt", "private": True},
        base_dir=tmp_path,
    )
    stored = library.load_library(tmp_path)
    stored["prompts"][0]["encrypted_payload"]["schema"] = privacy.LEGACY_ENVELOPE_SCHEMA
    library.library_path(tmp_path).write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(library.Ideogram4PromptLibraryError, match="can still be deleted"):
        library.use_prompt("private-prompt", base_dir=tmp_path)

    assert library.delete_prompt("private-prompt", base_dir=tmp_path) == {
        "id": "private-prompt",
        "kind": "prompt",
    }


def test_private_ideogram_prompt_old_shell_metadata_is_scrubbed_on_write(tmp_path):
    old_library = {
        "schema_version": library.LIBRARY_SCHEMA_VERSION,
        "version": library.LIBRARY_VERSION,
        "prompts": [
            {
                "id": "old-private",
                "kind": "prompt",
                "type": library.PROMPT_LIBRARY_ITEM_TYPE,
                "name": "secret old name",
                "description": "secret old description",
                "tags": ["secret old tag"],
                "private": True,
                "is_private": True,
                "summary": {"prompt_char_count": 999, "is_private": True},
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "encrypted_payload": {
                    "encrypted": True,
                    "schema": privacy.LEGACY_ENVELOPE_SCHEMA,
                    "algorithm": privacy.ALGORITHM,
                },
            }
        ],
    }
    path = library.library_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old_library), encoding="utf-8")

    library.create_prompt(sample_payload(), metadata={"id": "public-prompt", "name": "Public"}, base_dir=tmp_path)

    stored_text = path.read_text(encoding="utf-8")
    listed_text = json.dumps(library.list_items(tmp_path), ensure_ascii=False)
    assert "secret old name" not in stored_text
    assert "secret old description" not in stored_text
    assert "secret old tag" not in stored_text
    assert "secret old name" not in listed_text
    assert library.list_items(tmp_path)["prompts"][0]["name"] == library.PRIVATE_ITEM_NAME
