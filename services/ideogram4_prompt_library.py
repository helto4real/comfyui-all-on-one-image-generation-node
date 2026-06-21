"""Persistent local library for Ideogram 4 prompt-builder states."""

from __future__ import annotations

import copy
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from .privacy import decrypt_state, encrypt_state
except ImportError:  # pragma: no cover - direct test imports
    from privacy import decrypt_state, encrypt_state


LIBRARY_FILE_NAME = "ideogram4_prompt_library.json"
LIBRARY_SCHEMA_VERSION = "1.0"
LIBRARY_VERSION = 1
PROMPT_KIND = "prompt"
PROMPT_LIBRARY_ITEM_TYPE = "IDEOGRAM4_PROMPT_LIBRARY_ITEM"


class Ideogram4PromptLibraryError(ValueError):
    """Raised for user-fixable prompt library failures."""


def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def library_path(base_dir: str | os.PathLike[str] | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else config_dir()
    return root / LIBRARY_FILE_NAME


def load_library(base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = library_path(base_dir)
    if not path.exists():
        return _empty_library()
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception as exc:  # noqa: BLE001 - corrupt user config should stay readable
        raise Ideogram4PromptLibraryError(f"Could not read Ideogram prompt library config: {exc}") from exc
    return _normalize_library(payload)


def list_items(base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    library = load_library(base_dir)
    return {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "version": LIBRARY_VERSION,
        "prompts": [_public_item(entry) for entry in library["prompts"]],
    }


def create_prompt(
    payload: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    now = _utc_now()
    normalized_payload = _normalize_payload(payload)
    entry = _pack_entry(
        item_id=str(metadata.get("id") or _new_id()),
        name=_coerce_text(metadata.get("name")) or _default_name(normalized_payload),
        description=_coerce_text(metadata.get("description")),
        tags=_coerce_tags(metadata.get("tags")),
        private=bool(metadata.get("private")),
        payload=normalized_payload,
        created_at=now,
        updated_at=now,
        base_dir=base_dir,
    )
    library = load_library(base_dir)
    if any(item.get("id") == entry["id"] for item in library["prompts"]):
        raise Ideogram4PromptLibraryError(f"Ideogram prompt already exists: {entry['id']}")
    library["prompts"].append(entry)
    _save_library(library, base_dir)
    return _with_payload(entry, base_dir=base_dir)


def replace_prompt(
    item_id: str,
    payload: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    library = load_library(base_dir)
    entry = _find_entry(library, item_id)
    normalized_payload = _normalize_payload(payload)
    replacement = _pack_entry(
        item_id=item_id,
        name=_coerce_text(metadata.get("name")) or str(entry.get("name") or _default_name(normalized_payload)),
        description=_coerce_text(metadata.get("description")),
        tags=_coerce_tags(metadata.get("tags", entry.get("tags", []))),
        private=bool(metadata.get("private", entry.get("private", False))),
        payload=normalized_payload,
        created_at=str(entry.get("created_at") or _utc_now()),
        updated_at=_utc_now(),
        base_dir=base_dir,
    )
    _replace_entry(library, item_id, replacement)
    _save_library(library, base_dir)
    return _with_payload(replacement, base_dir=base_dir)


def patch_prompt(
    item_id: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    library = load_library(base_dir)
    entry = _find_entry(library, item_id)
    current_payload = _unpack_payload(entry, base_dir=base_dir)
    next_payload = _normalize_payload(payload if payload is not None else current_payload)
    patched = _pack_entry(
        item_id=item_id,
        name=_coerce_text(metadata.get("name", entry.get("name"))) or _default_name(next_payload),
        description=_coerce_text(metadata.get("description", _unpack_description(entry, base_dir=base_dir))),
        tags=_coerce_tags(metadata.get("tags", entry.get("tags", []))),
        private=bool(metadata.get("private", entry.get("private", False))),
        payload=next_payload,
        created_at=str(entry.get("created_at") or _utc_now()),
        updated_at=_utc_now(),
        base_dir=base_dir,
    )
    _replace_entry(library, item_id, patched)
    _save_library(library, base_dir)
    return _with_payload(patched, base_dir=base_dir)


def duplicate_prompt(
    item_id: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    library = load_library(base_dir)
    source = _find_entry(library, item_id)
    payload = _unpack_payload(source, base_dir=base_dir)
    now = _utc_now()
    duplicate = _pack_entry(
        item_id=str(metadata.get("id") or _new_id()),
        name=_coerce_text(metadata.get("name")) or f"{source.get('name') or _default_name(payload)} Copy",
        description=_coerce_text(metadata.get("description", _unpack_description(source, base_dir=base_dir))),
        tags=_coerce_tags(metadata.get("tags", source.get("tags", []))),
        private=bool(metadata.get("private", source.get("private", False))),
        payload=payload,
        created_at=now,
        updated_at=now,
        base_dir=base_dir,
    )
    if any(item.get("id") == duplicate["id"] for item in library["prompts"]):
        raise Ideogram4PromptLibraryError(f"Ideogram prompt already exists: {duplicate['id']}")
    library["prompts"].append(duplicate)
    _save_library(library, base_dir)
    return _with_payload(duplicate, base_dir=base_dir)


def delete_prompt(
    item_id: str,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    library = load_library(base_dir)
    before = len(library["prompts"])
    library["prompts"] = [entry for entry in library["prompts"] if entry.get("id") != item_id]
    if len(library["prompts"]) == before:
        raise Ideogram4PromptLibraryError(f"Ideogram prompt not found: {item_id}")
    _save_library(library, base_dir)
    return {"id": item_id, "kind": PROMPT_KIND}


def use_prompt(
    item_id: str,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    library = load_library(base_dir)
    entry = _find_entry(library, item_id)
    item = _with_payload(entry, base_dir=base_dir)
    entry["last_used_at"] = _utc_now()
    _save_library(library, base_dir)
    item["last_used_at"] = entry["last_used_at"]
    return item


def _pack_entry(
    *,
    item_id: str,
    name: str,
    description: str,
    tags: list[str],
    private: bool,
    payload: Mapping[str, Any],
    created_at: str,
    updated_at: str,
    base_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    entry = {
        "id": str(item_id),
        "kind": PROMPT_KIND,
        "type": PROMPT_LIBRARY_ITEM_TYPE,
        "name": str(name),
        "tags": tags,
        "private": bool(private),
        "is_private": bool(private),
        "summary": {**_summary_for(payload), "is_private": bool(private)},
        "created_at": created_at,
        "updated_at": updated_at,
    }
    if private:
        entry["encrypted_payload"] = encrypt_state(
            {"payload": payload, "description": description},
            base_dir=base_dir,
        )
    else:
        entry["description"] = description
        entry["payload"] = copy.deepcopy(dict(payload))
    return entry


def _with_payload(entry: Mapping[str, Any], *, base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    item = _public_item(entry)
    payload = _unpack_payload(entry, base_dir=base_dir)
    item["description"] = _unpack_description(entry, base_dir=base_dir)
    item["payload"] = payload
    item["prompt"] = payload
    return item


def _public_item(entry: Mapping[str, Any]) -> dict[str, Any]:
    private = bool(entry.get("private") or entry.get("is_private"))
    item = {
        "id": str(entry.get("id") or ""),
        "kind": PROMPT_KIND,
        "type": str(entry.get("type") or PROMPT_LIBRARY_ITEM_TYPE),
        "name": str(entry.get("name") or ""),
        "description": "" if private else str(entry.get("description") or ""),
        "tags": _coerce_tags(entry.get("tags")),
        "private": private,
        "is_private": private,
        "summary": copy.deepcopy(entry.get("summary") if isinstance(entry.get("summary"), dict) else {}),
        "created_at": str(entry.get("created_at") or ""),
        "updated_at": str(entry.get("updated_at") or ""),
        "last_used_at": entry.get("last_used_at") if entry.get("last_used_at") else None,
    }
    if not private and isinstance(entry.get("payload"), Mapping):
        item["prompt_preview"] = _prompt_preview(entry["payload"])
    else:
        item["prompt_preview"] = ""
    return item


def _unpack_payload(entry: Mapping[str, Any], *, base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    if entry.get("private"):
        state = decrypt_state(entry.get("encrypted_payload"), base_dir=base_dir)
        payload = state.get("payload")
    else:
        payload = entry.get("payload")
    return _normalize_payload(payload if isinstance(payload, Mapping) else {})


def _unpack_description(entry: Mapping[str, Any], *, base_dir: str | os.PathLike[str] | None = None) -> str:
    if entry.get("private"):
        state = decrypt_state(entry.get("encrypted_payload"), base_dir=base_dir)
        return _coerce_text(state.get("description"))
    return _coerce_text(entry.get("description"))


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise Ideogram4PromptLibraryError("Ideogram prompt library payload must be an object.")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise Ideogram4PromptLibraryError("Ideogram prompt library payload requires a state object.")
    return {
        "family": "ideogram4",
        "version": _safe_int(payload.get("version"), 1),
        "state": copy.deepcopy(dict(state)),
        "prompt": str(payload.get("prompt") or ""),
    }


def _summary_for(payload: Mapping[str, Any]) -> dict[str, Any]:
    state = payload.get("state") if isinstance(payload.get("state"), Mapping) else {}
    widgets = state.get("widgets") if isinstance(state.get("widgets"), Mapping) else {}
    elements = state.get("elements") if isinstance(state.get("elements"), list) else []
    palette = state.get("style_palette") if isinstance(state.get("style_palette"), list) else []
    return {
        "family": "ideogram4",
        "element_count": len(elements),
        "style_color_count": len(palette),
        "max_side": _safe_int(widgets.get("max side"), 0),
        "aspect_ratio": str(widgets.get("aspect ratio") or ""),
        "output_format": str(state.get("output_format") or widgets.get("output_format") or ""),
        "prompt_char_count": len(str(payload.get("prompt") or "")),
    }


def _prompt_preview(payload: Mapping[str, Any]) -> str:
    text = str(payload.get("prompt") or "").strip().replace("\n", " ")
    return text[:240]


def _empty_library() -> dict[str, Any]:
    return {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "version": LIBRARY_VERSION,
        "prompts": [],
    }


def _normalize_library(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _empty_library()
    library = _empty_library()
    prompts = payload.get("prompts")
    if isinstance(prompts, list):
        library["prompts"] = [dict(item) for item in prompts if isinstance(item, dict)]
    return library


def _save_library(library: Mapping[str, Any], base_dir: str | os.PathLike[str] | None = None) -> None:
    payload = _normalize_library(library)
    path = library_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp_path = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    tmp_path.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _find_entry(library: Mapping[str, Any], item_id: str) -> dict[str, Any]:
    for entry in library["prompts"]:  # type: ignore[index]
        if entry.get("id") == item_id:
            return entry
    raise Ideogram4PromptLibraryError(f"Ideogram prompt not found: {item_id}")


def _replace_entry(library: dict[str, Any], item_id: str, replacement: dict[str, Any]) -> None:
    for index, entry in enumerate(library["prompts"]):
        if entry.get("id") == item_id:
            library["prompts"][index] = replacement
            return
    raise Ideogram4PromptLibraryError(f"Ideogram prompt not found: {item_id}")


def _default_name(payload: Mapping[str, Any]) -> str:
    preview = _prompt_preview(payload)
    return preview[:48] or "Untitled Ideogram Prompt"


def _new_id() -> str:
    return f"prompt_{secrets.token_hex(8)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_text(value: Any) -> str:
    return str(value or "").strip()


def _coerce_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    for item in value:
        tag = str(item or "").strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


__all__ = [
    "Ideogram4PromptLibraryError",
    "LIBRARY_FILE_NAME",
    "LIBRARY_SCHEMA_VERSION",
    "LIBRARY_VERSION",
    "PROMPT_LIBRARY_ITEM_TYPE",
    "config_dir",
    "create_prompt",
    "delete_prompt",
    "duplicate_prompt",
    "library_path",
    "list_items",
    "load_library",
    "patch_prompt",
    "replace_prompt",
    "use_prompt",
]
