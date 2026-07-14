"""Inactive shared-privacy adapter for the Ideogram prompt library.

The adapter owns product JSON persistence and normalization only.  Encryption,
authorization, locked shells, deletion confirmation, and route errors remain
the responsibility of :mod:`helto_privacy`.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import secrets
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from helto_privacy import MigrationVerification, RecordProjectionResult, RecordSnapshot


PROMPT_LIBRARY_RESOURCE_ID = "ideogram-prompts"
PROMPT_RECORD_KIND = "ideogram-prompt"
PROMPT_LIBRARY_STORE_ADAPTER_ID = "ideogram-prompt-store"
PROMPT_LIBRARY_CURRENT_SCHEMA = "helto.aio-image-generate.v2"
PROMPT_LIBRARY_LEGACY_BINDING_ID = "ideogram-prompt-aio-v1"
PROMPT_LIBRARY_LEGACY_KEY_BINDING_ID = "ideogram-prompt-aio-json-key-v1"

MANAGED_LIBRARY_FILE_NAME = "ideogram4_prompt_library_v2.json"
MANAGED_LIBRARY_SCHEMA_VERSION = "2.0"
MANAGED_LIBRARY_VERSION = 2
LEGACY_LIBRARY_FILE_NAME = "ideogram4_prompt_library.json"
LEGACY_LIBRARY_SCHEMA_VERSION = "1.0"
LEGACY_LIBRARY_VERSION = 1


class AioPromptLibraryDataError(ValueError):
    """Raised when product-owned library data is invalid."""


def managed_library_path(
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    root = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parents[1] / "config"
    )
    return root / MANAGED_LIBRARY_FILE_NAME


def legacy_library_path(
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return managed_library_path(base_dir).with_name(LEGACY_LIBRARY_FILE_NAME)


@dataclass(frozen=True, slots=True)
class AioLegacyPromptRecordSource:
    """One non-decrypting private-record source from the exact v1 document."""

    legacy_id: str = field(repr=False)
    protected: object = field(repr=False)
    created_at: str = field(default="", repr=False)
    updated_at: str = field(default="", repr=False)
    last_used_at: str | None = field(default=None, repr=False)

    @property
    def current_format(self) -> bool:
        return (
            isinstance(self.protected, Mapping)
            and self.protected.get("schema") == PROMPT_LIBRARY_CURRENT_SCHEMA
        )


def discover_legacy_prompt_record_sources(
    base_dir: str | os.PathLike[str] | None = None,
) -> tuple[AioLegacyPromptRecordSource, ...]:
    """Read exact v1 document structure without decrypting or building shells."""

    path = legacy_library_path(base_dir)
    if not path.exists():
        return ()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise AioPromptLibraryDataError("Legacy prompt library is unreadable.") from None
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != LEGACY_LIBRARY_SCHEMA_VERSION
        or document.get("version") != LEGACY_LIBRARY_VERSION
        or not isinstance(document.get("prompts"), list)
    ):
        raise AioPromptLibraryDataError("Legacy prompt library format is invalid.")

    sources = []
    seen = set()
    for item in document["prompts"]:
        if not isinstance(item, Mapping):
            raise AioPromptLibraryDataError("Legacy prompt library record is invalid.")
        if not bool(item.get("private") or item.get("is_private")):
            continue
        legacy_id = item.get("id")
        protected = item.get("encrypted_payload")
        if isinstance(protected, str):
            try:
                protected = json.loads(protected)
            except json.JSONDecodeError:
                raise AioPromptLibraryDataError(
                    "Legacy prompt library record is invalid."
                ) from None
        if (
            not isinstance(legacy_id, str)
            or not legacy_id
            or legacy_id in seen
            or not isinstance(protected, Mapping)
        ):
            raise AioPromptLibraryDataError("Legacy prompt library record is invalid.")
        seen.add(legacy_id)
        sources.append(
            AioLegacyPromptRecordSource(
                legacy_id,
                copy.deepcopy(dict(protected)),
                _text(item.get("created_at")),
                _text(item.get("updated_at")),
                _text(item.get("last_used_at")) or None,
            )
        )
    return tuple(sources)


class AioPromptLibraryStoreAdapter:
    """Persist opaque current envelopes and normalize authorized record edits."""

    def __init__(self, base_dir: str | os.PathLike[str] | None = None) -> None:
        self._base_dir = base_dir
        self._thread_lock = RLock()

    def list_ids(self) -> tuple[str, ...]:
        with self._managed_lock():
            return tuple(
                record["id"]
                for record in self._read_document()["records"]
                if record["protected"] is not None
            )

    def read_record(self, record_id: str) -> RecordSnapshot:
        with self._managed_lock():
            record = self._find_optional(self._read_document(), record_id)
            if record is None:
                return RecordSnapshot(0)
            return RecordSnapshot(record["revision"], record["protected"])

    def compare_and_swap_record(
        self,
        record_id: str,
        expected: RecordSnapshot,
        replacement: RecordSnapshot,
    ) -> bool:
        _require_record_snapshot(expected)
        _require_record_snapshot(replacement)
        if replacement.revision != expected.revision + 1:
            raise AioPromptLibraryDataError("Prompt record revision is invalid.")
        with self._managed_lock():
            document = self._read_document()
            record = self._find_optional(document, record_id)
            current = (
                RecordSnapshot(0)
                if record is None
                else RecordSnapshot(record["revision"], record["protected"])
            )
            if not _record_snapshot_equal(current, expected):
                return False
            stored = {
                "id": str(record_id),
                "revision": replacement.revision,
                "protected": copy.deepcopy(replacement.protected),
            }
            if record is None:
                document["records"].append(stored)
            else:
                document["records"][document["records"].index(record)] = stored
            self._write_document(document)
            return True

    def read_protected(self, record_id: str) -> object:
        snapshot = self.read_record(record_id)
        if snapshot.protected is None:
            raise AioPromptLibraryDataError("Prompt record was not found.")
        return copy.deepcopy(snapshot.protected)

    def write_protected(self, record_id: str, protected: object) -> None:
        with self._managed_lock():
            document = self._read_document()
            existing = self._find_optional(document, record_id)
            replacement = {
                "id": str(record_id),
                "revision": 1 if existing is None else existing["revision"] + 1,
                "protected": copy.deepcopy(protected),
            }
            if existing is None:
                document["records"].append(replacement)
            else:
                document["records"][document["records"].index(existing)] = replacement
            self._write_document(document)

    def delete(self, record_id: str) -> None:
        with self._managed_lock():
            document = self._read_document()
            existing = self._find_optional(document, record_id)
            if existing is None or existing["protected"] is None:
                raise AioPromptLibraryDataError("Prompt record was not found.")
            document["records"][document["records"].index(existing)] = {
                "id": str(record_id),
                "revision": existing["revision"] + 1,
                "protected": None,
            }
            self._write_document(document)

    def mutate(
        self,
        current: object,
        operation: str,
        value: object,
    ) -> dict[str, object]:
        request = _mapping(value, "Prompt library mutation must be an object.")
        metadata = _metadata(request.get("metadata"))
        now = _utc_now()
        if operation == "create":
            payload = normalize_ideogram_prompt_payload(request.get("payload"))
            return _record(
                payload,
                metadata,
                created_at=now,
                updated_at=now,
            )

        source = normalize_ideogram_prompt_record(current)
        if operation == "duplicate":
            duplicate_metadata = {
                "name": metadata.get("name") or f"{source['name']} Copy",
                "description": metadata.get("description", source["description"]),
                "tags": metadata.get("tags", source["tags"]),
            }
            return _record(
                source["payload"],
                duplicate_metadata,
                created_at=now,
                updated_at=now,
            )
        if operation not in {"replace", "patch"}:
            raise AioPromptLibraryDataError("Prompt library mutation is invalid.")

        payload = (
            normalize_ideogram_prompt_payload(request.get("payload"))
            if "payload" in request
            else source["payload"]
        )
        if operation == "replace" and "payload" not in request:
            raise AioPromptLibraryDataError("Replacement requires a prompt payload.")
        next_metadata = {
            "name": metadata.get("name", source["name"]),
            "description": metadata.get("description", source["description"]),
            "tags": metadata.get("tags", source["tags"]),
        }
        result = _record(
            payload,
            next_metadata,
            created_at=source["created_at"],
            updated_at=now,
        )
        result["last_used_at"] = source["last_used_at"]
        return result

    def project(self, value: object, operation: str) -> RecordProjectionResult:
        record = normalize_ideogram_prompt_record(value)
        if operation == "details":
            return RecordProjectionResult({"record": record})
        if operation != "use":
            raise AioPromptLibraryDataError("Prompt library projection is invalid.")
        used = copy.deepcopy(record)
        used["last_used_at"] = _utc_now()
        return RecordProjectionResult({"record": used}, used)

    def prepare_mode_transition(self, *_args) -> None:
        return None

    def commit_mode_transition(self, *_args) -> None:
        return None

    def rollback_mode_transition(self, *_args) -> None:
        return None

    def _read_document(self) -> dict[str, Any]:
        path = managed_library_path(self._base_dir)
        if not path.exists():
            return _empty_document()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise AioPromptLibraryDataError(
                "Managed prompt library is unreadable."
            ) from None
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != MANAGED_LIBRARY_SCHEMA_VERSION
            or value.get("version") != MANAGED_LIBRARY_VERSION
            or not isinstance(value.get("records"), list)
        ):
            raise AioPromptLibraryDataError("Managed prompt library format is invalid.")
        records = []
        seen = set()
        for item in value["records"]:
            if (
                not isinstance(item, dict)
                or set(item) not in ({"id", "protected"}, {"id", "revision", "protected"})
                or not isinstance(item["id"], str)
                or not item["id"]
                or item["id"] in seen
            ):
                raise AioPromptLibraryDataError(
                    "Managed prompt library record is invalid."
                )
            seen.add(item["id"])
            revision = item.get("revision", 1)
            if type(revision) is not int or revision < 1:
                raise AioPromptLibraryDataError(
                    "Managed prompt library record is invalid."
                )
            records.append(
                {
                    "id": item["id"],
                    "revision": revision,
                    "protected": copy.deepcopy(item["protected"]),
                }
            )
        return {**_empty_document(), "records": records}

    @contextmanager
    def _managed_lock(self):
        path = managed_library_path(self._base_dir)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        lock_path = path.with_name(f".{path.name}.lock")
        with self._thread_lock:
            descriptor = os.open(
                lock_path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                with os.fdopen(descriptor, "a+b") as handle:
                    descriptor = -1
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

    def _write_document(self, document: Mapping[str, object]) -> None:
        path = managed_library_path(self._base_dir)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _find(document: Mapping[str, object], record_id: str) -> Mapping[str, object]:
        record = AioPromptLibraryStoreAdapter._find_optional(document, record_id)
        if record is not None and record.get("protected") is not None:
            return record
        raise AioPromptLibraryDataError("Prompt record was not found.")

    @staticmethod
    def _find_optional(
        document: Mapping[str, object],
        record_id: str,
    ) -> MutableMapping[str, object] | None:
        records = document.get("records")
        if isinstance(records, list):
            for record in records:
                if isinstance(record, MutableMapping) and record.get("id") == record_id:
                    return record
        return None


@dataclass(slots=True)
class AioPromptLibraryMigrationTransaction:
    """Commit one genuine AIO v1 record only after current read-back succeeds."""

    records: object
    adapter: AioPromptLibraryStoreAdapter
    record_id: str
    protect_authorization: object
    reveal_authorization: object
    created_at: str = ""
    updated_at: str = ""
    last_used_at: str | None = None
    original: object = None
    expected: object = None
    staged: object = None

    def capture_original(self) -> object:
        try:
            self.original = {
                "present": True,
                "protected": self.adapter.read_protected(self.record_id),
            }
        except AioPromptLibraryDataError:
            self.original = {"present": False}
        return copy.deepcopy(self.original)

    def stage_current(self, normalized: object) -> None:
        legacy = normalize_legacy_prompt_record(
            normalized,
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_used_at=self.last_used_at,
        )
        self.expected = copy.deepcopy(normalized)
        self.staged = self.records.protect(
            PROMPT_RECORD_KIND,
            legacy,
            self.protect_authorization,
        ).envelope

    def stage_durable_adjuncts(self, _normalized: object) -> None:
        return None

    def commit(self) -> None:
        if self.staged is None:
            raise AioPromptLibraryDataError("Prompt migration was not staged.")
        self.adapter.write_protected(self.record_id, self.staged)

    def read_back(self) -> MigrationVerification:
        revealed = self.records.reveal(
            PROMPT_RECORD_KIND,
            self.record_id,
            "details",
            self.reveal_authorization,
        ).value
        record = normalize_ideogram_prompt_record(revealed.get("record"))
        legacy = normalize_legacy_prompt_record(
            self.expected,
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_used_at=self.last_used_at,
        )
        if any(
            record[key] != legacy[key]
            for key in ("name", "description", "tags", "payload")
        ):
            raise AioPromptLibraryDataError("Prompt migration verification failed.")
        current = (
            isinstance(self.staged, Mapping)
            and self.staged.get("schema") == PROMPT_LIBRARY_CURRENT_SCHEMA
        )
        return MigrationVerification(copy.deepcopy(self.expected), current, True)

    def rollback(self, original: object) -> None:
        snapshot = _mapping(original, "Prompt migration original is invalid.")
        if snapshot.get("present") is False:
            try:
                self.adapter.delete(self.record_id)
            except AioPromptLibraryDataError:
                pass
        elif snapshot.get("present") is True and "protected" in snapshot:
            self.adapter.write_protected(self.record_id, snapshot["protected"])
        else:
            raise AioPromptLibraryDataError("Prompt migration original is invalid.")
        self.staged = None

    def finalize(self, _original: object) -> None:
        self.original = None
        self.expected = None
        self.staged = None


def normalize_legacy_prompt_record(
    value: object,
    *,
    created_at: str = "",
    updated_at: str = "",
    last_used_at: str | None = None,
) -> dict[str, object]:
    legacy = _mapping(value, "Legacy prompt record is invalid.")
    payload = normalize_ideogram_prompt_payload(legacy.get("payload"))
    now = _utc_now()
    result = _record(
        payload,
        {
            "name": legacy.get("name"),
            "description": legacy.get("description"),
            "tags": legacy.get("tags"),
        },
        created_at=_text(created_at) or now,
        updated_at=_text(updated_at) or now,
    )
    result["last_used_at"] = _text(last_used_at) or None
    return result


def normalize_ideogram_prompt_record(value: object) -> dict[str, object]:
    record = _mapping(value, "Prompt record is invalid.")
    payload = normalize_ideogram_prompt_payload(record.get("payload"))
    created_at = _text(record.get("created_at"))
    updated_at = _text(record.get("updated_at"))
    if not created_at or not updated_at:
        raise AioPromptLibraryDataError("Prompt record timestamps are invalid.")
    return {
        "name": _text(record.get("name")) or _default_name(payload),
        "description": _text(record.get("description")),
        "tags": _tags(record.get("tags")),
        "payload": payload,
        "created_at": created_at,
        "updated_at": updated_at,
        "last_used_at": _text(record.get("last_used_at")) or None,
    }


def normalize_ideogram_prompt_payload(value: object) -> dict[str, object]:
    payload = _mapping(value, "Ideogram prompt payload must be an object.")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise AioPromptLibraryDataError(
            "Ideogram prompt payload requires a state object."
        )
    return {
        "family": "ideogram4",
        "version": _integer(payload.get("version"), 1),
        "state": copy.deepcopy(dict(state)),
        "prompt": str(payload.get("prompt") or ""),
    }


def _record(
    payload: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    created_at: str,
    updated_at: str,
) -> dict[str, object]:
    return {
        "name": _text(metadata.get("name")) or _default_name(payload),
        "description": _text(metadata.get("description")),
        "tags": _tags(metadata.get("tags")),
        "payload": copy.deepcopy(dict(payload)),
        "created_at": created_at,
        "updated_at": updated_at,
        "last_used_at": None,
    }


def _metadata(value: object) -> dict[str, object]:
    if value is None:
        return {}
    metadata = _mapping(value, "Prompt metadata must be an object.")
    allowed = {"name", "description", "tags"}
    if set(metadata) - allowed:
        raise AioPromptLibraryDataError("Prompt metadata contains unknown fields.")
    result: dict[str, object] = {}
    if "name" in metadata:
        result["name"] = _text(metadata["name"])
    if "description" in metadata:
        result["description"] = _text(metadata["description"])
    if "tags" in metadata:
        result["tags"] = _tags(metadata["tags"])
    return result


def _require_record_snapshot(value: object) -> None:
    if not isinstance(value, RecordSnapshot):
        raise AioPromptLibraryDataError("Prompt record snapshot is invalid.")


def _record_snapshot_equal(left: RecordSnapshot, right: RecordSnapshot) -> bool:
    return (
        left.revision == right.revision
        and left.protected == right.protected
    )


def _empty_document() -> dict[str, object]:
    return {
        "schema_version": MANAGED_LIBRARY_SCHEMA_VERSION,
        "version": MANAGED_LIBRARY_VERSION,
        "records": [],
    }


def _mapping(value: object, message: str) -> MutableMapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AioPromptLibraryDataError(message)
    return dict(value)


def _default_name(payload: Mapping[str, object]) -> str:
    text = str(payload.get("prompt") or "").strip().replace("\n", " ")
    return text[:48] or "Untitled Ideogram Prompt"


def _text(value: object) -> str:
    return str(value or "").strip()


def _tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        tag = _text(item)
        if tag and tag not in result:
            result.append(tag)
    return result


def _integer(value: object, fallback: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "AioLegacyPromptRecordSource",
    "AioPromptLibraryDataError",
    "AioPromptLibraryMigrationTransaction",
    "AioPromptLibraryStoreAdapter",
    "LEGACY_LIBRARY_FILE_NAME",
    "MANAGED_LIBRARY_FILE_NAME",
    "PROMPT_LIBRARY_CURRENT_SCHEMA",
    "PROMPT_LIBRARY_LEGACY_BINDING_ID",
    "PROMPT_LIBRARY_LEGACY_KEY_BINDING_ID",
    "PROMPT_LIBRARY_RESOURCE_ID",
    "PROMPT_LIBRARY_STORE_ADAPTER_ID",
    "PROMPT_RECORD_KIND",
    "discover_legacy_prompt_record_sources",
    "legacy_library_path",
    "managed_library_path",
    "normalize_ideogram_prompt_payload",
]
