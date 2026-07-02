"""Shared privacy helpers for AIO Image Generate."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    import helto_privacy.envelope as _envelope
    import helto_privacy.keystore as _keystore
    from helto_privacy import PrivacyEnvelopeCodec
    from helto_privacy import PrivacyError as _HeltoPrivacyError
    from helto_privacy import PrivacyKeystoreError

    CRYPTO_AVAILABLE = _envelope.CRYPTO_AVAILABLE and _keystore.KEYSTORE_CRYPTO_AVAILABLE
    CRYPTO_IMPORT_ERROR = _envelope.CRYPTO_IMPORT_ERROR or _keystore.KEYSTORE_CRYPTO_IMPORT_ERROR
    _IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - keep imports readable when dependency is missing.
    _envelope = None  # type: ignore[assignment]
    _keystore = None  # type: ignore[assignment]
    PrivacyEnvelopeCodec = None  # type: ignore[assignment]
    PrivacyKeystoreError = RuntimeError  # type: ignore[assignment]
    CRYPTO_AVAILABLE = False
    CRYPTO_IMPORT_ERROR = str(exc)
    _IMPORT_ERROR = str(exc)

    class _HeltoPrivacyError(RuntimeError):
        """Fallback privacy error when helto-privacy is not installed."""


ENVELOPE_SCHEMA = "helto.aio-image-generate.v2"
LEGACY_ENVELOPE_SCHEMA = "helto.aio-image-generate"
ENVELOPE_VERSION = 1
ALGORITHM = "AES-256-GCM"
KEY_FILE_NAME = "privacy_key.json"


class PrivacyError(_HeltoPrivacyError):
    """Raised when AIO privacy encryption/decryption cannot complete safely."""


_codec = PrivacyEnvelopeCodec(ENVELOPE_SCHEMA) if PrivacyEnvelopeCodec is not None else None


def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def key_path(base_dir: str | os.PathLike[str] | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else config_dir()
    return root / KEY_FILE_NAME


def crypto_status(base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    del base_dir
    status = {
        "available": CRYPTO_AVAILABLE,
        "algorithm": ALGORITHM,
        "schema": ENVELOPE_SCHEMA,
        "legacySchema": LEGACY_ENVELOPE_SCHEMA,
        "legacyKeyEnabled": False,
        "keyExists": False,
        "keyPath": "",
        "error": "",
    }
    if _IMPORT_ERROR:
        status["error"] = f"Python package 'helto-privacy' is required for privacy mode: {_IMPORT_ERROR}"
    elif not CRYPTO_AVAILABLE:
        status["error"] = f"Python package 'cryptography' is required for privacy mode: {CRYPTO_IMPORT_ERROR}"
    if _keystore is not None:
        status.update(_keystore.keystore_status())
    return status


def is_encrypted_payload(value: Any) -> bool:
    payload = _parse_payload(value)
    return (
        isinstance(payload, Mapping)
        and payload.get("encrypted") is True
        and payload.get("schema") == ENVELOPE_SCHEMA
        and payload.get("algorithm") == ALGORITHM
    )


def is_legacy_encrypted_payload(value: Any) -> bool:
    payload = _parse_payload(value)
    return (
        isinstance(payload, Mapping)
        and payload.get("encrypted") is True
        and payload.get("schema") == LEGACY_ENVELOPE_SCHEMA
        and payload.get("algorithm") == ALGORITHM
    )


def encrypt_state(state: Mapping[str, Any], base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    del base_dir
    try:
        return _require_codec().encrypt_state(state)
    except (_HeltoPrivacyError, PrivacyKeystoreError) as exc:
        raise PrivacyError(str(exc)) from exc


def decrypt_state(payload: Any, base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    del base_dir
    payload = _parse_payload_or_raise(payload)
    if is_legacy_encrypted_payload(payload):
        raise PrivacyError(
            "Unsupported legacy AIO privacy payload. Re-enter the private value to save it with the shared privacy keystore."
        )
    if not is_encrypted_payload(payload):
        raise PrivacyError("AIO data is not an encrypted privacy payload.")
    try:
        return _require_codec().decrypt_state(payload)
    except (_HeltoPrivacyError, PrivacyKeystoreError) as exc:
        raise PrivacyError(str(exc)) from exc


def decrypt_if_encrypted(value: Any) -> Any:
    if is_legacy_encrypted_payload(value):
        raise PrivacyError(
            "Unsupported legacy AIO privacy payload. Re-enter the private value to save it with the shared privacy keystore."
        )
    if not is_encrypted_payload(value):
        return value
    state = decrypt_state(value)
    return state.get("value", state)


def decrypt_text_if_encrypted(value: Any) -> str:
    decrypted = decrypt_if_encrypted(value)
    return "" if decrypted is None else str(decrypted)


def _require_codec():
    if _IMPORT_ERROR:
        raise PrivacyError(f"Python package 'helto-privacy' is required for privacy mode: {_IMPORT_ERROR}")
    if not CRYPTO_AVAILABLE:
        raise PrivacyError(f"Python package 'cryptography' is required for privacy mode: {CRYPTO_IMPORT_ERROR}")
    if _keystore is None or _codec is None:
        raise PrivacyError("Python package 'helto-privacy' is required for privacy mode.")
    if not _keystore.keystore_exists():
        raise PrivacyError(
            f"{_keystore.ERROR_UNINITIALIZED}: Privacy keystore has not been created yet. "
            "Open the Helto privacy dialog and set a privacy password."
        )
    return _codec


def _parse_payload(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def _parse_payload_or_raise(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception as exc:
            raise PrivacyError(f"Encrypted AIO data is not valid JSON: {exc}") from exc
    return value
