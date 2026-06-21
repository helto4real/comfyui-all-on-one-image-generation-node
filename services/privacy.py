"""Local privacy encryption helpers for AIO Image Generate."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Mapping

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
    CRYPTO_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - optional dependency in ComfyUI installs
    AESGCM = None  # type: ignore[assignment]
    CRYPTO_AVAILABLE = False
    CRYPTO_IMPORT_ERROR = str(exc)


ENVELOPE_SCHEMA = "helto.aio-image-generate"
ENVELOPE_VERSION = 1
ALGORITHM = "AES-256-GCM"
KEY_FILE_NAME = "privacy_key.json"


class PrivacyError(RuntimeError):
    """Raised when local privacy encryption/decryption cannot complete safely."""


def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def key_path(base_dir: str | os.PathLike[str] | None = None) -> Path:
    return Path(base_dir) / KEY_FILE_NAME if base_dir is not None else config_dir() / KEY_FILE_NAME


def crypto_status(base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = key_path(base_dir)
    return {
        "available": CRYPTO_AVAILABLE,
        "algorithm": ALGORITHM,
        "keyExists": path.exists(),
        "keyPath": str(path),
        "error": "" if CRYPTO_AVAILABLE else f"Python package 'cryptography' is required: {CRYPTO_IMPORT_ERROR}",
    }


def is_encrypted_payload(value: Any) -> bool:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return False
    return (
        isinstance(value, Mapping)
        and value.get("encrypted") is True
        and value.get("schema") == ENVELOPE_SCHEMA
        and value.get("algorithm") == ALGORITHM
    )


def encrypt_state(state: Mapping[str, Any], base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    key, key_id = _load_or_create_key(base_dir, create=True)
    nonce = secrets.token_bytes(12)
    plaintext = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _aad(key_id))  # type: ignore[operator]
    return {
        "version": ENVELOPE_VERSION,
        "schema": ENVELOPE_SCHEMA,
        "encrypted": True,
        "algorithm": ALGORITHM,
        "keyId": key_id,
        "nonce": _b64url_encode(nonce),
        "ciphertext": _b64url_encode(ciphertext),
    }


def decrypt_state(payload: Any, base_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception as exc:
            raise PrivacyError(f"Encrypted AIO data is not valid JSON: {exc}") from exc
    if not is_encrypted_payload(payload):
        raise PrivacyError("AIO data is not an encrypted privacy payload.")
    key, key_id = _load_or_create_key(base_dir, create=False)
    if str(payload.get("keyId", "")) != key_id:
        raise PrivacyError("Encrypted AIO data was created with a different local privacy key.")
    try:
        nonce = _b64url_decode(str(payload.get("nonce", "")))
        ciphertext = _b64url_decode(str(payload.get("ciphertext", "")))
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, _aad(key_id))  # type: ignore[operator]
        loaded = json.loads(plaintext.decode("utf-8"))
    except PrivacyError:
        raise
    except Exception as exc:  # noqa: BLE001 - auth/key failures should stay user-readable
        raise PrivacyError(f"Could not decrypt AIO data: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise PrivacyError("Encrypted AIO data did not contain a state object.")
    return dict(loaded)


def decrypt_if_encrypted(value: Any) -> Any:
    if not is_encrypted_payload(value):
        return value
    state = decrypt_state(value)
    return state.get("value", state)


def decrypt_text_if_encrypted(value: Any) -> str:
    decrypted = decrypt_if_encrypted(value)
    return "" if decrypted is None else str(decrypted)


def _load_or_create_key(base_dir: str | os.PathLike[str] | None = None, create: bool = True) -> tuple[bytes, str]:
    if not CRYPTO_AVAILABLE:
        raise PrivacyError(f"Python package 'cryptography' is required for privacy mode: {CRYPTO_IMPORT_ERROR}")

    path = key_path(base_dir)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = _b64url_decode(str(payload.get("key", "")))
            key_id = str(payload.get("keyId", "")).strip()
        except Exception as exc:  # noqa: BLE001
            raise PrivacyError(f"Could not read privacy key file '{path}': {exc}") from exc
        if len(key) != 32 or not key_id:
            raise PrivacyError(f"Privacy key file '{path}' is malformed.")
        return key, key_id

    if not create:
        raise PrivacyError(f"Privacy key file is missing: {path}")

    key = secrets.token_bytes(32)
    key_id = _b64url_encode(hashlib.sha256(key).digest()[:12])
    _write_private_json(
        path,
        {
            "version": 1,
            "algorithm": ALGORITHM,
            "keyId": key_id,
            "key": _b64url_encode(key),
        },
    )
    return key, key_id


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
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


def _aad(key_id: str) -> bytes:
    return f"{ENVELOPE_SCHEMA}|{ENVELOPE_VERSION}|{ALGORITHM}|{key_id}".encode("utf-8")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))
