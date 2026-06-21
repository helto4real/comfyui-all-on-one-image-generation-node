import json

import pytest

from services import privacy


pytestmark = pytest.mark.skipif(not privacy.CRYPTO_AVAILABLE, reason="cryptography is not installed")


def test_privacy_envelope_round_trips_without_cleartext(tmp_path):
    state = {"value": "secret prompt", "nested": {"prompt": "second secret"}}

    envelope = privacy.encrypt_state(state, base_dir=tmp_path)
    dumped = json.dumps(envelope)

    assert envelope["schema"] == privacy.ENVELOPE_SCHEMA
    assert "secret prompt" not in dumped
    assert "second secret" not in dumped
    assert privacy.decrypt_state(envelope, base_dir=tmp_path) == state


def test_privacy_rejects_plain_payload(tmp_path):
    with pytest.raises(privacy.PrivacyError, match="not an encrypted privacy payload"):
        privacy.decrypt_state({"value": "plain"}, base_dir=tmp_path)


def test_privacy_rejects_missing_key(tmp_path):
    envelope = privacy.encrypt_state({"value": "secret"}, base_dir=tmp_path)
    missing_key_dir = tmp_path / "other"

    with pytest.raises(privacy.PrivacyError, match="key file is missing"):
        privacy.decrypt_state(envelope, base_dir=missing_key_dir)


def test_decrypt_text_if_encrypted_round_trips(monkeypatch, tmp_path):
    monkeypatch.setattr(privacy, "config_dir", lambda: tmp_path)
    envelope = json.dumps(privacy.encrypt_state({"value": "private prompt"}, base_dir=tmp_path))

    assert privacy.decrypt_text_if_encrypted(envelope) == "private prompt"
    assert privacy.decrypt_text_if_encrypted("plain prompt") == "plain prompt"
