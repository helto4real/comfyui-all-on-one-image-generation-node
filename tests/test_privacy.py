import json

import pytest

import helto_privacy.keystore as keystore
from helto_privacy import initialize_keystore
from helto_privacy.guard import check_privacy_token
from services import privacy


pytestmark = pytest.mark.skipif(not privacy.CRYPTO_AVAILABLE, reason="cryptography is not installed")

PASSWORD = "correct horse battery"


def _init_keystore():
    return initialize_keystore(PASSWORD)


def test_privacy_envelope_round_trips_without_cleartext():
    _init_keystore()
    state = {"value": "secret prompt", "nested": {"prompt": "second secret"}}

    envelope = privacy.encrypt_state(state)
    dumped = json.dumps(envelope)

    assert envelope["schema"] == privacy.ENVELOPE_SCHEMA
    assert envelope["schema"] == "helto.aio-image-generate.v2"
    assert "secret prompt" not in dumped
    assert "second secret" not in dumped
    assert privacy.decrypt_state(envelope) == state


def test_privacy_envelope_hides_prompt_builder_state():
    _init_keystore()
    state = {
        "widgets": {
            "high_level_description": "private overview",
            "background": "private background",
            "import_json": '{"private":"import"}',
        },
        "elements": [
            {
                "type": "text",
                "text": "PRIVATE SIGN",
                "desc": "private element description",
                "palette": ["#ABCDEF"],
            }
        ],
        "style_palette": ["#123456"],
    }

    envelope = privacy.encrypt_state(state)
    dumped = json.dumps(envelope)

    assert "private overview" not in dumped
    assert "private background" not in dumped
    assert "private element description" not in dumped
    assert "PRIVATE SIGN" not in dumped
    assert "#ABCDEF" not in dumped
    assert "#123456" not in dumped
    assert privacy.decrypt_state(envelope) == state


def test_privacy_rejects_plain_payload():
    _init_keystore()
    with pytest.raises(privacy.PrivacyError, match="not an encrypted privacy payload"):
        privacy.decrypt_state({"value": "plain"})


def test_privacy_requires_initialized_keystore():
    with pytest.raises(privacy.PrivacyError, match="PRIVACY_KEYSTORE_UNINITIALIZED"):
        privacy.encrypt_state({"value": "secret"})


def test_privacy_rejects_locked_keystore():
    _init_keystore()
    envelope = privacy.encrypt_state({"value": "secret"})
    keystore.lock_keystore()

    with pytest.raises(privacy.PrivacyError, match="PRIVACY_LOCKED"):
        privacy.decrypt_state(envelope)


def test_privacy_rejects_legacy_schema():
    _init_keystore()
    envelope = privacy.encrypt_state({"value": "secret"})
    envelope["schema"] = privacy.LEGACY_ENVELOPE_SCHEMA

    with pytest.raises(privacy.PrivacyError, match="Unsupported legacy AIO privacy payload"):
        privacy.decrypt_state(envelope)


def test_decrypt_text_if_encrypted_round_trips():
    _init_keystore()
    envelope = json.dumps(privacy.encrypt_state({"value": "private prompt"}))

    assert privacy.decrypt_text_if_encrypted(envelope) == "private prompt"
    assert privacy.decrypt_text_if_encrypted("plain prompt") == "plain prompt"


class _FakeRequest:
    def __init__(self, header_token=None, cookie_token=None):
        self.headers = {}
        self.cookies = {}
        if header_token is not None:
            self.headers["X-Helto-Privacy-Token"] = header_token
        if cookie_token is not None:
            self.cookies["helto_privacy_token"] = cookie_token


def test_privacy_token_guard_accepts_header_or_cookie():
    result = _init_keystore()
    token = result["token"]

    assert check_privacy_token(_FakeRequest(header_token=token)) is None
    assert check_privacy_token(_FakeRequest(cookie_token=token)) is None
    assert check_privacy_token(_FakeRequest())["status"] == 401
