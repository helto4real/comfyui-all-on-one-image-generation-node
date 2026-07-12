import pytest

import helto_privacy.envelope as hp_envelope
import helto_privacy.guard as hp_guard
import helto_privacy.keystore as hp_keystore
import helto_privacy.suite_runtime as hp_suite_runtime
from services import lora_config
from services import privacy


@pytest.fixture(autouse=True)
def isolated_privacy(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("privacy")
    monkeypatch.setenv(hp_keystore.KEYSTORE_ENV, str(root / "privacy_keystore.json"))
    monkeypatch.setenv(hp_keystore.SESSION_DIR_ENV, str(root / "session"))
    monkeypatch.setattr(hp_keystore, "SCRYPT_N", 2**12, raising=False)
    monkeypatch.setattr(hp_suite_runtime, "require_active_process_suite", lambda: None)
    monkeypatch.setattr(hp_keystore, "require_active_process_suite", lambda: None)
    monkeypatch.setattr(hp_envelope, "require_active_process_suite", lambda: None)
    monkeypatch.setattr(hp_guard, "require_active_process_suite", lambda: None)
    monkeypatch.setattr(privacy, "config_dir", lambda: root / "legacy_config")


@pytest.fixture(autouse=True)
def isolated_lora_lookup(monkeypatch):
    monkeypatch.setattr(lora_config, "_available_loras", lambda: None)
