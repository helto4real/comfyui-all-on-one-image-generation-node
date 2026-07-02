import pytest

import helto_privacy.keystore as hp_keystore
from services import privacy


@pytest.fixture(autouse=True)
def isolated_privacy(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("privacy")
    monkeypatch.setenv(hp_keystore.KEYSTORE_ENV, str(root / "privacy_keystore.json"))
    monkeypatch.setenv(hp_keystore.SESSION_DIR_ENV, str(root / "session"))
    monkeypatch.setattr(hp_keystore, "SCRYPT_N", 2**12, raising=False)
    monkeypatch.setattr(privacy, "config_dir", lambda: root / "legacy_config")
