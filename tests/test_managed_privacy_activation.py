from __future__ import annotations

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

import helto_privacy.migration as migration
import helto_privacy.runtime as runtime

from services import managed_prompt_privacy as managed


DECLARED_SHARED_PRIVACY_REQUIREMENT = "helto-privacy==0.4.5"


def test_production_adapter_builder_binds_the_exact_profile_set(tmp_path):
    profile = managed.build_aio_prompt_privacy_profile()
    adapters = managed.build_aio_prompt_server_adapters(
        prompt_library_base_dir=str(tmp_path),
    )

    assert set(adapters) == {slot.id for slot in profile.server_adapters}
    assert profile.fingerprint == managed.AIO_PRIVACY_PROFILE_FINGERPRINT


def test_atomic_install_refuses_a_missing_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_INSTALLATIONS", {})
    monkeypatch.setattr(runtime, "register_helto_privacy_ui", lambda **_kwargs: True)
    monkeypatch.setattr(managed, "_PACK", None)
    monkeypatch.setattr(managed, "_ADAPTERS", None)
    migration.reset_migration_runtime_for_tests()
    complete = managed.build_aio_prompt_server_adapters(
        prompt_library_base_dir=str(tmp_path),
    )
    complete.pop(next(iter(complete)))
    monkeypatch.setattr(
        managed,
        "build_aio_prompt_server_adapters",
        lambda **_kwargs: complete,
    )

    with pytest.raises(RuntimeError, match="adapter binding is incomplete"):
        managed.install_aio_privacy(tmp_path)

    assert managed._PACK is None
    assert managed._ADAPTERS is None


def test_package_bootstrap_has_no_local_privacy_or_library_route_registration():
    root = Path(__file__).resolve().parents[1]
    source = (root / "__init__.py").read_text(encoding="utf-8")

    assert "register_privacy_routes" not in source
    assert "register_ideogram4_prompt_library_routes" not in source
    assert "install_aio_privacy(_PACKAGE_ROOT)" in source
    assert "_register_safely(\"shared privacy" not in source
    assert not (root / "routes" / "privacy.py").exists()


def test_missing_shared_privacy_dependency_blocks_package_import():
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        import builtins
        import importlib.util
        import sys

        real_import = builtins.__import__
        def blocked_import(name, *args, **kwargs):
            if name == "helto_privacy" or name.startswith("helto_privacy."):
                raise ModuleNotFoundError("synthetic missing helto-privacy")
            return real_import(name, *args, **kwargs)
        builtins.__import__ = blocked_import

        root = {str(root)!r}
        spec = importlib.util.spec_from_file_location(
            "aio_missing_privacy_testpack",
            root + "/__init__.py",
            submodule_search_locations=[root],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "synthetic missing helto-privacy" in result.stderr


def test_distribution_metadata_is_aligned_and_packages_browser_entrypoint():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = tuple(
        line.strip()
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    assert project["project"]["name"] == root.name
    assert project["project"]["dependencies"] == [DECLARED_SHARED_PRIVACY_REQUIREMENT]
    assert requirements == (DECLARED_SHARED_PRIVACY_REQUIREMENT,)
    assert project["project"]["version"] == "0.1.7"
    assert managed.AIO_SUITE_ID == "helto-suite-2026-07-17.3"
    assert project["project"]["readme"] == "README.md"
    assert project["project"]["urls"]["Repository"] == (
        "https://github.com/helto4real/comfyui-all-on-one-image-generation-node"
    )
    assert project["tool"]["comfy"] == {
        "PublisherId": "helto",
        "DisplayName": "AIO Image Generate",
        "Icon": "",
        "web": "web",
    }
    assert all(
        marker not in DECLARED_SHARED_PRIVACY_REQUIREMENT
        for marker in ("file:", "/home/", "@main", "@master", "git+")
    )
    assert (root / "web/js/aio_managed_privacy.js").is_file()
    managed_privacy = (root / "web/js/aio_managed_privacy.js").read_text(encoding="utf-8")
    assert "await pack.readiness.waitUntilReady();" in managed_privacy
    assert managed_privacy.index("await pack.readiness.waitUntilReady();") < managed_privacy.index("pack.authorization.requireReady();")
    assert 'import "./aio_managed_privacy.js"' in (
        root / "web/js/aio_image_generate.js"
    ).read_text(encoding="utf-8")
