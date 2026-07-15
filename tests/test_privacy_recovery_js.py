import json
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_privacy_browser_core_is_removed():
    assert not (ROOT / "web/js/aio_privacy.js").exists()
    assert not (ROOT / "web/js/aio_privacy_recovery.js").exists()


def test_product_frontends_do_not_reintroduce_privacy_authority():
    sources = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "web/js/aio_image_generate.js",
            "web/js/aio_ideogram4_prompt_builder.js",
        )
    )
    for forbidden in (
        "encryptValueSync",
        "decryptValue",
        "registerAioPrivacyRecoveryDescriptors",
        "appendPrivacyRecoveryMenuOption",
        "privacyFetchHeaders",
        "scheduleAioPrivacyGraphToPromptPatch",
        "setPrivacyRevealSource",
        "hover to reveal",
    ):
        assert forbidden not in sources


def test_managed_activation_uses_attested_digest_routed_runtime():
    source = (ROOT / "web/js/aio_managed_privacy.js").read_text(encoding="utf-8")

    assert 'fetch("/helto_privacy/status"' in source
    assert 'from "/helto_privacy/ui/privacy_snapshot.js"' in source
    assert "installPrivacyConnectionSerializationGate(app)" in source
    assert "activationGate.markUnavailable()" in source
    assert "activationGate.coalesce()" in source
    assert 'status?.suiteStatus !== "active"' in source
    assert "/helto_privacy/ui/privacy_profile/${suiteManifestDigest}.js" in source
    assert "runtime.connectPrivacyPack" in source
    assert 'packId: AIO_PRIVACY_PROFILE_ID' in source
    assert 'profileFingerprint: AIO_PRIVACY_PROFILE_FINGERPRINT' in source
    assert '"generate-workflow-browser"' in source
    assert '"krea-workflow-browser"' in source
    assert '"ideogram-builder-workflow-browser"' in source
    assert "PENDING_PROFILE_FINGERPRINT" not in source


def _activation_fixture(tmp_path: Path, status: dict[str, object]) -> Path:
    app_path = tmp_path / "app.mjs"
    gate_path = tmp_path / "gate.mjs"
    runtime_path = tmp_path / "runtime.mjs"
    activation_path = tmp_path / "activation.mjs"
    script_path = tmp_path / "test.mjs"
    app_path.write_text(
        "export const app = { registerExtension() {} };\n",
        encoding="utf-8",
    )
    gate_path.write_text(
        textwrap.dedent(
            """
            export const state = { unavailable: 0, coalesced: 0 };
            export function installPrivacyConnectionSerializationGate() {
              return {
                markUnavailable() { state.unavailable += 1; },
                coalesce() { state.coalesced += 1; },
              };
            }
            """
        ),
        encoding="utf-8",
    )
    runtime_path.write_text(
        "export async function connectPrivacyPack() { throw new Error('UNEXPECTED_CONNECT'); }\n",
        encoding="utf-8",
    )
    source = (ROOT / "web/js/aio_managed_privacy.js").read_text(encoding="utf-8")
    replacements = {
        '"/scripts/app.js"': json.dumps(app_path.as_uri()),
        '"/helto_privacy/ui/privacy_snapshot.js"': json.dumps(gate_path.as_uri()),
        '"./aio_managed_builder_privacy.js"': json.dumps(
            (ROOT / "web/js/aio_managed_builder_privacy.js").as_uri()
        ),
        '"./aio_managed_prompt_library_privacy.js"': json.dumps(
            (ROOT / "web/js/aio_managed_prompt_library_privacy.js").as_uri()
        ),
        '"./aio_managed_prompt_privacy.js"': json.dumps(
            (ROOT / "web/js/aio_managed_prompt_privacy.js").as_uri()
        ),
        "`/helto_privacy/ui/privacy_profile/${suiteManifestDigest}.js`": json.dumps(
            runtime_path.as_uri()
        ),
    }
    for original, replacement in replacements.items():
        source = source.replace(original, replacement)
    activation_path.write_text(source, encoding="utf-8")
    script_path.write_text(
        textwrap.dedent(
            f"""
            import assert from "node:assert/strict";
            import {{ state }} from {gate_path.as_uri()!r};
            globalThis.fetch = async () => ({{
              ok: true,
              json: async () => ({json.dumps(status)}),
            }});
            const activation = await import({activation_path.as_uri()!r});
            await assert.rejects(activation.aioPrivacy, /PRIVACY_SUITE_BLOCKED/);
            assert.equal(state.unavailable, 1);
            assert.equal(state.coalesced, 0);
            """
        ),
        encoding="utf-8",
    )
    return script_path


def test_inactive_or_malformed_suite_closes_bootstrap_gate(tmp_path):
    statuses = (
        {"suiteStatus": "cutover-pending", "suiteManifestDigest": "a" * 64},
        {"suiteStatus": "active", "suiteManifestDigest": "malformed"},
    )
    for index, status in enumerate(statuses):
        case = tmp_path / str(index)
        case.mkdir()
        result = subprocess.run(
            ["node", "--experimental-default-type=module", str(_activation_fixture(case, status))],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr + result.stdout


def test_managed_adapters_bind_edits_and_fail_closed_when_locked():
    prompt = (ROOT / "web/js/aio_managed_prompt_privacy.js").read_text(
        encoding="utf-8"
    )
    builder = (ROOT / "web/js/aio_managed_builder_privacy.js").read_text(
        encoding="utf-8"
    )

    assert "workflowHandle.markEdited(node, fieldId)" in prompt
    assert 'setDomText(target, locked ? "" : plaintext)' in prompt
    assert "node.__aioManagedPrivacyLocked = locked" in prompt
    assert "writeWorkflowProjection(node, serializedNode, protectedValue, context)" in prompt
    assert "workflowHandle.markEdited(node, fieldId)" in builder
    assert "editorApi(node).clearManagedState()" in builder
    assert "node.__aioManagedPrivacyLocked = locked" in builder
    assert "writeWorkflowProjection(node, serializedNode, protectedValue, context)" in builder
