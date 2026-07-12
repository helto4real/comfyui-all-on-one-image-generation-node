import subprocess
import textwrap
from pathlib import Path

import helto_privacy
import pytest


ROOT = Path(__file__).resolve().parents[1]
AIO_PRIVACY = ROOT / "web" / "js" / "aio_privacy.js"
AIO_RECOVERY = ROOT / "web" / "js" / "aio_privacy_recovery.js"
LOCAL_PRIVACY_UI = Path(helto_privacy.__file__).resolve().parent / "web" / "privacy_ui.js"
LOCAL_PRIVACY_WEB = LOCAL_PRIVACY_UI.parent


def _write_module_test(tmp_path: Path, body: str, *, shared_source: str | None = None) -> Path:
    if shared_source is None:
        if not LOCAL_PRIVACY_UI.exists():
            pytest.skip("local helto-privacy recovery UI is not available")
        shared_source = LOCAL_PRIVACY_UI.read_text(encoding="utf-8")
        for dependency in (
            "privacy_client.js",
            "privacy_records.js",
            "privacy_artifacts.js",
        ):
            source = LOCAL_PRIVACY_WEB / dependency
            if not source.exists():
                pytest.skip(f"local helto-privacy {dependency} is not available")
            (tmp_path / dependency).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        (tmp_path / "package.json").write_text('{"type":"module"}', encoding="utf-8")

    shared_path = tmp_path / "shared_privacy.mjs"
    shared_path.write_text(shared_source, encoding="utf-8")
    aio_privacy_path = tmp_path / "aio_privacy.mjs"
    aio_recovery_path = tmp_path / "aio_privacy_recovery.mjs"
    aio_privacy_path.write_text(
        AIO_PRIVACY.read_text(encoding="utf-8").replace(
            'const SHARED_PRIVACY_ROUTE = "/helto_privacy/ui/privacy.js";',
            f"const SHARED_PRIVACY_ROUTE = {shared_path.as_uri()!r};",
        ),
        encoding="utf-8",
    )
    aio_recovery_path.write_text(
        AIO_RECOVERY.read_text(encoding="utf-8").replace("./aio_privacy.js", "./aio_privacy.mjs"),
        encoding="utf-8",
    )
    script_path = tmp_path / "test.mjs"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import assert from "node:assert/strict";
            import * as shared from {shared_path.as_uri()!r};
            import * as aioPrivacy from {aio_privacy_path.as_uri()!r};
            import * as recovery from {aio_recovery_path.as_uri()!r};

            function envelope(schema = aioPrivacy.PRIVACY_SCHEMA, extra = {{}}) {{
              return {{
                version: 1,
                encrypted: true,
                algorithm: "AES-256-GCM",
                schema,
                keyId: "key",
                nonce: "nonce",
                ciphertext: "ciphertext",
                ...extra,
              }};
            }}

            function node(type, widgets = [], properties = {{}}, extra = {{}}) {{
              return {{
                id: extra.id ?? 7,
                type,
                title: extra.title ?? type,
                widgets,
                properties,
                setDirtyCanvas() {{ this.dirty = true; }},
                ...extra,
              }};
            }}

            function widget(name, value) {{
              return {{ name, value }};
            }}

            {textwrap.dedent(body)}
            """
        ),
        encoding="utf-8",
    )
    return script_path


def _run_node_module_test(tmp_path: Path, body: str, *, shared_source: str | None = None) -> None:
    script_path = _write_module_test(tmp_path, body, shared_source=shared_source)
    result = subprocess.run(
        ["node", str(script_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_recovery_descriptors_cover_aio_private_controls(tmp_path):
    _run_node_module_test(
        tmp_path,
        """
        const registration = await recovery.registerAioPrivacyRecoveryDescriptors();
        const descriptors = shared.registeredPrivacyRecoveryDescriptors();
        const localDescriptors = recovery.aioPrivacyRecoveryDescriptors();

        assert.equal(registration.descriptorCount, 3);
        assert.deepEqual(descriptors.map((item) => item.id).sort(), [
          "aio-image-generate:krea-inpaint-prompt",
          "aio-image-generate:main-prompts",
          "aio-image-generate:prompt-builder",
        ].sort());

        const fieldsById = new Map(localDescriptors.map((descriptor) => [
          descriptor.id,
          descriptor.fields.map((field) => field.name),
        ]));
        assert.deepEqual(fieldsById.get("aio-image-generate:main-prompts"), ["positive_prompt", "negative_prompt"]);
        assert.deepEqual(fieldsById.get("aio-image-generate:krea-inpaint-prompt"), ["inpaint_positive_prompt"]);
        assert(fieldsById.get("aio-image-generate:prompt-builder").includes("aio_ideogram4_prompt_builder_state"));
        assert(fieldsById.get("aio-image-generate:prompt-builder").includes("elements_data"));
        """,
    )


def test_recovery_scan_detects_unsafe_values_without_leaking_payloads(tmp_path):
    _run_node_module_test(
        tmp_path,
        """
        await recovery.registerAioPrivacyRecoveryDescriptors();
        const graph = { nodes: [
          node("AIOImageGenerate", [
            widget("privacy_mode", true),
            widget("positive_prompt", JSON.stringify(envelope("helto.aio-image-generate", { ciphertext: "OLD_SECRET" }))),
            widget("negative_prompt", JSON.stringify(envelope("wrong.schema", { ciphertext: "WRONG_SECRET" }))),
          ]),
          node("AIOIdeogram4PromptBuilder", [
            widget("privacy_mode", true),
            widget("high_level_description", "PLAIN_PRIVATE_TEXT"),
          ], {
            aio_ideogram4_prompt_builder_state: JSON.stringify(envelope("wrong.schema", { ciphertext: "STATE_SECRET" })),
          }),
        ] };

        const issues = shared.scanPrivacyRecoveryIssues(graph);
        const types = issues.map((issue) => issue.type).sort();
        assert.deepEqual(types, [
          "invalid_encrypted_value",
          "invalid_encrypted_value",
          "invalid_encrypted_value",
          "plaintext_sensitive_value",
        ].sort());

        const publicIssues = JSON.stringify(issues);
        const model = JSON.stringify(shared.buildPrivacyRecoveryDialogModel(issues));
        for (const secret of ["OLD_SECRET", "WRONG_SECRET", "STATE_SECRET", "PLAIN_PRIVATE_TEXT"]) {
          assert(!publicIssues.includes(secret));
          assert(!model.includes(secret));
        }
        """,
    )


def test_same_schema_decrypt_failure_becomes_resettable_recovery_issue(tmp_path):
    _run_node_module_test(
        tmp_path,
        """
        await recovery.registerAioPrivacyRecoveryDescriptors();
        const failedEnvelope = envelope();
        const graphNode = node("AIOImageGenerate", [
          widget("privacy_mode", true),
          widget("positive_prompt", JSON.stringify(failedEnvelope)),
          widget("negative_prompt", ""),
        ]);

        assert.equal(shared.scanPrivacyRecoveryIssues({ nodes: [graphNode] }).length, 0);
        aioPrivacy.rememberFailedPrivacyEnvelope(failedEnvelope);

        const issues = shared.scanPrivacyRecoveryIssues({ nodes: [graphNode] });
        assert.equal(issues.length, 1);
        assert.equal(issues[0].type, "invalid_encrypted_value");
        assert.equal(issues[0].canReset, true);
        """,
    )


def test_recovery_reset_applies_defaults_and_clears_runtime_state(tmp_path):
    _run_node_module_test(
        tmp_path,
        """
        await recovery.registerAioPrivacyRecoveryDescriptors();
        const promptWidget = widget("positive_prompt", JSON.stringify(envelope("wrong.schema")));
        promptWidget.__aioPrivacyEnvelopeMemo = { plaintext: "secret", envelope: "old" };
        const graphNode = node("AIOImageGenerate", [
          widget("privacy_mode", true),
          promptWidget,
          widget("negative_prompt", ""),
        ], {}, { _aioPrivacyStatus: "locked" });

        const result = await shared.recoverPrivacyIssues({ action: "reset", graph: { nodes: [graphNode] } });

        assert.equal(result.ok, true);
        assert.equal(promptWidget.value, recovery.DEFAULT_GENERATE_PROMPT);
        assert.equal("__aioPrivacyEnvelopeMemo" in promptWidget, false);
        assert.equal("_aioPrivacyStatus" in graphNode, false);
        assert.equal(graphNode.dirty, true);
        """,
    )


def test_prompt_builder_state_reset_clears_property_and_live_runtime(tmp_path):
    _run_node_module_test(
        tmp_path,
        """
        await recovery.registerAioPrivacyRecoveryDescriptors();
        let resetCalled = false;
        const graphNode = node("AIOIdeogram4PromptBuilder", [
          widget("privacy_mode", true),
          widget("high_level_description", ""),
        ], {
          aio_ideogram4_prompt_builder_state: JSON.stringify(envelope("wrong.schema")),
        }, {
          _aioIdeogram4LastPrivatePayload: envelope("wrong.schema"),
          _aioIdeogram4PendingWorkflowInfo: {
            aio_ideogram4_prompt_builder: envelope("wrong.schema"),
            ideo: envelope("wrong.schema"),
          },
          _aioIdeogram4RecoveryReset() { resetCalled = true; },
        });

        const result = await shared.recoverPrivacyIssues({ action: "reset", graph: { nodes: [graphNode] } });

        assert.equal(result.ok, true);
        assert.equal(resetCalled, true);
        assert.equal("aio_ideogram4_prompt_builder_state" in graphNode.properties, false);
        assert.equal("_aioIdeogram4LastPrivatePayload" in graphNode, false);
        assert.equal("aio_ideogram4_prompt_builder" in graphNode._aioIdeogram4PendingWorkflowInfo, false);
        assert.equal("ideo" in graphNode._aioIdeogram4PendingWorkflowInfo, false);
        """,
    )


def test_plaintext_private_value_reencrypts_through_aio_route(tmp_path):
    _run_node_module_test(
        tmp_path,
        """
        await recovery.registerAioPrivacyRecoveryDescriptors();
        const graphNode = node("AIOImageGenerate", [
          widget("privacy_mode", true),
          widget("positive_prompt", "PLAIN_SECRET"),
          widget("negative_prompt", ""),
        ]);
        let captured = null;
        globalThis.fetch = async (url, options) => {
          assert(String(url).endsWith("/aio_image_generate/privacy/encrypt"));
          captured = JSON.parse(options.body).state.value;
          return {
            ok: true,
            status: 200,
            statusText: "OK",
            text: async () => JSON.stringify({ ok: true, envelope: envelope() }),
          };
        };

        const result = await shared.recoverPrivacyIssues({ action: "reencrypt", graph: { nodes: [graphNode] } });

        assert.equal(result.ok, true);
        assert.equal(captured, "PLAIN_SECRET");
        assert.equal(JSON.parse(graphNode.widgets[1].value).schema, aioPrivacy.PRIVACY_SCHEMA);
        """,
    )


def test_aio_async_encrypt_unlocks_and_retries_token_errors(tmp_path):
    const_shared = """
    export function isPrivacyUnlockRequiredError(error) {
      return String(error?.message ?? error ?? "").includes("PRIVACY_TOKEN_REQUIRED");
    }
    export async function showPrivacyKeystoreDialog() {
      globalThis.unlockCount = (globalThis.unlockCount || 0) + 1;
      return true;
    }
    export function ensureStoredPrivacyTokenCookie() {}
    """
    _run_node_module_test(
        tmp_path,
        """
        let calls = 0;
        globalThis.fetch = async () => {
          calls += 1;
          if (calls === 1) {
            return {
              ok: false,
              status: 401,
              statusText: "Locked",
              text: async () => JSON.stringify({ ok: false, error: "PRIVACY_TOKEN_REQUIRED: token required" }),
            };
          }
          return {
            ok: true,
            status: 200,
            statusText: "OK",
            text: async () => JSON.stringify({ ok: true, envelope: envelope() }),
          };
        };

        const encrypted = await aioPrivacy.encryptValue("secret after unlock");

        assert.equal(calls, 2);
        assert.equal(globalThis.unlockCount, 1);
        assert.equal(JSON.parse(encrypted).schema, aioPrivacy.PRIVACY_SCHEMA);
        """,
        shared_source=const_shared,
    )
