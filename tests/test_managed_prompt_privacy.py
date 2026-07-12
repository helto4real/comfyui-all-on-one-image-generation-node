from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import helto_privacy.execution as shared_execution
import helto_privacy.envelope as shared_envelope
import helto_privacy.guard as shared_guard
import helto_privacy.keystore as shared_keystore
import helto_privacy.migration as shared_migration
import helto_privacy.runtime as shared_runtime
import helto_privacy.suite_runtime as shared_suite_runtime
from helto_privacy import (
    AIO_V1_JSON_KEY_IMPORT_ID,
    AIO_V1_READER_ID,
    EffectivePrivacyMode,
    ExecutionError,
    LegacyKeyFormat,
    MigrationVerification,
    ModeEvidence,
    ModeFacts,
    PrivacyEnvelopeCodec,
    PrivacyAuthorizationError,
    SnapshotError,
    aio_v1_reader_unit,
    install,
    lock_keystore,
    register_legacy_reader_units,
)
from helto_privacy.guard import authorize_privacy_request

from nodes.aio_generate import AIOImageGenerate
from nodes.ideogram4_prompt_builder import AIOIdeogram4PromptBuilder
from nodes.krea2_settings import AIOKrea2Settings
from services import pipeline, privacy
from services.managed_prompt_privacy import (
    AIO_CURRENT_PROMPT_SCHEMA,
    GENERATE_EXECUTION_RESOURCE_ID,
    GENERATE_PROJECTION_ID,
    GENERATE_SCOPE_ID,
    GENERATE_WORKFLOW_RESOURCE_ID,
    KREA_INPAINT_PROMPT_FIELD_ID,
    KREA_SCOPE_ID,
    KREA_WORKFLOW_RESOURCE_ID,
    NEGATIVE_PROMPT_FIELD_ID,
    POSITIVE_PROMPT_FIELD_ID,
    AioPromptExecutionDispatchAdapter,
    AioPromptExecutionProjectionAdapter,
    AioPromptModeAdapter,
    AioPromptWorkflowStateAdapter,
    aio_prompt_legacy_binding_id,
    build_aio_prompt_privacy_profile,
    build_aio_prompt_server_adapters,
    resolve_execution_prompt_semantics,
)
from services.managed_builder_privacy import (
    AIO_BUILDER_CURRENT_SCHEMA,
    BUILDER_EXECUTION_FIELD_IDS,
    BUILDER_EXECUTION_RESOURCE_ID,
    BUILDER_LEGACY_WORKFLOW_STATE_KEY,
    BUILDER_PROJECTION_ID,
    BUILDER_SCOPE_ID,
    BUILDER_STATE_FIELD_ID,
    BUILDER_STATE_PROPERTY,
    BUILDER_WIDGET_FIELD_IDS,
    BUILDER_WORKFLOW_RESOURCE_ID,
    BUILDER_WORKFLOW_STATE_KEY,
    AioBuilderExecutionProjectionAdapter,
    AioBuilderMigrationTransaction,
    AioBuilderModeAdapter,
    builder_legacy_binding_id,
)
from services.prompt_resolution import resolve_prompt_source


FIXTURE = Path(__file__).parent / "fixtures" / "historical" / "aio_v1_prompt.json"
BUILDER_FIXTURE = Path(__file__).parent / "fixtures" / "historical" / "aio_v1_builder_state.json"


class Request:
    def __init__(self, token: str) -> None:
        self.headers = {"X-Helto-Privacy-Token": token}
        self.cookies = {}


def _authorization(pack, token: str, operation: str):
    return authorize_privacy_request(Request(token), operation, pack_id=pack.profile.id)


def _installed_pack(tmp_path, monkeypatch):
    monkeypatch.setenv(shared_migration.MIGRATION_STATE_ENV, str(tmp_path / "migration.json"))
    monkeypatch.setattr(shared_runtime, "_INSTALLATIONS", {})
    monkeypatch.setattr(shared_runtime, "register_helto_privacy_ui", lambda **_kwargs: True)
    monkeypatch.setattr(shared_suite_runtime, "require_active_process_suite", lambda: None)
    monkeypatch.setattr(shared_keystore, "require_active_process_suite", lambda: None)
    monkeypatch.setattr(shared_envelope, "require_active_process_suite", lambda: None)
    monkeypatch.setattr(shared_guard, "require_active_process_suite", lambda: None)
    monkeypatch.setattr(shared_keystore, "SCRYPT_N", 2**12)
    shared_migration.reset_migration_runtime_for_tests()
    shared_execution.invalidate_execution_session("test-reset")
    register_legacy_reader_units((aio_v1_reader_unit(),))
    pack = install(build_aio_prompt_privacy_profile(), build_aio_prompt_server_adapters())
    token = shared_keystore.initialize_keystore("synthetic AIO prompt password")["token"]
    return pack, token


def _write_historical_key(path: Path) -> None:
    key = hashlib.sha256(b"helto-aio-v1-historical-fixture-key").digest()
    encode = lambda value: base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "algorithm": "AES-256-GCM",
                "keyId": encode(hashlib.sha256(key).digest()[:12]),
                "key": encode(key),
            }
        ),
        encoding="utf-8",
    )


def _aio_v1_envelope(state: object, nonce_label: str) -> dict[str, object]:
    """Reproduce the pinned AIO v1 writer with synthetic deterministic inputs."""

    key = hashlib.sha256(b"helto-aio-v1-historical-fixture-key").digest()
    encode = lambda value: base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    key_id = encode(hashlib.sha256(key).digest()[:12])
    nonce = hashlib.sha256(f"aio-builder-{nonce_label}".encode()).digest()[:12]
    aad = f"helto.aio-image-generate|1|AES-256-GCM|{key_id}".encode()
    plaintext = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "version": 1,
        "schema": "helto.aio-image-generate",
        "encrypted": True,
        "algorithm": "AES-256-GCM",
        "keyId": key_id,
        "nonce": encode(nonce),
        "ciphertext": encode(AESGCM(key).encrypt(nonce, plaintext, aad)),
    }


def test_profile_declares_generate_krea_fields_floor_execution_and_legacy_units():
    profile = build_aio_prompt_privacy_profile()
    fields = {field.id: field for field in profile.protected_fields}
    scopes = {scope.id: scope for scope in profile.scopes}

    assert {
        POSITIVE_PROMPT_FIELD_ID,
        NEGATIVE_PROMPT_FIELD_ID,
        KREA_INPAINT_PROMPT_FIELD_ID,
    }.issubset(fields)
    assert fields[POSITIVE_PROMPT_FIELD_ID].workflow_resource_id == GENERATE_WORKFLOW_RESOURCE_ID
    assert all(field.execution for field in fields.values())
    assert all(field.legacy_reader_ids == (AIO_V1_READER_ID,) for field in fields.values())
    assert scopes[KREA_SCOPE_ID].floor_scope_ids == ()
    assert {item.id for item in profile.execution_projections} == {
        "generate", "ideogram-builder", "inpaint"
    }
    assert {item.id for item in profile.protected_operations} == {
        "generate", "ideogram-builder.build", "inpaint"
    }
    assert {item.import_id for item in profile.legacy_key_imports} == {AIO_V1_JSON_KEY_IMPORT_ID}
    assert len(profile.legacy_bindings) == 14
    assert len(profile.legacy_key_imports) == 14


def _builder_state(text: str = "synthetic builder") -> dict[str, object]:
    widgets = {name: f"{text} {name}" for name in BUILDER_WIDGET_FIELD_IDS}
    widgets.update(
        {
            "max side": 1024,
            "aspect ratio": "1:1",
            "multiple value": "none",
            "privacy_mode": True,
            "style": "none",
            "import_mode": "when empty",
            "output_format": "compact",
            "coord_mode": "normalized",
            "bbox_order": "yx",
            "bg_brightness": 25,
        }
    )
    widgets["style_palette_data"] = ""
    widgets["elements_data"] = ""
    widgets["import_json"] = ""
    return {
        "version": 1,
        "widgets": widgets,
        "effective_privacy_mode": bool(widgets["privacy_mode"]),
        "elements": [],
        "style_palette": [],
        "bg_brightness": 25,
        "output_format": "compact",
        "coord_mode": "normalized",
        "bbox_order": "yx",
        "active": -1,
    }


def test_builder_profile_declares_widgets_state_mirrors_mode_and_projection():
    profile = build_aio_prompt_privacy_profile()
    fields = {field.id: field for field in profile.protected_fields}
    state = fields[BUILDER_STATE_FIELD_ID]

    assert set(BUILDER_EXECUTION_FIELD_IDS).issubset(fields)
    assert state.workflow_resource_id == BUILDER_WORKFLOW_RESOURCE_ID
    assert state.location.name == BUILDER_STATE_PROPERTY
    assert {location.name for location in state.mirror_locations} == {
        BUILDER_WORKFLOW_STATE_KEY,
        BUILDER_LEGACY_WORKFLOW_STATE_KEY,
    }
    assert state.current_schema == AIO_BUILDER_CURRENT_SCHEMA
    assert next(scope for scope in profile.scopes if scope.id == BUILDER_SCOPE_ID).floor_scope_ids == ()
    assert AioBuilderModeAdapter().read_declared_mode(BUILDER_SCOPE_ID) == "inherit"
    assert AioBuilderModeAdapter({BUILDER_SCOPE_ID: False}).read_declared_mode(BUILDER_SCOPE_ID) == "public"


def test_builder_projection_requires_one_consistent_complete_generation():
    profile = build_aio_prompt_privacy_profile()
    declaration = next(
        item for item in profile.execution_projections if item.id == BUILDER_PROJECTION_ID
    )
    state = _builder_state()
    fields = {
        field_id: {"value": state["widgets"][widget_name]}
        for widget_name, field_id in BUILDER_WIDGET_FIELD_IDS.items()
    }
    fields[BUILDER_STATE_FIELD_ID] = state
    projection = AioBuilderExecutionProjectionAdapter()

    assert projection.project(fields, declaration) == state
    fields[BUILDER_WIDGET_FIELD_IDS["background"]] = {"value": "stale mirror"}
    with pytest.raises(ValueError, match="inconsistent"):
        projection.project(fields, declaration)
    del fields[BUILDER_WIDGET_FIELD_IDS["background"]]
    with pytest.raises(ValueError, match="incomplete"):
        projection.project(fields, declaration)


def test_builder_migration_staging_rejects_an_inconsistent_generation():
    state = _builder_state()
    fields = {
        field_id: {"value": state["widgets"][widget_name]}
        for widget_name, field_id in BUILDER_WIDGET_FIELD_IDS.items()
    }
    fields[BUILDER_STATE_FIELD_ID] = state
    fields[BUILDER_WIDGET_FIELD_IDS["background"]] = {"value": "stale mirror"}
    calls = []

    class Workflow:
        def protect(self, *args):
            calls.append(args)
            raise AssertionError("inconsistent fields must fail before encryption")

    transaction = AioBuilderMigrationTransaction(
        Workflow(), {}, object(), object(), True
    )
    with pytest.raises(ValueError, match="inconsistent"):
        transaction.stage_current(fields)
    assert calls == []


def test_builder_migration_readback_rejects_a_lost_effective_floor():
    state = _builder_state()
    fields = {
        field_id: {"value": state["widgets"][widget_name]}
        for widget_name, field_id in BUILDER_WIDGET_FIELD_IDS.items()
    }
    fields[BUILDER_STATE_FIELD_ID] = state

    class Result:
        def __init__(self, *, envelope=None, value=None):
            self.envelope = envelope
            self.value = value

    class Workflow:
        def protect(self, field_id, value, _authorization):
            return Result(
                envelope={
                    "field": field_id,
                    "value": value,
                    "schema": AIO_BUILDER_CURRENT_SCHEMA,
                }
            )

        def reveal(self, field_id, envelope, _authorization):
            value = json.loads(json.dumps(envelope["value"]))
            if field_id == BUILDER_STATE_FIELD_ID:
                value["effective_privacy_mode"] = False
            return Result(value=value)

    transaction = AioBuilderMigrationTransaction(
        Workflow(),
        {"widgets": {}, "properties": {}, "workflow": {}},
        object(),
        object(),
        True,
    )
    transaction.capture_original()
    transaction.stage_current(fields)
    transaction.commit()
    with pytest.raises(ValueError, match="effective mode was not preserved"):
        transaction.read_back()


def test_mode_mapping_defaults_private_and_krea_cannot_weaken_generate_floor(tmp_path, monkeypatch):
    pack, _token = _installed_pack(tmp_path, monkeypatch)
    mode = pack.mode("prompt-mode")

    generate = mode.resolve(GENERATE_SCOPE_ID)
    krea = mode.resolve(KREA_SCOPE_ID)
    explicit_public = mode.resolve_declaration(GENERATE_SCOPE_ID, False)
    explicit_private = mode.resolve_declaration(GENERATE_SCOPE_ID, True)
    all_public_krea = mode.resolve_declaration(
        KREA_SCOPE_ID,
        "public",
        ModeFacts(upstream=(ModeEvidence("aio-generate-1", "public"),)),
    )
    mixed_krea = mode.resolve_declaration(
        KREA_SCOPE_ID,
        "public",
        ModeFacts(
            upstream=(
                ModeEvidence("aio-generate-1", "public"),
                ModeEvidence("aio-generate-2", "private"),
            )
        ),
    )
    assert generate.effective is EffectivePrivacyMode.PRIVATE
    assert krea.effective is EffectivePrivacyMode.PRIVATE
    assert explicit_public.effective is EffectivePrivacyMode.PUBLIC
    assert explicit_private.effective is EffectivePrivacyMode.PRIVATE
    assert krea.floors == ()
    assert all_public_krea.effective is EffectivePrivacyMode.PUBLIC
    assert mixed_krea.effective is EffectivePrivacyMode.PRIVATE
    assert [floor.source_id for floor in mixed_krea.floors] == ["aio-generate-2"]

    public_pack_adapters = build_aio_prompt_server_adapters(
        declarations={GENERATE_SCOPE_ID: False}
    )
    adapter = public_pack_adapters["prompt-mode-state"]
    assert adapter.read_declared_mode(GENERATE_SCOPE_ID) == "public"
    assert adapter.read_declared_mode(KREA_SCOPE_ID) == "inherit"


def test_prompt_adapters_preserve_linked_empty_and_unlinked_workflow_recovery():
    profile = build_aio_prompt_privacy_profile()
    fields = {field.id: field for field in profile.protected_fields}
    state = AioPromptWorkflowStateAdapter()
    projection = AioPromptExecutionProjectionAdapter()

    assert resolve_prompt_source("", linked=True, workflow_value="stale local") == ""
    assert resolve_prompt_source("", linked=False, workflow_value="recovered local") == "recovered local"
    assert resolve_prompt_source("live", linked=False, workflow_value="stale local") == "live"
    assert resolve_prompt_source("", linked=False, workflow_value="[private prompt]") == ""
    assert state.normalize({"value": "  prompt  "}, fields[POSITIVE_PROMPT_FIELD_ID]) == {
        "value": "  prompt  "
    }
    generate_declaration = next(
        item for item in profile.execution_projections if item.id == GENERATE_PROJECTION_ID
    )
    assert projection.project(
        {
            POSITIVE_PROMPT_FIELD_ID: {"value": "positive"},
            NEGATIVE_PROMPT_FIELD_ID: {"value": "negative"},
        },
        generate_declaration,
    ) == {"positive_prompt": "positive", "negative_prompt": "negative"}


def test_shared_execution_has_one_semantic_identity_and_ram_cache(tmp_path, monkeypatch):
    pack, token = _installed_pack(tmp_path, monkeypatch)
    codec = PrivacyEnvelopeCodec(AIO_CURRENT_PROMPT_SCHEMA)
    protected = {
        POSITIVE_PROMPT_FIELD_ID: codec.encrypt_state({"value": "synthetic positive"}),
        NEGATIVE_PROMPT_FIELD_ID: codec.encrypt_state({"value": "synthetic negative"}),
    }
    execution = pack.execution(GENERATE_EXECUTION_RESOURCE_ID)
    calls = []
    context = {"dispatch": lambda value: calls.append(value) or {"image": "unchanged"}}

    first = execution.prepare(
        GENERATE_PROJECTION_ID,
        protected,
        _authorization(pack, token, "execution.prepare"),
    )
    first_result = execution.dispatch(first.reference, context)
    execution.cache_store(first_result.cache_identity, first_result.value)
    fresh_protected = {
        POSITIVE_PROMPT_FIELD_ID: codec.encrypt_state({"value": "synthetic positive"}),
        NEGATIVE_PROMPT_FIELD_ID: codec.encrypt_state({"value": "synthetic negative"}),
    }
    assert fresh_protected != protected
    second = execution.prepare(
        GENERATE_PROJECTION_ID,
        fresh_protected,
        _authorization(pack, token, "execution.prepare"),
    )
    second_result = execution.dispatch(second.reference, context)

    assert first_result.value == second_result.value == {"image": "unchanged"}
    assert first_result.cache_identity == second_result.cache_identity
    assert calls == [
        {"positive_prompt": "synthetic positive", "negative_prompt": "synthetic negative"}
    ]
    assert not list(tmp_path.rglob("*execution*cache*"))


def test_execution_grant_and_locked_reveal_fail_before_product(tmp_path, monkeypatch):
    pack, token = _installed_pack(tmp_path, monkeypatch)
    codec = PrivacyEnvelopeCodec(AIO_CURRENT_PROMPT_SCHEMA)
    protected = {
        POSITIVE_PROMPT_FIELD_ID: codec.encrypt_state({"value": "synthetic positive"}),
        NEGATIVE_PROMPT_FIELD_ID: codec.encrypt_state({"value": "synthetic negative"}),
    }
    execution = pack.execution(GENERATE_EXECUTION_RESOURCE_ID)
    calls = []
    prepared = execution.prepare(
        GENERATE_PROJECTION_ID,
        protected,
        _authorization(pack, token, "execution.prepare"),
    )
    tampered = dict(prepared.reference)
    tampered["grant"] = "invalid-grant"
    with pytest.raises(ExecutionError):
        execution.dispatch(tampered, {"dispatch": lambda value: calls.append(value)})

    reveal_authorization = _authorization(pack, token, "snapshot.reveal")
    lock_keystore()
    with pytest.raises(ExecutionError):
        execution.dispatch(prepared.reference, {"dispatch": lambda value: calls.append(value)})
    with pytest.raises((PrivacyAuthorizationError, SnapshotError)):
        pack.workflow(GENERATE_WORKFLOW_RESOURCE_ID).reveal(
            POSITIVE_PROMPT_FIELD_ID,
            protected[POSITIVE_PROMPT_FIELD_ID],
            reveal_authorization,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("field_id", "workflow_resource_id"),
    (
        (POSITIVE_PROMPT_FIELD_ID, GENERATE_WORKFLOW_RESOURCE_ID),
        (NEGATIVE_PROMPT_FIELD_ID, GENERATE_WORKFLOW_RESOURCE_ID),
        (KREA_INPAINT_PROMPT_FIELD_ID, KREA_WORKFLOW_RESOURCE_ID),
    ),
)
def test_genuine_aio_v1_fields_are_read_then_rewritten_as_current(
    tmp_path,
    monkeypatch,
    field_id,
    workflow_resource_id,
):
    pack, token = _installed_pack(tmp_path, monkeypatch)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    source = tmp_path / "privacy_key.json"
    _write_historical_key(source)
    pack.migration.import_legacy_key_source(
        AIO_V1_JSON_KEY_IMPORT_ID,
        source,
        "synthetic AIO prompt password",
        LegacyKeyFormat.JSON,
        _authorization(pack, token, "migration.key-import"),
    )
    token = shared_keystore.session_token()
    discovered = pack.migration.discover_and_read(
        aio_prompt_legacy_binding_id(field_id),
        fixture["envelope"],
        _authorization(pack, token, "migration.read"),
    )
    assert discovered.value == fixture["expectedNormalized"]

    workflow = pack.workflow(workflow_resource_id)

    class RewriteTransaction:
        def __init__(self):
            self.original = fixture["envelope"]
            self.current = None
            self.expected = None

        def capture_original(self):
            return self.original

        def stage_current(self, normalized):
            self.expected = normalized
            self.current = workflow.protect(
                field_id,
                normalized,
                _authorization(pack, token, "snapshot.protect"),
            ).envelope

        def stage_durable_adjuncts(self, _normalized):
            return None

        def commit(self):
            return None

        def read_back(self):
            revealed = workflow.reveal(
                field_id,
                self.current,
                _authorization(pack, token, "snapshot.reveal"),
            )
            return MigrationVerification(revealed.value, True, True)

        def rollback(self, original):
            self.original = original
            self.current = None

        def finalize(self, _original):
            self.original = None

    transaction = RewriteTransaction()
    receipt = pack.migration.complete(
        discovered.obligation.id,
        discovered.value,
        transaction,
        _authorization(pack, token, "migration.complete"),
    )
    assert receipt.disposition == "migrated"
    assert transaction.original is None
    assert transaction.current["schema"] == AIO_CURRENT_PROMPT_SCHEMA
    assert transaction.current["schema"] != fixture["envelope"]["schema"]


def test_dispatch_adapter_returns_existing_pipeline_result_unchanged():
    semantic = {"positive_prompt": "positive", "negative_prompt": "negative"}
    adapter = AioPromptExecutionDispatchAdapter()
    result = object()
    assert adapter.dispatch(semantic, {"dispatch": lambda value: result}, None) is result


def test_dispatch_semantics_use_evaluated_linked_inputs_over_local_snapshot():
    semantic = {"positive_prompt": "local positive", "negative_prompt": "local negative"}
    context = {
        "linked_inputs": {"positive_prompt": True, "negative_prompt": False},
        "prompt_inputs": {"positive_prompt": "linked positive"},
    }
    assert resolve_execution_prompt_semantics(semantic, context) == {
        "positive_prompt": "linked positive",
        "negative_prompt": "local negative",
    }
    with pytest.raises(ValueError):
        resolve_execution_prompt_semantics(
            semantic,
            {"linked_inputs": {"positive_prompt": True}, "prompt_inputs": {}},
        )


def test_generate_node_dispatches_private_reference_through_existing_pipeline(
    tmp_path,
    monkeypatch,
):
    from nodes import aio_generate

    pack, token = _installed_pack(tmp_path, monkeypatch)
    codec = PrivacyEnvelopeCodec(AIO_CURRENT_PROMPT_SCHEMA)
    prepared = pack.execution(GENERATE_EXECUTION_RESOURCE_ID).prepare(
        GENERATE_PROJECTION_ID,
        {
            POSITIVE_PROMPT_FIELD_ID: codec.encrypt_state({"value": "managed positive"}),
            NEGATIVE_PROMPT_FIELD_ID: codec.encrypt_state({"value": "managed negative"}),
        },
        _authorization(pack, token, "execution.prepare"),
    )
    captured = {}

    class FakeAdapter:
        version = "managed-test"

        def resolve_settings(self, **kwargs):
            return {
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": 1,
                "cfg": 1.0,
                "sampler": "auto",
                "scheduler": "auto",
            }

        def validate_inputs(self, **kwargs):
            captured["validated"] = kwargs
            return []

        def generate(self, **kwargs):
            captured["generated"] = kwargs
            return pipeline.GenerationResult(
                image="image",
                latent={"samples": "latent"},
                positive="positive",
                negative="negative",
                vae="vae",
                model=kwargs["loaded_model"],
                clip=kwargs["loaded_clip"],
            )

    monkeypatch.setattr(aio_generate, "get_adapter", lambda _model_type: FakeAdapter())
    result = AIOImageGenerate().generate(
        model_type="z_image_turbo",
        diffusion_model="model.safetensors",
        text_encoder="text.safetensors",
        vae="vae.safetensors",
        positive_prompt="untrusted raw positive",
        negative_prompt="untrusted raw negative",
        width=1024,
        height=1024,
        seed=0,
        steps=1,
        cfg=1.0,
        sampler="auto",
        scheduler="auto",
        model="model",
        clip="clip",
        private_execution=json.dumps(prepared.reference),
    )

    assert result[0] == "image"
    assert captured["generated"]["positive_prompt"] == "managed positive"
    assert captured["generated"]["negative_prompt"] == "managed negative"


def test_krea_node_dispatches_private_reference_into_settings(tmp_path, monkeypatch):
    pack, token = _installed_pack(tmp_path, monkeypatch)
    codec = PrivacyEnvelopeCodec(AIO_CURRENT_PROMPT_SCHEMA)
    prepared = pack.execution("krea-inpaint-execution").prepare(
        "inpaint",
        {
            KREA_INPAINT_PROMPT_FIELD_ID: codec.encrypt_state(
                {"value": "managed inpaint"}
            )
        },
        _authorization(pack, token, "execution.prepare"),
    )

    settings = AIOKrea2Settings().build_settings(
        enhancer_enabled=True,
        enhancer_strength=1.0,
        precision_policy="auto",
        inpaint_positive_prompt="untrusted raw inpaint",
        private_execution=json.dumps(prepared.reference),
    )[0]

    assert settings["positive_prompt_override"] == "managed inpaint"
    assert settings["positive_prompt_source"] == "krea2_inpaint_settings"


def test_builder_genuine_v1_fields_and_mirrors_rewrite_under_one_receipt(
    tmp_path,
    monkeypatch,
):
    pack, token = _installed_pack(tmp_path, monkeypatch)
    source = tmp_path / "privacy_key.json"
    _write_historical_key(source)
    pack.migration.import_legacy_key_source(
        AIO_V1_JSON_KEY_IMPORT_ID,
        source,
        "synthetic AIO prompt password",
        LegacyKeyFormat.JSON,
        _authorization(pack, token, "migration.key-import"),
    )
    token = shared_keystore.session_token()
    state = _builder_state("legacy")
    state["widgets"]["privacy_mode"] = False
    state.pop("effective_privacy_mode")  # AIO v1 predates the derived execution fact.
    expected = {
        field_id: {"value": state["widgets"][widget_name]}
        for widget_name, field_id in BUILDER_WIDGET_FIELD_IDS.items()
    }
    expected[BUILDER_STATE_FIELD_ID] = state
    legacy = {
        field_id: _aio_v1_envelope(value, field_id)
        for field_id, value in expected.items()
    }
    discovered = []
    for field_id in BUILDER_EXECUTION_FIELD_IDS:
        result = pack.migration.discover_and_read(
            builder_legacy_binding_id(field_id),
            legacy[field_id],
            _authorization(pack, token, "migration.read"),
        )
        discovered.append(result)
        assert result.value == expected[field_id]

    original_state = legacy[BUILDER_STATE_FIELD_ID]
    store = {
        "widgets": {
            name: legacy[field_id]
            for name, field_id in BUILDER_WIDGET_FIELD_IDS.items()
        },
        "properties": {BUILDER_STATE_PROPERTY: original_state},
        "workflow": {
            BUILDER_WORKFLOW_STATE_KEY: original_state,
            BUILDER_LEGACY_WORKFLOW_STATE_KEY: original_state,
        },
    }
    transaction = AioBuilderMigrationTransaction(
        pack.workflow(BUILDER_WORKFLOW_RESOURCE_ID),
        store,
        _authorization(pack, token, "snapshot.protect"),
        _authorization(pack, token, "snapshot.reveal"),
        True,
    )
    receipt = pack.migration.complete_many(
        [item.obligation.id for item in discovered],
        expected,
        transaction,
        _authorization(pack, token, "migration.complete"),
    )

    mirrors = (
        store["properties"][BUILDER_STATE_PROPERTY],
        store["workflow"][BUILDER_WORKFLOW_STATE_KEY],
        store["workflow"][BUILDER_LEGACY_WORKFLOW_STATE_KEY],
    )
    assert len(receipt.obligation_ids) == len(BUILDER_EXECUTION_FIELD_IDS)
    assert mirrors[0] == mirrors[1] == mirrors[2]
    assert mirrors[0]["schema"] == AIO_BUILDER_CURRENT_SCHEMA
    migrated_state = pack.workflow(BUILDER_WORKFLOW_RESOURCE_ID).reveal(
        BUILDER_STATE_FIELD_ID,
        mirrors[0],
        _authorization(pack, token, "snapshot.reveal"),
    ).value
    assert migrated_state["widgets"]["privacy_mode"] is False
    assert migrated_state["effective_privacy_mode"] is True
    assert all(
        envelope["schema"] == AIO_BUILDER_CURRENT_SCHEMA
        for envelope in store["widgets"].values()
    )
    assert json.dumps(store).find('"schema": "helto.aio-image-generate"') == -1


def test_committed_genuine_builder_fixture_still_uses_the_exact_aio_v1_reader(
    tmp_path,
    monkeypatch,
):
    pack, token = _installed_pack(tmp_path, monkeypatch)
    fixture = json.loads(BUILDER_FIXTURE.read_text(encoding="utf-8"))
    source = tmp_path / "privacy_key.json"
    _write_historical_key(source)
    pack.migration.import_legacy_key_source(
        AIO_V1_JSON_KEY_IMPORT_ID,
        source,
        "synthetic AIO prompt password",
        LegacyKeyFormat.JSON,
        _authorization(pack, token, "migration.key-import"),
    )
    token = shared_keystore.session_token()
    discovered = pack.migration.discover_and_read(
        builder_legacy_binding_id(BUILDER_STATE_FIELD_ID),
        fixture["envelope"],
        _authorization(pack, token, "migration.read"),
    )
    assert discovered.value == fixture["expectedNormalized"]


def test_builder_node_dispatches_managed_generation_through_product_builder(
    tmp_path,
    monkeypatch,
):
    pack, token = _installed_pack(tmp_path, monkeypatch)
    state = _builder_state("managed")
    state["widgets"].update(
        {
            "high_level_description": "Managed overview",
            "background": "Managed room",
            "style": "none",
            "privacy_mode": False,
            "style_palette_data": '["#fab387"]',
            "elements_data": json.dumps(
                [
                    {
                        "x": 0.1,
                        "y": 0.2,
                        "w": 0.3,
                        "h": 0.4,
                        "type": "obj",
                        "desc": "Managed subject",
                    }
                ]
            ),
            "coord_mode": "absolute",
            "bbox_order": "xy",
        }
    )
    state["effective_privacy_mode"] = True
    state.update(
        {
            "style_palette": ["#fab387"],
            "elements": json.loads(state["widgets"]["elements_data"]),
            "coord_mode": "absolute",
            "bbox_order": "xy",
        }
    )
    codec = PrivacyEnvelopeCodec(AIO_BUILDER_CURRENT_SCHEMA)
    protected = {
        field_id: codec.encrypt_state({"value": state["widgets"][widget_name]})
        for widget_name, field_id in BUILDER_WIDGET_FIELD_IDS.items()
    }
    protected[BUILDER_STATE_FIELD_ID] = codec.encrypt_state(state)
    execution = pack.execution(BUILDER_EXECUTION_RESOURCE_ID)
    prepared = execution.prepare(
        BUILDER_PROJECTION_ID,
        protected,
        _authorization(pack, token, "execution.prepare"),
    )
    second_prepared = execution.prepare(
        BUILDER_PROJECTION_ID,
        protected,
        _authorization(pack, token, "execution.prepare"),
    )
    monkeypatch.setattr(
        AIOIdeogram4PromptBuilder,
        "_render_preview",
        staticmethod(lambda _boxes, _width, _height, image, _brightness: image),
    )

    result = AIOIdeogram4PromptBuilder().build_prompt(
        background="untrusted raw background",
        image="preview-one",
        privacy_mode=False,
        private_execution=json.dumps(prepared.reference),
        **{"max side": 512, "aspect ratio": "1:1", "multiple value": "none"},
    )["result"]
    second_result = AIOIdeogram4PromptBuilder().build_prompt(
        background="another untrusted background",
        image="preview-two",
        privacy_mode=False,
        private_execution=json.dumps(second_prepared.reference),
        **{"max side": 512, "aspect ratio": "1:1", "multiple value": "none"},
    )["result"]

    assert result[2] == "preview-one"
    assert second_result[2] == "preview-two"
    assert result[0]["privacy_mode"] is True
    assert privacy.is_encrypted_payload(result[0]["prompt"])
    assert result[4:6] == (1024, 1024)
    assert "Managed overview" in result[1]
    assert "Managed room" in result[1]
    assert json.loads(result[1])["compositional_deconstruction"]["elements"][0]["bbox"] == [102, 205, 410, 614]
    assert "untrusted raw background" not in result[1]
