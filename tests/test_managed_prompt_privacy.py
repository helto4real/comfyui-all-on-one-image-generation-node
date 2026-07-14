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
    ProtectedStateAuthority,
    RecordSnapshot,
    SnapshotError,
    aio_v1_reader_unit,
    install,
    lock_keystore,
    confirm_record_mutation,
    generate_private_record_id,
    register_legacy_reader_units,
)
from helto_privacy.records import RecordError
from helto_privacy.guard import authorize_privacy_request

from nodes.aio_generate import AIOImageGenerate
from nodes.ideogram4_prompt_builder import AIOIdeogram4PromptBuilder
from nodes.krea2_settings import AIOKrea2Settings
from services import pipeline
from services.managed_prompt_privacy import (
    AIO_CURRENT_PROMPT_SCHEMA,
    AIO_PRIVACY_PROFILE_FINGERPRINT,
    GENERATE_EXECUTION_RESOURCE_ID,
    GENERATE_PROJECTION_ID,
    GENERATE_SCOPE_ID,
    GENERATE_SUBJECT_MODE_BINDING_ID,
    GENERATE_WORKFLOW_RESOURCE_ID,
    KREA_INPAINT_PROMPT_FIELD_ID,
    KREA_SCOPE_ID,
    KREA_SUBJECT_MODE_BINDING_ID,
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
from services.managed_run_info_privacy import (
    RUN_INFO_ADAPTER_ID,
    RUN_INFO_OPERATION_ID,
    build_managed_run_info_json,
)
from services.run_info import build_run_info_candidate, to_json
from services.managed_builder_privacy import (
    AIO_BUILDER_CURRENT_SCHEMA,
    BUILDER_EXECUTION_FIELD_IDS,
    BUILDER_EXECUTION_RESOURCE_ID,
    BUILDER_LEGACY_WORKFLOW_STATE_KEY,
    BUILDER_PROJECTION_ID,
    BUILDER_SCOPE_ID,
    BUILDER_SUBJECT_MODE_BINDING_ID,
    BUILDER_STATE_FIELD_ID,
    BUILDER_STATE_PROPERTY,
    BUILDER_WIDGET_FIELD_IDS,
    BUILDER_WORKFLOW_RESOURCE_ID,
    BUILDER_WORKFLOW_STATE_KEY,
    AioBuilderExecutionProjectionAdapter,
    AioBuilderMigrationTransaction,
    AioBuilderModeAdapter,
    AioBuilderWorkflowStateAdapter,
    builder_legacy_binding_id,
)
from services.managed_prompt_library_privacy import (
    MANAGED_LIBRARY_FILE_NAME,
    PROMPT_LIBRARY_CURRENT_SCHEMA,
    PROMPT_LIBRARY_LEGACY_BINDING_ID,
    PROMPT_LIBRARY_RESOURCE_ID,
    PROMPT_RECORD_KIND,
    AioPromptLibraryMigrationTransaction,
    AioPromptLibraryStoreAdapter,
    discover_legacy_prompt_record_sources,
    legacy_library_path,
    managed_library_path,
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


def _installed_pack(tmp_path, monkeypatch, *, declarations=None):
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
    pack = install(
        build_aio_prompt_privacy_profile(),
        build_aio_prompt_server_adapters(
            declarations=declarations,
            prompt_library_base_dir=str(tmp_path),
        ),
    )
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


def _prepared_subject(pack, token: str, binding_id: str, subject_id: str, declaration=True):
    return pack.subject_modes(binding_id).prepare(
        subject_id,
        declaration,
        ModeFacts(),
        _authorization(pack, token, "subject-mode.prepare"),
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
        "emit-run-info", "generate", "ideogram-builder.build", "inpaint"
    }
    assert {item.import_id for item in profile.legacy_key_imports} == {AIO_V1_JSON_KEY_IMPORT_ID}
    assert len(profile.legacy_bindings) == 15
    assert len(profile.legacy_key_imports) == 15

    run_info = next(
        item for item in profile.protected_operations
        if item.id == RUN_INFO_OPERATION_ID
    )
    assert run_info.scope_id == GENERATE_SCOPE_ID
    assert any(
        item.path == "*" and item.field_class.value == "consumer-derived"
        for item in run_info.sensitive_fields
    )
    assert {item.path for item in run_info.safe_projection} == {
        "performance.configured",
        "performance.duplicate_inpaint_reference_count",
        "performance.duplicate_inpaint_reference_skipped",
        "performance.fp16_accumulation_enabled",
        "performance.memory_cleanup_applied",
        "performance.resolved_fp16_accumulation_enabled",
        "performance.warning_count",
    }
    assert profile.server_adapter_contracts[RUN_INFO_ADAPTER_ID] == (
        "project",
    )
    assert profile.fingerprint == AIO_PRIVACY_PROFILE_FINGERPRINT
    assert {
        binding.id: binding.input_name for binding in profile.subject_mode_bindings
    } == {
        GENERATE_SUBJECT_MODE_BINDING_ID: "privacy_mode_reference",
        KREA_SUBJECT_MODE_BINDING_ID: "privacy_mode_reference",
        BUILDER_SUBJECT_MODE_BINDING_ID: "privacy_mode_reference",
    }
    projections = {item.id: item for item in profile.execution_projections}
    assert projections[GENERATE_PROJECTION_ID].subject_mode_binding_id == (
        GENERATE_SUBJECT_MODE_BINDING_ID
    )
    assert projections["inpaint"].subject_mode_binding_id == KREA_SUBJECT_MODE_BINDING_ID
    assert projections[BUILDER_PROJECTION_ID].subject_mode_binding_id == (
        BUILDER_SUBJECT_MODE_BINDING_ID
    )
    assert run_info.subject_mode_binding_id == GENERATE_SUBJECT_MODE_BINDING_ID


def _library_payload(prompt: str = "synthetic prompt") -> dict[str, object]:
    return {
        "family": "ideogram4",
        "version": 1,
        "state": {"widgets": {"aspect ratio": "1:1"}, "elements": []},
        "prompt": prompt,
    }


def test_prompt_library_profile_is_strict_private_and_product_normalized():
    profile = build_aio_prompt_privacy_profile()
    declaration = profile.records[0]

    assert declaration.id == PROMPT_RECORD_KIND
    assert declaration.resource_id == PROMPT_LIBRARY_RESOURCE_ID
    assert declaration.current_schema == PROMPT_LIBRARY_CURRENT_SCHEMA
    assert declaration.reveal_operations == ("details", "use")
    assert declaration.mutation_operations == ("create", "duplicate", "patch", "replace")
    assert declaration.safe_projection == ()
    assert declaration.fixed_private_label == "Private record"
    assert next(
        item for item in profile.legacy_bindings
        if item.id == PROMPT_LIBRARY_LEGACY_BINDING_ID
    ).location_kind.value == "record"


def _run_info_facts() -> dict[str, object]:
    return {
        "model_type": "synthetic-model-family",
        "display_name": "SYNTHETIC_PRIVATE_DISPLAY_NAME",
        "diffusion_model": "/SYNTHETIC/PRIVATE/diffusion.safetensors",
        "diffusion_model_format": "safetensors",
        "text_encoder": "/SYNTHETIC/PRIVATE/text.safetensors",
        "text_encoder_format": "safetensors",
        "vae": "/SYNTHETIC/PRIVATE/vae.safetensors",
        "vae_format": "safetensors",
        "width": 1024,
        "height": 1024,
        "seed": 123,
        "steps": 8,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "beta",
        "settings": {
            "positive_prompt_override": "SYNTHETIC_PRIVATE_PROMPT",
            "privacy_mode": False,
            "attention_mode": "auto",
            "resolved_attention_mode": "sage",
            "fp16_accumulation_enabled": True,
            "resolved_fp16_accumulation_enabled": False,
            "memory_policy": "balanced",
            "memory_cleanup_applied": True,
            "duplicate_inpaint_reference_skipped": True,
            "duplicate_inpaint_reference_count": 2,
            "performance_warnings": ["SYNTHETIC_PRIVATE_WARNING"],
        },
        "warnings": ["SYNTHETIC_PRIVATE_TOP_LEVEL_WARNING"],
        "adapter_version": "SYNTHETIC_PRIVATE_ADAPTER_VERSION",
        "loras": [{"name": "/SYNTHETIC/PRIVATE/style.safetensors"}],
        "debug": {"workflow": "SYNTHETIC_PRIVATE_WORKFLOW"},
    }


def test_private_run_info_releases_only_validated_coarse_performance_facts(
    tmp_path,
    monkeypatch,
    caplog,
):
    pack, token = _installed_pack(
        tmp_path,
        monkeypatch,
        declarations={GENERATE_SCOPE_ID: True},
    )
    facts = _run_info_facts()
    candidate = build_run_info_candidate(**facts)
    assert candidate["settings"]["positive_prompt_override"] == (
        "SYNTHETIC_PRIVATE_PROMPT"
    )

    subject = _prepared_subject(
        pack,
        token,
        GENERATE_SUBJECT_MODE_BINDING_ID,
        "run-info-private-1",
    )
    with pack.subject_modes(GENERATE_SUBJECT_MODE_BINDING_ID).consume(
        subject.reference,
        "run-info-private-1",
    ) as lease:
        dumped = build_managed_run_info_json(
            pack,
            subject_mode=lease,
            **facts,
        )
    projected = json.loads(dumped)

    assert projected == {
        "performance": {
            "configured": True,
            "duplicate_inpaint_reference_count": 2,
            "duplicate_inpaint_reference_skipped": True,
            "fp16_accumulation_enabled": True,
            "memory_cleanup_applied": True,
            "resolved_fp16_accumulation_enabled": False,
            "warning_count": 1,
        }
    }
    assert "SYNTHETIC_PRIVATE" not in dumped
    assert "SYNTHETIC_PRIVATE" not in caplog.text
    assert not any(
        key in projected
        for key in ("debug", "settings", "warnings", "loras", "diffusion_model")
    )


def test_private_run_info_rejects_unsafe_coarse_diagnostic_type(
    tmp_path,
    monkeypatch,
):
    from helto_privacy import ProtectedOperationError

    pack, token = _installed_pack(
        tmp_path,
        monkeypatch,
        declarations={GENERATE_SCOPE_ID: True},
    )
    candidate = build_run_info_candidate(**_run_info_facts())
    candidate["performance"]["configured"] = "SYNTHETIC_PRIVATE_BOOLEAN"

    subject = _prepared_subject(
        pack,
        token,
        GENERATE_SUBJECT_MODE_BINDING_ID,
        "run-info-invalid-1",
    )
    with pack.subject_modes(GENERATE_SUBJECT_MODE_BINDING_ID).consume(
        subject.reference,
        "run-info-invalid-1",
    ) as lease:
        with pytest.raises(ProtectedOperationError) as failed:
            pack.operations(GENERATE_WORKFLOW_RESOURCE_ID).project(
                RUN_INFO_OPERATION_ID,
                candidate,
                subject_mode=lease,
            )

    assert failed.value.code == "PRIVACY_PROTECTED_OPERATION_PROJECTION_INVALID"
    assert "SYNTHETIC" not in str(failed.value)
    assert "SYNTHETIC" not in repr(failed.value)


def test_public_run_info_preserves_the_existing_product_schema(
    tmp_path,
    monkeypatch,
):
    pack, token = _installed_pack(
        tmp_path,
        monkeypatch,
        declarations={GENERATE_SCOPE_ID: False},
    )
    facts = _run_info_facts()
    facts["settings"]["privacy_mode"] = True
    candidate = build_run_info_candidate(**facts)

    subject = _prepared_subject(
        pack,
        token,
        GENERATE_SUBJECT_MODE_BINDING_ID,
        "run-info-public-1",
        False,
    )
    with pack.subject_modes(GENERATE_SUBJECT_MODE_BINDING_ID).consume(
        subject.reference,
        "run-info-public-1",
    ) as lease:
        assert json.loads(
            build_managed_run_info_json(pack, subject_mode=lease, **facts)
        ) == json.loads(to_json(candidate))


def test_prompt_library_shared_crud_locked_shell_delete_and_no_metadata_leak(
    tmp_path,
    monkeypatch,
):
    pack, token = _installed_pack(tmp_path, monkeypatch)
    records = pack.records(PROMPT_LIBRARY_RESOURCE_ID)
    created = records.mutate(
        PROMPT_RECORD_KIND,
        "create",
        {
            "payload": _library_payload("private canary prompt"),
            "metadata": {
                "name": "private canary name",
                "description": "private canary description",
                "tags": ["private-canary-tag"],
            },
        },
        _authorization(pack, token, "record.create"),
    )
    record_id = created.record_id

    assert [shell.to_payload() for shell in records.list_shells(PROMPT_RECORD_KIND)] == [{
        "id": record_id,
        "kind": PROMPT_RECORD_KIND,
        "private": True,
        "label": "Private record",
    }]
    stored = managed_library_path(tmp_path).read_text(encoding="utf-8")
    assert MANAGED_LIBRARY_FILE_NAME in str(managed_library_path(tmp_path))
    for canary in (
        "private canary prompt",
        "private canary name",
        "private canary description",
        "private-canary-tag",
    ):
        assert canary not in stored

    details = records.reveal(
        PROMPT_RECORD_KIND,
        record_id,
        "details",
        _authorization(pack, token, "record.details"),
    ).value["record"]
    assert details["name"] == "private canary name"
    assert details["payload"] == _library_payload("private canary prompt")

    used = records.reveal(
        PROMPT_RECORD_KIND,
        record_id,
        "use",
        _authorization(pack, token, "record.use"),
    ).value["record"]
    assert used["last_used_at"]
    persisted_use = records.reveal(
        PROMPT_RECORD_KIND,
        record_id,
        "details",
        _authorization(pack, token, "record.details"),
    ).value["record"]
    assert persisted_use["last_used_at"] == used["last_used_at"]
    patched = records.mutate(
        PROMPT_RECORD_KIND,
        "patch",
        {"metadata": {"name": "patched name"}},
        _authorization(pack, token, "record.patch"),
        record_id=record_id,
    )
    assert patched.record_id == record_id
    assert records.reveal(
        PROMPT_RECORD_KIND,
        record_id,
        "details",
        _authorization(pack, token, "record.details"),
    ).value["record"]["name"] == "patched name"
    replaced = records.mutate(
        PROMPT_RECORD_KIND,
        "replace",
        {
            "payload": _library_payload("replacement prompt"),
            "metadata": {"name": "replacement name"},
        },
        _authorization(pack, token, "record.replace"),
        record_id=record_id,
    )
    assert replaced.record_id == record_id
    replaced_record = records.reveal(
        PROMPT_RECORD_KIND,
        record_id,
        "details",
        _authorization(pack, token, "record.details"),
    ).value["record"]
    assert replaced_record["name"] == "replacement name"
    assert replaced_record["payload"] == _library_payload("replacement prompt")
    assert replaced_record["last_used_at"] == used["last_used_at"]
    duplicated = records.mutate(
        PROMPT_RECORD_KIND,
        "duplicate",
        {"metadata": {}},
        _authorization(pack, token, "record.duplicate"),
        record_id=record_id,
    )
    assert duplicated.record_id != record_id

    lock_keystore()
    assert len(records.list_shells(PROMPT_RECORD_KIND)) == 2
    confirmation = confirm_record_mutation(
        pack_id=pack.profile.id,
        resource_id=PROMPT_LIBRARY_RESOURCE_ID,
        record_kind=PROMPT_RECORD_KIND,
        record_id=record_id,
        operation="delete",
        confirmed=True,
    )
    assert records.delete(PROMPT_RECORD_KIND, record_id, confirmation).operation == "delete"
    assert [shell.id for shell in records.list_shells(PROMPT_RECORD_KIND)] == [
        duplicated.record_id
    ]


def test_prompt_library_failed_decrypt_is_fail_closed(tmp_path, monkeypatch):
    pack, token = _installed_pack(tmp_path, monkeypatch)
    records = pack.records(PROMPT_LIBRARY_RESOURCE_ID)
    created = records.mutate(
        PROMPT_RECORD_KIND,
        "create",
        {"payload": _library_payload()},
        _authorization(pack, token, "record.create"),
    )
    document = json.loads(managed_library_path(tmp_path).read_text(encoding="utf-8"))
    document["records"][0]["protected"]["ciphertext"] = "corrupt"
    managed_library_path(tmp_path).write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RecordError) as exc:
        records.reveal(
            PROMPT_RECORD_KIND,
            created.record_id,
            "details",
            _authorization(pack, token, "record.details"),
        )
    assert exc.value.code == "PRIVACY_RECORD_DECRYPT_FAILED"


def test_prompt_library_genuine_v1_record_gets_verified_current_receipt(
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
    legacy = {
        "payload": _library_payload("legacy synthetic prompt"),
        "name": "legacy synthetic name",
        "description": "legacy synthetic description",
        "tags": ["legacy-synthetic-tag"],
        "created_at": "2026-01-02T03:04:05Z",
        "updated_at": "2026-02-03T04:05:06Z",
        "last_used_at": "2026-03-04T05:06:07Z",
    }
    legacy_envelope = _aio_v1_envelope(legacy, "prompt-library")
    current_container_envelope = PrivacyEnvelopeCodec(
        PROMPT_LIBRARY_CURRENT_SCHEMA
    ).encrypt_state(legacy)
    legacy_library_path(tmp_path).write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "version": 1,
                "prompts": [
                    {
                        "id": "prompt_legacy_synthetic",
                        "private": True,
                        "is_private": True,
                        "created_at": "2026-01-02T03:04:05Z",
                        "updated_at": "2026-02-03T04:05:06Z",
                        "last_used_at": "2026-03-04T05:06:07Z",
                        "encrypted_payload": legacy_envelope,
                    },
                    {
                        "id": "prompt_current_container",
                        "private": True,
                        "is_private": True,
                        "encrypted_payload": current_container_envelope,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    sources = discover_legacy_prompt_record_sources(tmp_path)
    assert len(sources) == 2
    assert sources[0].legacy_id == "prompt_legacy_synthetic"
    assert sources[0].current_format is False
    assert sources[1].current_format is True
    assert "prompt_legacy_synthetic" not in repr(sources[0])
    assert "ciphertext" not in repr(sources[0])
    record_id = generate_private_record_id()
    records = pack.records(PROMPT_LIBRARY_RESOURCE_ID)
    # Store instances are stateless views over the same atomic JSON document.
    adapter = AioPromptLibraryStoreAdapter(tmp_path)
    adapter.write_protected(record_id, sources[0].protected)
    current = records.reveal(
        PROMPT_RECORD_KIND,
        record_id,
        "details",
        _authorization(pack, token, "record.details"),
    ).value["record"]
    protected = adapter.read_protected(record_id)
    assert protected["schema"] == PROMPT_LIBRARY_CURRENT_SCHEMA
    assert current["name"] == "legacy synthetic name"
    assert current["created_at"] == "2026-01-02T03:04:05Z"
    assert current["updated_at"] == "2026-02-03T04:05:06Z"
    assert current["last_used_at"] == "2026-03-04T05:06:07Z"
    stored = managed_library_path(tmp_path).read_text(encoding="utf-8")
    assert "legacy synthetic" not in stored


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


def test_profile_declares_every_workflow_field_as_bounded_browser_authority():
    profile = build_aio_prompt_privacy_profile()

    assert profile.protected_fields
    for field in profile.protected_fields:
        assert field.state_authority is ProtectedStateAuthority.EXTERNAL_BROWSER_WORKFLOW
        assert field.external_transition_policy is not None
        assert field.external_transition_policy.max_original_bytes_per_owner == 1024 * 1024
        assert field.external_transition_policy.max_target_bytes_per_owner == 1024 * 1024


@pytest.mark.parametrize(
    ("adapter", "scope_id"),
    (
        (AioPromptModeAdapter(), GENERATE_SCOPE_ID),
        (AioPromptModeAdapter(), KREA_SCOPE_ID),
        (AioBuilderModeAdapter(), BUILDER_SCOPE_ID),
    ),
)
def test_mode_sources_use_revisioned_cas_and_exact_rollback(adapter, scope_id):
    prior = adapter.read_mode_source(scope_id)
    target = adapter.compare_and_set_mode_source(
        scope_id,
        prior["revision"],
        prior["declared"],
        "public",
    )

    assert target == {"revision": 1, "declared": "public"}
    assert adapter.classify_mode_source(scope_id, prior, target) == "target"
    with pytest.raises(RuntimeError, match="concurrently"):
        adapter.compare_and_set_mode_source(
            scope_id,
            prior["revision"],
            prior["declared"],
            "private",
        )
    restored = adapter.rollback_mode_source(scope_id, target, prior)
    assert restored == {"revision": 2, "declared": "inherit"}
    assert adapter.rollback_mode_source(scope_id, target, prior) == restored


def test_workflow_transition_codecs_round_trip_private_and_public_exact_bytes(
    tmp_path,
    monkeypatch,
):
    _installed_pack(tmp_path, monkeypatch)
    prompt = AioPromptWorkflowStateAdapter()
    prompt_value = {"value": "synthetic exact prompt"}
    prompt_envelope = PrivacyEnvelopeCodec(AIO_CURRENT_PROMPT_SCHEMA).encrypt_state(
        prompt_value
    )
    private_exact = json.dumps(
        prompt_envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert prompt.classify_mode_transition_representation(private_exact, None) == "private"
    assert prompt.decode_mode_transition_representation(private_exact, None) == prompt_value
    public_exact = prompt.encode_public_mode_transition(prompt_value, None)
    assert public_exact == b'{"value":"synthetic exact prompt"}'
    assert prompt.classify_mode_transition_representation(public_exact, None) == "public"

    builder = AioBuilderWorkflowStateAdapter()
    state = _builder_state("transition")
    builder_public = builder.encode_public_mode_transition(state, None)
    assert builder.classify_mode_transition_representation(builder_public, None) == "public"
    assert builder.decode_mode_transition_representation(builder_public, None) == state
    assert builder.normalize_mode_transition_value({"value": "widget"}, None) == {
        "value": "widget"
    }

    for invalid in (
        b"",
        b"[]",
        b'{"value":"one","value":"two"}',
        b'{"schema":"protected-marker","widgets":{}}',
    ):
        with pytest.raises(ValueError, match="representation is invalid"):
            prompt.classify_mode_transition_representation(invalid, None)


def test_prompt_library_store_has_durable_revisioned_cas(tmp_path):
    adapter = AioPromptLibraryStoreAdapter(tmp_path)
    record_id = generate_private_record_id()
    missing = RecordSnapshot(0)
    first = RecordSnapshot(1, {"ciphertext": "first"})
    second = RecordSnapshot(2, {"ciphertext": "second"})

    assert adapter.read_record(record_id) == missing
    assert adapter.compare_and_swap_record(record_id, missing, first) is True
    assert adapter.compare_and_swap_record(record_id, missing, first) is False
    assert adapter.compare_and_swap_record(record_id, first, second) is True
    assert adapter.read_record(record_id) == second
    assert adapter.list_ids() == (record_id,)

    tombstone = RecordSnapshot(3)
    assert adapter.compare_and_swap_record(record_id, second, tombstone) is True
    assert adapter.read_record(record_id) == tombstone
    assert adapter.list_ids() == ()


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
        subject_id="generate-cache-1",
    )
    first_result = execution.dispatch(
        first.reference,
        context,
        subject_id="generate-cache-1",
    )
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
        subject_id="generate-cache-1",
    )
    second_result = execution.dispatch(
        second.reference,
        context,
        subject_id="generate-cache-1",
    )

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
        subject_id="generate-locked-1",
    )
    tampered = dict(prepared.reference)
    tampered["grant"] = "invalid-grant"
    with pytest.raises(ExecutionError):
        execution.dispatch(
            tampered,
            {"dispatch": lambda value: calls.append(value)},
            subject_id="generate-locked-1",
        )

    reveal_authorization = _authorization(pack, token, "snapshot.reveal")
    lock_keystore()
    with pytest.raises(ExecutionError):
        execution.dispatch(
            prepared.reference,
            {"dispatch": lambda value: calls.append(value)},
            subject_id="generate-locked-1",
        )
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
    subject_id = "generate-node-1"
    subject = _prepared_subject(
        pack,
        token,
        GENERATE_SUBJECT_MODE_BINDING_ID,
        subject_id,
    )
    codec = PrivacyEnvelopeCodec(AIO_CURRENT_PROMPT_SCHEMA)
    prepared = pack.execution(GENERATE_EXECUTION_RESOURCE_ID).prepare(
        GENERATE_PROJECTION_ID,
        {
            POSITIVE_PROMPT_FIELD_ID: codec.encrypt_state({"value": "managed positive"}),
            NEGATIVE_PROMPT_FIELD_ID: codec.encrypt_state({"value": "managed negative"}),
        },
        _authorization(pack, token, "execution.prepare"),
        subject_id=subject_id,
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
        unique_id=subject_id,
        privacy_mode_reference=json.dumps(subject.reference),
        private_execution=json.dumps(prepared.reference),
    )

    assert result[0] == "image"
    assert captured["generated"]["positive_prompt"] == "managed positive"
    assert captured["generated"]["negative_prompt"] == "managed negative"


def test_krea_node_dispatches_private_reference_into_settings(tmp_path, monkeypatch):
    pack, token = _installed_pack(tmp_path, monkeypatch)
    subject_id = "krea-node-1"
    subject = _prepared_subject(
        pack,
        token,
        KREA_SUBJECT_MODE_BINDING_ID,
        subject_id,
    )
    codec = PrivacyEnvelopeCodec(AIO_CURRENT_PROMPT_SCHEMA)
    prepared = pack.execution("krea-inpaint-execution").prepare(
        "inpaint",
        {
            KREA_INPAINT_PROMPT_FIELD_ID: codec.encrypt_state(
                {"value": "managed inpaint"}
            )
        },
        _authorization(pack, token, "execution.prepare"),
        subject_id=subject_id,
    )

    settings = AIOKrea2Settings().build_settings(
        enhancer_enabled=True,
        enhancer_strength=1.0,
        precision_policy="auto",
        inpaint_positive_prompt="untrusted raw inpaint",
        unique_id=subject_id,
        privacy_mode_reference=json.dumps(subject.reference),
        private_execution=json.dumps(prepared.reference),
    )[0]

    assert settings["positive_prompt_override"] == "managed inpaint"
    assert settings["positive_prompt_source"] == "krea2_inpaint_settings"


def test_private_subject_leases_reject_missing_execution_pair(tmp_path, monkeypatch):
    pack, token = _installed_pack(tmp_path, monkeypatch)

    generate_id = "generate-missing-execution"
    generate_subject = _prepared_subject(
        pack,
        token,
        GENERATE_SUBJECT_MODE_BINDING_ID,
        generate_id,
    )
    with pytest.raises(ValueError, match="managed execution reference"):
        AIOImageGenerate().generate(
            model_type="z_image_turbo",
            diffusion_model="model.safetensors",
            text_encoder="text.safetensors",
            vae="vae.safetensors",
            positive_prompt="untrusted request prompt",
            negative_prompt="",
            width=1024,
            height=1024,
            seed=0,
            steps=1,
            cfg=1.0,
            sampler="auto",
            scheduler="auto",
            unique_id=generate_id,
            privacy_mode_reference=json.dumps(generate_subject.reference),
        )

    krea_id = "krea-missing-execution"
    krea_subject = _prepared_subject(
        pack,
        token,
        KREA_SUBJECT_MODE_BINDING_ID,
        krea_id,
    )
    with pytest.raises(ValueError, match="managed execution reference"):
        AIOKrea2Settings().build_settings(
            enhancer_enabled=True,
            enhancer_strength=1.0,
            precision_policy="auto",
            inpaint_positive_prompt="untrusted request prompt",
            unique_id=krea_id,
            privacy_mode_reference=json.dumps(krea_subject.reference),
        )

    builder_id = "builder-missing-execution"
    builder_subject = _prepared_subject(
        pack,
        token,
        BUILDER_SUBJECT_MODE_BINDING_ID,
        builder_id,
    )
    with pytest.raises(ValueError, match="managed execution reference"):
        AIOIdeogram4PromptBuilder().build_prompt(
            background="untrusted request prompt",
            unique_id=builder_id,
            privacy_mode_reference=json.dumps(builder_subject.reference),
            **{"max side": 1024, "aspect ratio": "1:1", "multiple value": "none"},
        )


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
    subject_id = "builder-node-1"
    subject = _prepared_subject(
        pack,
        token,
        BUILDER_SUBJECT_MODE_BINDING_ID,
        subject_id,
    )
    second_subject = _prepared_subject(
        pack,
        token,
        BUILDER_SUBJECT_MODE_BINDING_ID,
        subject_id,
    )
    prepared = execution.prepare(
        BUILDER_PROJECTION_ID,
        protected,
        _authorization(pack, token, "execution.prepare"),
        subject_id=subject_id,
    )
    second_prepared = execution.prepare(
        BUILDER_PROJECTION_ID,
        protected,
        _authorization(pack, token, "execution.prepare"),
        subject_id=subject_id,
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
        unique_id=subject_id,
        privacy_mode_reference=json.dumps(subject.reference),
        private_execution=json.dumps(prepared.reference),
        **{"max side": 512, "aspect ratio": "1:1", "multiple value": "none"},
    )["result"]
    second_result = AIOIdeogram4PromptBuilder().build_prompt(
        background="another untrusted background",
        image="preview-two",
        privacy_mode=False,
        unique_id=subject_id,
        privacy_mode_reference=json.dumps(second_subject.reference),
        private_execution=json.dumps(second_prepared.reference),
        **{"max side": 512, "aspect ratio": "1:1", "multiple value": "none"},
    )["result"]

    assert result[2] == "preview-one"
    assert second_result[2] == "preview-two"
    assert result[0]["privacy_mode"] is True
    assert result[0]["prompt"] == result[1]
    assert result[4:6] == (1024, 1024)
    assert "Managed overview" in result[1]
    assert "Managed room" in result[1]
    assert json.loads(result[1])["compositional_deconstruction"]["elements"][0]["bbox"] == [102, 205, 410, 614]
    assert "untrusted raw background" not in result[1]
