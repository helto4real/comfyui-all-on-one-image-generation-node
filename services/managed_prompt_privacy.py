"""One complete shared-privacy profile for AIO Image Generate."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from helto_privacy import (
    AIO_V1_JSON_KEY_IMPORT_ID,
    AIO_V1_READER_ID,
    AdapterSlot,
    BoundPrivacyPack,
    ConsumerSuiteDeclaration,
    FieldLocation,
    FieldLocationKind,
    ExternalTransitionPolicy,
    LegacyKeyFormat,
    LegacyKeyImportBinding,
    LegacyLocationKind,
    LegacyReaderBinding,
    PrivacyProfile,
    PrivacyScope,
    ProfileResource,
    ProtectedField,
    ProtectedStateAuthority,
    ProtectedOperation,
    RecordDeclaration,
    RecordRevealProjection,
    ResourceKind,
    SemanticExecutionProjection,
    SubjectModeBinding,
    aio_v1_reader_unit,
    install,
    register_consumer_suite_declaration,
    register_legacy_reader_units,
)
from helto_privacy.runtime import bound_privacy_pack

from .prompt_resolution import (
    GENERATE_EXECUTION_RESOURCE_ID,
    KREA_EXECUTION_RESOURCE_ID,
    normalize_prompt_text,
)
from .managed_builder_privacy import (
    BUILDER_DISPATCH_ADAPTER_ID,
    BUILDER_EXECUTION_RESOURCE_ID,
    BUILDER_MODE_ADAPTER_ID,
    BUILDER_MODE_BROWSER_ADAPTER_ID,
    BUILDER_NODE_TYPE,
    BUILDER_OPERATION_ADAPTER_ID,
    BUILDER_PROJECTION_ADAPTER_ID,
    BUILDER_PROJECTION_ID,
    BUILDER_SUBJECT_MODE_BINDING_ID,
    BUILDER_SCOPE_ID,
    BUILDER_WORKFLOW_ADAPTER_ID,
    BUILDER_WORKFLOW_BROWSER_ADAPTER_ID,
    BUILDER_WORKFLOW_RESOURCE_ID,
    AioBuilderExecutionDispatchAdapter,
    AioBuilderExecutionProjectionAdapter,
    AioBuilderModeAdapter,
    AioBuilderOperationAdapter,
    AioBuilderWorkflowStateAdapter,
    builder_legacy_bindings,
    builder_legacy_key_imports,
    builder_protected_fields,
)
from .managed_prompt_library_privacy import (
    PROMPT_LIBRARY_CURRENT_SCHEMA,
    PROMPT_LIBRARY_LEGACY_BINDING_ID,
    PROMPT_LIBRARY_LEGACY_KEY_BINDING_ID,
    PROMPT_LIBRARY_RESOURCE_ID,
    PROMPT_LIBRARY_STORE_ADAPTER_ID,
    PROMPT_RECORD_KIND,
    AioPromptLibraryStoreAdapter,
)
from .managed_run_info_privacy import (
    RUN_INFO_ADAPTER_ID,
    RUN_INFO_OPERATION_ID,
    RUN_INFO_RESOURCE_ID,
    AioRunInfoProjectionAdapter,
    run_info_safe_projection,
    run_info_sensitive_fields,
)
from .managed_privacy_execution import (
    AIO_MANAGED_PRIVACY_PROFILE_ID,
    dispatch_aio_managed_execution,
)
from .managed_mode_transition import (
    ExternalWorkflowTransitionCodec,
    RevisionedModeSourceAdapter,
)


AIO_PRIVACY_PROFILE_ID = AIO_MANAGED_PRIVACY_PROFILE_ID
AIO_PRIVACY_DISTRIBUTION = "comfyui-all-on-one-image-generation-node"
AIO_SUITE_ID = "helto-suite-2026-07-17.2"
AIO_CURRENT_PROMPT_SCHEMA = "helto.aio-image-generate.v2"
AIO_PRIVACY_PROFILE_FINGERPRINT = (
    "f63424f85dfa083277d43069d1a399f500f77e132f001a9355da20dab0f133a1"
)

GENERATE_NODE_TYPE = "AIOImageGenerate"
KREA_NODE_TYPE = "AIOKrea2Settings"

PROMPT_MODE_RESOURCE_ID = "prompt-mode"
GENERATE_WORKFLOW_RESOURCE_ID = "generate-prompts"
KREA_WORKFLOW_RESOURCE_ID = "krea-inpaint"

GENERATE_SCOPE_ID = "generate"
KREA_SCOPE_ID = "krea-inpaint"

PROMPT_MODE_ADAPTER_ID = "prompt-mode-state"
PROMPT_MODE_BROWSER_ADAPTER_ID = "prompt-mode-browser"
GENERATE_WORKFLOW_ADAPTER_ID = "generate-workflow-state"
KREA_WORKFLOW_ADAPTER_ID = "krea-workflow-state"
GENERATE_WORKFLOW_BROWSER_ADAPTER_ID = "generate-workflow-browser"
KREA_WORKFLOW_BROWSER_ADAPTER_ID = "krea-workflow-browser"
GENERATE_PROJECTION_ADAPTER_ID = "generate-execution-projection"
KREA_PROJECTION_ADAPTER_ID = "krea-execution-projection"
GENERATE_DISPATCH_ADAPTER_ID = "generate-execution-dispatch"
KREA_DISPATCH_ADAPTER_ID = "krea-execution-dispatch"
GENERATE_OPERATION_ADAPTER_ID = "generate-protected-operations"
KREA_OPERATION_ADAPTER_ID = "krea-protected-operations"

POSITIVE_PROMPT_FIELD_ID = "generate-positive-prompt"
NEGATIVE_PROMPT_FIELD_ID = "generate-negative-prompt"
KREA_INPAINT_PROMPT_FIELD_ID = "krea-inpaint-positive-prompt"

GENERATE_PROJECTION_ID = "generate"
KREA_INPAINT_PROJECTION_ID = "inpaint"
GENERATE_SUBJECT_MODE_BINDING_ID = "generate-mode"
KREA_SUBJECT_MODE_BINDING_ID = "krea-inpaint-mode"

_INSTALL_LOCK = RLock()
_PACK: BoundPrivacyPack | None = None
_ADAPTERS: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class AioPromptFieldFacts:
    workflow_resource_id: str
    scope_id: str
    node_type: str
    widget_name: str


@dataclass(frozen=True, slots=True)
class AioPromptProjectionFacts:
    field_to_semantic: tuple[tuple[str, str], ...]
    fixed_semantics: tuple[tuple[str, str], ...] = ()
    stripped_semantics: tuple[str, ...] = ()
    linked_inputs: tuple[tuple[str, str], ...] = ()

    @property
    def semantic_names(self) -> frozenset[str]:
        return frozenset(
            (*dict(self.field_to_semantic).values(), *dict(self.fixed_semantics))
        )


_FIELD_FACTS = {
    POSITIVE_PROMPT_FIELD_ID: AioPromptFieldFacts(
        GENERATE_WORKFLOW_RESOURCE_ID,
        GENERATE_SCOPE_ID,
        GENERATE_NODE_TYPE,
        "positive_prompt",
    ),
    NEGATIVE_PROMPT_FIELD_ID: AioPromptFieldFacts(
        GENERATE_WORKFLOW_RESOURCE_ID,
        GENERATE_SCOPE_ID,
        GENERATE_NODE_TYPE,
        "negative_prompt",
    ),
    KREA_INPAINT_PROMPT_FIELD_ID: AioPromptFieldFacts(
        KREA_WORKFLOW_RESOURCE_ID,
        KREA_SCOPE_ID,
        KREA_NODE_TYPE,
        "inpaint_positive_prompt",
    ),
}
_PROJECTION_FACTS = {
    GENERATE_PROJECTION_ID: AioPromptProjectionFacts(
        field_to_semantic=(
            (POSITIVE_PROMPT_FIELD_ID, "positive_prompt"),
            (NEGATIVE_PROMPT_FIELD_ID, "negative_prompt"),
        ),
        linked_inputs=(
            ("positive_prompt", "positive_prompt"),
            ("negative_prompt", "negative_prompt"),
        ),
    ),
    KREA_INPAINT_PROJECTION_ID: AioPromptProjectionFacts(
        field_to_semantic=((KREA_INPAINT_PROMPT_FIELD_ID, "positive_prompt_override"),),
        fixed_semantics=(("positive_prompt_source", "krea2_inpaint_settings"),),
        stripped_semantics=("positive_prompt_override",),
        linked_inputs=(("inpaint_positive_prompt", "positive_prompt_override"),),
    ),
}


def aio_prompt_legacy_binding_id(field_id: str) -> str:
    _field_facts(field_id)
    return f"{field_id}-aio-v1"


def aio_prompt_legacy_key_binding_id(field_id: str) -> str:
    _field_facts(field_id)
    return f"{field_id}-aio-json-key-v1"


def _protected_field(field_id: str, facts: AioPromptFieldFacts) -> ProtectedField:
    generate = facts.workflow_resource_id == GENERATE_WORKFLOW_RESOURCE_ID
    return ProtectedField(
        field_id,
        facts.workflow_resource_id,
        facts.scope_id,
        GENERATE_WORKFLOW_ADAPTER_ID if generate else KREA_WORKFLOW_ADAPTER_ID,
        (
            GENERATE_WORKFLOW_BROWSER_ADAPTER_ID
            if generate
            else KREA_WORKFLOW_BROWSER_ADAPTER_ID
        ),
        (facts.node_type,),
        FieldLocation(FieldLocationKind.WIDGET, facts.widget_name),
        AIO_CURRENT_PROMPT_SCHEMA,
        field_id,
        ProtectedStateAuthority.EXTERNAL_BROWSER_WORKFLOW,
        ExternalTransitionPolicy(
            max_original_bytes_per_owner=1024 * 1024,
            max_target_bytes_per_owner=1024 * 1024,
        ),
        legacy_reader_ids=(AIO_V1_READER_ID,),
        execution=True,
    )


def build_aio_prompt_privacy_profile() -> PrivacyProfile:
    """Build the complete immutable AIO V3 privacy profile."""

    fields = (
        *tuple(_protected_field(field_id, facts) for field_id, facts in _FIELD_FACTS.items()),
        *builder_protected_fields(),
    )
    resources = (
        ProfileResource(
            PROMPT_MODE_RESOURCE_ID,
            ResourceKind.MODE,
            (
                PROMPT_MODE_ADAPTER_ID,
                PROMPT_MODE_BROWSER_ADAPTER_ID,
                BUILDER_MODE_ADAPTER_ID,
                BUILDER_MODE_BROWSER_ADAPTER_ID,
            ),
        ),
        ProfileResource(
            GENERATE_WORKFLOW_RESOURCE_ID,
            ResourceKind.WORKFLOW,
            (
                GENERATE_WORKFLOW_ADAPTER_ID,
                GENERATE_WORKFLOW_BROWSER_ADAPTER_ID,
                GENERATE_OPERATION_ADAPTER_ID,
                RUN_INFO_ADAPTER_ID,
            ),
        ),
        ProfileResource(
            KREA_WORKFLOW_RESOURCE_ID,
            ResourceKind.WORKFLOW,
            (
                KREA_WORKFLOW_ADAPTER_ID,
                KREA_WORKFLOW_BROWSER_ADAPTER_ID,
                KREA_OPERATION_ADAPTER_ID,
            ),
        ),
        ProfileResource(
            BUILDER_WORKFLOW_RESOURCE_ID,
            ResourceKind.WORKFLOW,
            (
                BUILDER_WORKFLOW_ADAPTER_ID,
                BUILDER_WORKFLOW_BROWSER_ADAPTER_ID,
                BUILDER_OPERATION_ADAPTER_ID,
            ),
        ),
        ProfileResource(
            PROMPT_LIBRARY_RESOURCE_ID,
            ResourceKind.RECORD,
            (PROMPT_LIBRARY_STORE_ADAPTER_ID,),
        ),
        ProfileResource(
            GENERATE_EXECUTION_RESOURCE_ID,
            ResourceKind.EXECUTION,
            (GENERATE_PROJECTION_ADAPTER_ID, GENERATE_DISPATCH_ADAPTER_ID),
        ),
        ProfileResource(
            KREA_EXECUTION_RESOURCE_ID,
            ResourceKind.EXECUTION,
            (KREA_PROJECTION_ADAPTER_ID, KREA_DISPATCH_ADAPTER_ID),
        ),
        ProfileResource(
            BUILDER_EXECUTION_RESOURCE_ID,
            ResourceKind.EXECUTION,
            (BUILDER_PROJECTION_ADAPTER_ID, BUILDER_DISPATCH_ADAPTER_ID),
        ),
    )
    profile = PrivacyProfile(
        id=AIO_PRIVACY_PROFILE_ID,
        distribution=AIO_PRIVACY_DISTRIBUTION,
        resources=resources,
        server_adapters=(
            AdapterSlot(PROMPT_MODE_ADAPTER_ID, ResourceKind.MODE, PROMPT_MODE_RESOURCE_ID),
            AdapterSlot(BUILDER_MODE_ADAPTER_ID, ResourceKind.MODE, PROMPT_MODE_RESOURCE_ID),
            AdapterSlot(
                GENERATE_WORKFLOW_ADAPTER_ID,
                ResourceKind.WORKFLOW,
                GENERATE_WORKFLOW_RESOURCE_ID,
            ),
            AdapterSlot(
                RUN_INFO_ADAPTER_ID,
                ResourceKind.WORKFLOW,
                GENERATE_WORKFLOW_RESOURCE_ID,
            ),
            AdapterSlot(
                KREA_WORKFLOW_ADAPTER_ID,
                ResourceKind.WORKFLOW,
                KREA_WORKFLOW_RESOURCE_ID,
            ),
            AdapterSlot(
                BUILDER_WORKFLOW_ADAPTER_ID,
                ResourceKind.WORKFLOW,
                BUILDER_WORKFLOW_RESOURCE_ID,
            ),
            AdapterSlot(
                PROMPT_LIBRARY_STORE_ADAPTER_ID,
                ResourceKind.RECORD,
                PROMPT_LIBRARY_RESOURCE_ID,
            ),
            AdapterSlot(
                GENERATE_PROJECTION_ADAPTER_ID,
                ResourceKind.EXECUTION,
                GENERATE_EXECUTION_RESOURCE_ID,
            ),
            AdapterSlot(
                GENERATE_DISPATCH_ADAPTER_ID,
                ResourceKind.EXECUTION,
                GENERATE_EXECUTION_RESOURCE_ID,
            ),
            AdapterSlot(
                KREA_PROJECTION_ADAPTER_ID,
                ResourceKind.EXECUTION,
                KREA_EXECUTION_RESOURCE_ID,
            ),
            AdapterSlot(
                KREA_DISPATCH_ADAPTER_ID,
                ResourceKind.EXECUTION,
                KREA_EXECUTION_RESOURCE_ID,
            ),
            AdapterSlot(
                BUILDER_PROJECTION_ADAPTER_ID,
                ResourceKind.EXECUTION,
                BUILDER_EXECUTION_RESOURCE_ID,
            ),
            AdapterSlot(
                BUILDER_DISPATCH_ADAPTER_ID,
                ResourceKind.EXECUTION,
                BUILDER_EXECUTION_RESOURCE_ID,
            ),
            AdapterSlot(
                GENERATE_OPERATION_ADAPTER_ID,
                ResourceKind.WORKFLOW,
                GENERATE_WORKFLOW_RESOURCE_ID,
            ),
            AdapterSlot(
                KREA_OPERATION_ADAPTER_ID,
                ResourceKind.WORKFLOW,
                KREA_WORKFLOW_RESOURCE_ID,
            ),
            AdapterSlot(
                BUILDER_OPERATION_ADAPTER_ID,
                ResourceKind.WORKFLOW,
                BUILDER_WORKFLOW_RESOURCE_ID,
            ),
        ),
        browser_adapters=(
            AdapterSlot(
                BUILDER_MODE_BROWSER_ADAPTER_ID,
                ResourceKind.MODE,
                PROMPT_MODE_RESOURCE_ID,
                (BUILDER_NODE_TYPE,),
            ),
            AdapterSlot(
                PROMPT_MODE_BROWSER_ADAPTER_ID,
                ResourceKind.MODE,
                PROMPT_MODE_RESOURCE_ID,
                (GENERATE_NODE_TYPE, KREA_NODE_TYPE),
            ),
            AdapterSlot(
                GENERATE_WORKFLOW_BROWSER_ADAPTER_ID,
                ResourceKind.WORKFLOW,
                GENERATE_WORKFLOW_RESOURCE_ID,
                (GENERATE_NODE_TYPE,),
            ),
            AdapterSlot(
                KREA_WORKFLOW_BROWSER_ADAPTER_ID,
                ResourceKind.WORKFLOW,
                KREA_WORKFLOW_RESOURCE_ID,
                (KREA_NODE_TYPE,),
            ),
            AdapterSlot(
                BUILDER_WORKFLOW_BROWSER_ADAPTER_ID,
                ResourceKind.WORKFLOW,
                BUILDER_WORKFLOW_RESOURCE_ID,
                (BUILDER_NODE_TYPE,),
            ),
        ),
        scopes=(
            PrivacyScope(
                GENERATE_SCOPE_ID,
                PROMPT_MODE_RESOURCE_ID,
                PROMPT_MODE_ADAPTER_ID,
                PROMPT_MODE_BROWSER_ADAPTER_ID,
            ),
            PrivacyScope(
                KREA_SCOPE_ID,
                PROMPT_MODE_RESOURCE_ID,
                PROMPT_MODE_ADAPTER_ID,
                PROMPT_MODE_BROWSER_ADAPTER_ID,
            ),
            PrivacyScope(
                BUILDER_SCOPE_ID,
                PROMPT_MODE_RESOURCE_ID,
                BUILDER_MODE_ADAPTER_ID,
                BUILDER_MODE_BROWSER_ADAPTER_ID,
            ),
        ),
        protected_fields=fields,
        subject_mode_bindings=(
            SubjectModeBinding(
                GENERATE_SUBJECT_MODE_BINDING_ID,
                GENERATE_SCOPE_ID,
                "privacy_mode_reference",
                (GENERATE_NODE_TYPE,),
            ),
            SubjectModeBinding(
                KREA_SUBJECT_MODE_BINDING_ID,
                KREA_SCOPE_ID,
                "privacy_mode_reference",
                (KREA_NODE_TYPE,),
            ),
            SubjectModeBinding(
                BUILDER_SUBJECT_MODE_BINDING_ID,
                BUILDER_SCOPE_ID,
                "privacy_mode_reference",
                (BUILDER_NODE_TYPE,),
            ),
        ),
        records=(
            RecordDeclaration(
                PROMPT_RECORD_KIND,
                PROMPT_LIBRARY_RESOURCE_ID,
                BUILDER_SCOPE_ID,
                PROMPT_LIBRARY_CURRENT_SCHEMA,
                PROMPT_LIBRARY_STORE_ADAPTER_ID,
                projections=(
                    RecordRevealProjection("details", ("record",)),
                    RecordRevealProjection("use", ("record",)),
                ),
                mutation_operations=("create", "replace", "patch", "duplicate"),
                safe_projection=(),
                fixed_private_label="Private record",
            ),
        ),
        execution_projections=(
            SemanticExecutionProjection(
                GENERATE_PROJECTION_ID,
                GENERATE_EXECUTION_RESOURCE_ID,
                GENERATE_WORKFLOW_RESOURCE_ID,
                GENERATE_PROJECTION_ADAPTER_ID,
                GENERATE_DISPATCH_ADAPTER_ID,
                GENERATE_SUBJECT_MODE_BINDING_ID,
            ),
            SemanticExecutionProjection(
                KREA_INPAINT_PROJECTION_ID,
                KREA_EXECUTION_RESOURCE_ID,
                KREA_WORKFLOW_RESOURCE_ID,
                KREA_PROJECTION_ADAPTER_ID,
                KREA_DISPATCH_ADAPTER_ID,
                KREA_SUBJECT_MODE_BINDING_ID,
            ),
            SemanticExecutionProjection(
                BUILDER_PROJECTION_ID,
                BUILDER_EXECUTION_RESOURCE_ID,
                BUILDER_WORKFLOW_RESOURCE_ID,
                BUILDER_PROJECTION_ADAPTER_ID,
                BUILDER_DISPATCH_ADAPTER_ID,
                BUILDER_SUBJECT_MODE_BINDING_ID,
            ),
        ),
        protected_operations=(
            ProtectedOperation(
                "generate",
                GENERATE_WORKFLOW_RESOURCE_ID,
                GENERATE_OPERATION_ADAPTER_ID,
                "/aio_image_generate/generate",
            ),
            ProtectedOperation(
                RUN_INFO_OPERATION_ID,
                RUN_INFO_RESOURCE_ID,
                RUN_INFO_ADAPTER_ID,
                None,
                scope_id=GENERATE_SCOPE_ID,
                sensitive_fields=run_info_sensitive_fields(),
                safe_projection=run_info_safe_projection(),
                subject_mode_binding_id=GENERATE_SUBJECT_MODE_BINDING_ID,
            ),
            ProtectedOperation(
                "inpaint",
                KREA_WORKFLOW_RESOURCE_ID,
                KREA_OPERATION_ADAPTER_ID,
                "/aio_image_generate/inpaint",
            ),
            ProtectedOperation(
                "ideogram-builder.build",
                BUILDER_WORKFLOW_RESOURCE_ID,
                BUILDER_OPERATION_ADAPTER_ID,
                "/aio_image_generate/ideogram4/build_prompt",
            ),
        ),
        legacy_bindings=(*tuple(
            LegacyReaderBinding(
                aio_prompt_legacy_binding_id(field_id),
                AIO_V1_READER_ID,
                facts.workflow_resource_id,
                LegacyLocationKind.WORKFLOW_FIELD,
                field_id,
            )
            for field_id, facts in _FIELD_FACTS.items()
        ), *builder_legacy_bindings(), LegacyReaderBinding(
            PROMPT_LIBRARY_LEGACY_BINDING_ID,
            AIO_V1_READER_ID,
            PROMPT_LIBRARY_RESOURCE_ID,
            LegacyLocationKind.RECORD,
            PROMPT_RECORD_KIND,
        )),
        legacy_key_imports=(*tuple(
            LegacyKeyImportBinding(
                aio_prompt_legacy_key_binding_id(field_id),
                AIO_V1_JSON_KEY_IMPORT_ID,
                facts.workflow_resource_id,
                LegacyLocationKind.WORKFLOW_FIELD,
                field_id,
                LegacyKeyFormat.JSON,
            )
            for field_id, facts in _FIELD_FACTS.items()
        ), *builder_legacy_key_imports(), LegacyKeyImportBinding(
            PROMPT_LIBRARY_LEGACY_KEY_BINDING_ID,
            AIO_V1_JSON_KEY_IMPORT_ID,
            PROMPT_LIBRARY_RESOURCE_ID,
            LegacyLocationKind.RECORD,
            PROMPT_RECORD_KIND,
            LegacyKeyFormat.JSON,
        )),
    )
    if profile.fingerprint != AIO_PRIVACY_PROFILE_FINGERPRINT:
        raise RuntimeError(
            "AIO privacy profile fingerprint changed unexpectedly: "
            f"{profile.fingerprint}."
        )
    return profile


class AioPromptModeAdapter(RevisionedModeSourceAdapter):
    """Map the old Generate boolean while keeping missing state private."""

    def __init__(self, declarations: Mapping[str, object] | None = None) -> None:
        super().__init__((GENERATE_SCOPE_ID, KREA_SCOPE_ID), declarations)


class AioPromptWorkflowStateAdapter(ExternalWorkflowTransitionCodec):
    """Own prompt locations, fallback recovery, and canonical text semantics."""

    def __init__(self) -> None:
        super().__init__(AIO_CURRENT_PROMPT_SCHEMA)

    def capture(self, source: object, declaration: object) -> object:
        field_id = _declaration_id(declaration)
        widget_name = _field_facts(field_id).widget_name
        if isinstance(source, Mapping):
            value = source.get(widget_name)
        else:
            value = getattr(source, widget_name)
        return copy.deepcopy(value)

    def normalize(self, value: object, declaration: object) -> dict[str, str]:
        _declaration_id(declaration)
        return {"value": normalize_prompt_text(value)}

    def apply_revealed(self, target: object, value: object, declaration: object) -> None:
        widget_name = _field_facts(_declaration_id(declaration)).widget_name
        normalized = normalize_prompt_text(value)
        if isinstance(target, dict):
            target[widget_name] = normalized
        else:
            setattr(target, widget_name, normalized)

    def clear_plaintext(self, target: object, declaration: object) -> None:
        widget_name = _field_facts(_declaration_id(declaration)).widget_name
        if isinstance(target, dict):
            target[widget_name] = ""
        else:
            setattr(target, widget_name, "")

    def _normalize_transition_value(self, value: object) -> Mapping[str, object]:
        return {"value": normalize_prompt_text(value)}


class AioPromptExecutionProjectionAdapter:
    """Project protected workflow fields into pipeline prompt semantics."""

    def project(self, fields: Mapping[str, object], declaration: object) -> dict[str, str]:
        projection_id = getattr(declaration, "id", None)
        facts = _PROJECTION_FACTS.get(projection_id)
        if facts is None or set(fields) != set(dict(facts.field_to_semantic)):
            raise ValueError("AIO prompt snapshot is incomplete or unknown.")
        semantic = {
            semantic_name: normalize_prompt_text(fields[field_id])
            for field_id, semantic_name in facts.field_to_semantic
        }
        for name in facts.stripped_semantics:
            semantic[name] = semantic[name].strip()
        semantic.update(facts.fixed_semantics)
        return semantic


class AioPromptExecutionDispatchAdapter:
    """Delegate resolved semantics to the existing AIO pipeline callback."""

    def dispatch(self, value: object, context: object, cancellation: object) -> object:
        callback = context.get("dispatch") if isinstance(context, Mapping) else None
        if not callable(callback):
            raise ValueError("AIO prompt execution dispatch is unavailable.")
        checkpoint = getattr(cancellation, "checkpoint", None)
        if callable(checkpoint):
            checkpoint()
        return callback(resolve_execution_prompt_semantics(value, context))


class AioPromptOperationAdapter:
    def __init__(self, dispatcher: Callable[[object, object], object] | None = None) -> None:
        self._dispatcher = dispatcher

    def invoke(self, payload: object, context: object) -> object:
        if not callable(self._dispatcher):
            raise ValueError("AIO protected prompt operation is unavailable.")
        return self._dispatcher(payload, context)


def build_aio_prompt_server_adapters(
    *,
    declarations: Mapping[str, object] | None = None,
    operation_dispatcher: Callable[[object, object], object] | None = None,
    prompt_library_base_dir: str | None = None,
) -> dict[str, object]:
    mode = AioPromptModeAdapter(declarations)
    workflow = AioPromptWorkflowStateAdapter()
    projection = AioPromptExecutionProjectionAdapter()
    dispatch = AioPromptExecutionDispatchAdapter()
    operation = AioPromptOperationAdapter(operation_dispatcher)
    return {
        PROMPT_MODE_ADAPTER_ID: mode,
        BUILDER_MODE_ADAPTER_ID: AioBuilderModeAdapter(declarations),
        GENERATE_WORKFLOW_ADAPTER_ID: workflow,
        KREA_WORKFLOW_ADAPTER_ID: workflow,
        BUILDER_WORKFLOW_ADAPTER_ID: AioBuilderWorkflowStateAdapter(),
        GENERATE_PROJECTION_ADAPTER_ID: projection,
        KREA_PROJECTION_ADAPTER_ID: projection,
        BUILDER_PROJECTION_ADAPTER_ID: AioBuilderExecutionProjectionAdapter(),
        GENERATE_DISPATCH_ADAPTER_ID: dispatch,
        KREA_DISPATCH_ADAPTER_ID: dispatch,
        BUILDER_DISPATCH_ADAPTER_ID: AioBuilderExecutionDispatchAdapter(),
        GENERATE_OPERATION_ADAPTER_ID: operation,
        RUN_INFO_ADAPTER_ID: AioRunInfoProjectionAdapter(),
        KREA_OPERATION_ADAPTER_ID: operation,
        BUILDER_OPERATION_ADAPTER_ID: AioBuilderOperationAdapter(operation_dispatcher),
        PROMPT_LIBRARY_STORE_ADAPTER_ID: AioPromptLibraryStoreAdapter(
            prompt_library_base_dir
        ),
    }


def install_aio_privacy(
    package_root: str | Path,
    *,
    operation_dispatcher: Callable[[object, object], object] | None = None,
) -> BoundPrivacyPack:
    """Atomically install AIO only after every declared adapter is available."""

    global _ADAPTERS, _PACK
    with _INSTALL_LOCK:
        if _PACK is not None:
            return _PACK
        register_legacy_reader_units((aio_v1_reader_unit(),))
        profile = build_aio_prompt_privacy_profile()
        adapters = build_aio_prompt_server_adapters(
            operation_dispatcher=operation_dispatcher,
            prompt_library_base_dir=str(Path(package_root) / "config"),
        )
        expected = {slot.id for slot in profile.server_adapters}
        if set(adapters) != expected:
            raise RuntimeError("AIO privacy adapter binding is incomplete.")
        _PACK = install(profile, adapters)
        register_consumer_suite_declaration(
            ConsumerSuiteDeclaration(AIO_PRIVACY_DISTRIBUTION, AIO_SUITE_ID)
        )
        _ADAPTERS = adapters
        return _PACK


def aio_privacy_pack() -> BoundPrivacyPack:
    """Return the installed AIO pack without exposing mutable runtime state."""

    return bound_privacy_pack(AIO_PRIVACY_PROFILE_ID)


def aio_privacy_adapter(adapter_id: str) -> object:
    if _ADAPTERS is None or adapter_id not in _ADAPTERS:
        raise RuntimeError("AIO privacy adapters are not installed.")
    return _ADAPTERS[adapter_id]


def dispatch_aio_prompt_execution(
    reference: object,
    execution_resource_id: str,
    context: Mapping[str, object],
    *,
    subject_id: object,
) -> object:
    return dispatch_aio_managed_execution(
        reference,
        execution_resource_id,
        context,
        subject_id=subject_id,
    )


def resolve_execution_prompt_semantics(value: object, context: object) -> dict[str, str]:
    """Overlay evaluated linked inputs onto one protected local snapshot."""

    if not isinstance(value, Mapping) or not isinstance(context, Mapping):
        raise ValueError("AIO prompt execution state is invalid.")
    semantic = {str(key): normalize_prompt_text(item) for key, item in value.items()}
    linked = context.get("linked_inputs", {})
    inputs = context.get("prompt_inputs", {})
    if not isinstance(linked, Mapping) or not isinstance(inputs, Mapping):
        raise ValueError("AIO linked prompt context is invalid.")
    facts = next(
        (
            candidate
            for candidate in _PROJECTION_FACTS.values()
            if candidate.semantic_names == set(semantic)
        ),
        None,
    )
    if facts is None:
        raise ValueError("AIO prompt execution state is invalid.")
    for input_name, semantic_name in facts.linked_inputs:
        if linked.get(input_name) is not True:
            continue
        if input_name not in inputs:
            raise ValueError("AIO linked prompt input is unavailable.")
        semantic[semantic_name] = normalize_prompt_text(inputs[input_name])
        if semantic_name in facts.stripped_semantics:
            semantic[semantic_name] = semantic[semantic_name].strip()
    return semantic


def _declaration_id(declaration: object) -> str:
    field_id = getattr(declaration, "id", None)
    _field_facts(field_id)
    return field_id


def _field_facts(field_id: object) -> AioPromptFieldFacts:
    if field_id not in _FIELD_FACTS:
        raise ValueError("Unknown AIO prompt field.")
    return _FIELD_FACTS[field_id]  # type: ignore[index]
    def __init__(self) -> None:
        super().__init__(AIO_CURRENT_PROMPT_SCHEMA)
