"""Managed privacy adapters for the inactive Ideogram prompt-builder slice."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass

from helto_privacy import (
    AIO_V1_JSON_KEY_IMPORT_ID,
    AIO_V1_READER_ID,
    FieldLocation,
    FieldLocationKind,
    LegacyKeyFormat,
    LegacyKeyImportBinding,
    LegacyLocationKind,
    LegacyReaderBinding,
    MigrationVerification,
    ProtectedField,
)


AIO_BUILDER_CURRENT_SCHEMA = "helto.aio-ideogram4-builder.v2"
BUILDER_NODE_TYPE = "AIOIdeogram4PromptBuilder"
BUILDER_SCOPE_ID = "ideogram-builder"
BUILDER_WORKFLOW_RESOURCE_ID = "ideogram-builder-workflow"
BUILDER_EXECUTION_RESOURCE_ID = "ideogram-builder-execution"
BUILDER_PROJECTION_ID = "ideogram-builder"

BUILDER_MODE_ADAPTER_ID = "ideogram-builder-mode-state"
BUILDER_MODE_BROWSER_ADAPTER_ID = "ideogram-builder-mode-browser"
BUILDER_WORKFLOW_ADAPTER_ID = "ideogram-builder-workflow-state"
BUILDER_WORKFLOW_BROWSER_ADAPTER_ID = "ideogram-builder-workflow-browser"
BUILDER_PROJECTION_ADAPTER_ID = "ideogram-builder-execution-projection"
BUILDER_DISPATCH_ADAPTER_ID = "ideogram-builder-execution-dispatch"
BUILDER_OPERATION_ADAPTER_ID = "ideogram-builder-protected-operations"

BUILDER_STATE_FIELD_ID = "ideogram-builder-state"
BUILDER_STATE_PROPERTY = "aio_ideogram4_prompt_builder_state"
BUILDER_WORKFLOW_STATE_KEY = "aio_ideogram4_prompt_builder"
BUILDER_LEGACY_WORKFLOW_STATE_KEY = "ideo"

BUILDER_SENSITIVE_WIDGETS = (
    "high_level_description",
    "background",
    "photo",
    "art_style",
    "aesthetics",
    "lighting",
    "medium",
    "style_palette_data",
    "elements_data",
    "import_json",
)
BUILDER_WIDGET_FIELD_IDS = {
    widget_name: f"ideogram-builder-{widget_name.replace('_', '-')}"
    for widget_name in BUILDER_SENSITIVE_WIDGETS
}
BUILDER_EXECUTION_FIELD_IDS = (
    *BUILDER_WIDGET_FIELD_IDS.values(),
    BUILDER_STATE_FIELD_ID,
)
BUILDER_CONTROL_WIDGETS = (
    "max side",
    "aspect ratio",
    "multiple value",
    "privacy_mode",
    "style",
    "import_mode",
    "output_format",
    "coord_mode",
    "bbox_order",
    "bg_brightness",
)


def builder_legacy_binding_id(field_id: str) -> str:
    _field_widget(field_id, allow_state=True)
    return f"{field_id}-aio-v1"


def builder_legacy_key_binding_id(field_id: str) -> str:
    _field_widget(field_id, allow_state=True)
    return f"{field_id}-aio-json-key-v1"


def builder_protected_fields() -> tuple[ProtectedField, ...]:
    widget_fields = tuple(
        ProtectedField(
            field_id,
            BUILDER_WORKFLOW_RESOURCE_ID,
            BUILDER_SCOPE_ID,
            BUILDER_WORKFLOW_ADAPTER_ID,
            BUILDER_WORKFLOW_BROWSER_ADAPTER_ID,
            (BUILDER_NODE_TYPE,),
            FieldLocation(FieldLocationKind.WIDGET, widget_name),
            AIO_BUILDER_CURRENT_SCHEMA,
            field_id,
            legacy_reader_ids=(AIO_V1_READER_ID,),
            execution=True,
        )
        for widget_name, field_id in BUILDER_WIDGET_FIELD_IDS.items()
    )
    return (
        *widget_fields,
        ProtectedField(
            BUILDER_STATE_FIELD_ID,
            BUILDER_WORKFLOW_RESOURCE_ID,
            BUILDER_SCOPE_ID,
            BUILDER_WORKFLOW_ADAPTER_ID,
            BUILDER_WORKFLOW_BROWSER_ADAPTER_ID,
            (BUILDER_NODE_TYPE,),
            FieldLocation(FieldLocationKind.PROPERTY, BUILDER_STATE_PROPERTY),
            AIO_BUILDER_CURRENT_SCHEMA,
            BUILDER_STATE_FIELD_ID,
            legacy_reader_ids=(AIO_V1_READER_ID,),
            execution=True,
            mirror_locations=(
                FieldLocation(FieldLocationKind.BLOB, BUILDER_WORKFLOW_STATE_KEY),
                FieldLocation(FieldLocationKind.BLOB, BUILDER_LEGACY_WORKFLOW_STATE_KEY),
            ),
        ),
    )


def builder_legacy_bindings() -> tuple[LegacyReaderBinding, ...]:
    return tuple(
        LegacyReaderBinding(
            builder_legacy_binding_id(field_id),
            AIO_V1_READER_ID,
            BUILDER_WORKFLOW_RESOURCE_ID,
            LegacyLocationKind.WORKFLOW_FIELD,
            field_id,
        )
        for field_id in BUILDER_EXECUTION_FIELD_IDS
    )


def builder_legacy_key_imports() -> tuple[LegacyKeyImportBinding, ...]:
    return tuple(
        LegacyKeyImportBinding(
            builder_legacy_key_binding_id(field_id),
            AIO_V1_JSON_KEY_IMPORT_ID,
            BUILDER_WORKFLOW_RESOURCE_ID,
            LegacyLocationKind.WORKFLOW_FIELD,
            field_id,
            LegacyKeyFormat.JSON,
        )
        for field_id in BUILDER_EXECUTION_FIELD_IDS
    )


class AioBuilderModeAdapter:
    """Map the builder's legacy boolean while missing state inherits private."""

    def __init__(self, declarations: Mapping[str, object] | None = None) -> None:
        self._declarations = dict(declarations or {})

    def read_declared_mode(self, scope_id: str) -> str:
        if scope_id != BUILDER_SCOPE_ID:
            raise ValueError("Unknown AIO builder privacy scope.")
        if scope_id not in self._declarations:
            return "inherit"
        value = self._declarations[scope_id]
        if value in {False, "public"}:
            return "public"
        if value in {None, "inherit"}:
            return "inherit"
        return "private"

    def write_declared_mode(self, scope_id: str, mode: object) -> None:
        if scope_id != BUILDER_SCOPE_ID or mode not in {"private", "public", "inherit"}:
            raise ValueError("Invalid AIO builder privacy declaration.")
        self._declarations[scope_id] = mode

    def prepare_mode_transition(self, *_args) -> None:
        return None

    def commit_mode_transition(self, *_args) -> None:
        return None

    def rollback_mode_transition(self, *_args) -> None:
        return None


class AioBuilderWorkflowStateAdapter:
    """Normalize builder widgets and its mirrored whole-editor state."""

    def capture(self, source: object, declaration: object) -> object:
        field_id = _declaration_id(declaration)
        if field_id != BUILDER_STATE_FIELD_ID:
            return copy.deepcopy(_lookup(source, _field_widget(field_id)))
        if isinstance(source, Mapping):
            properties = source.get("properties", source)
            workflow = source.get("workflow", source)
        else:
            properties = getattr(source, "properties", {})
            workflow = getattr(source, "workflow", {})
        candidates = (
            _mapping_get(properties, BUILDER_STATE_PROPERTY),
            _mapping_get(workflow, BUILDER_WORKFLOW_STATE_KEY),
            _mapping_get(workflow, BUILDER_LEGACY_WORKFLOW_STATE_KEY),
        )
        present = [candidate for candidate in candidates if candidate is not None]
        if not present or any(candidate != present[0] for candidate in present[1:]):
            raise ValueError("AIO builder state mirrors are missing or inconsistent.")
        return copy.deepcopy(present[0])

    def normalize(self, value: object, declaration: object) -> object:
        field_id = _declaration_id(declaration)
        if field_id == BUILDER_STATE_FIELD_ID:
            return normalize_builder_state(value)
        return {"value": normalize_builder_text(value)}

    def apply_revealed(self, target: object, value: object, declaration: object) -> None:
        field_id = _declaration_id(declaration)
        if field_id == BUILDER_STATE_FIELD_ID:
            _write_builder_state(target, normalize_builder_state(value))
        else:
            _assign(target, _field_widget(field_id), normalize_builder_text(value))

    def clear_plaintext(self, target: object, declaration: object) -> None:
        field_id = _declaration_id(declaration)
        if field_id == BUILDER_STATE_FIELD_ID:
            _write_builder_state(target, {})
        else:
            _assign(target, _field_widget(field_id), "")

    def prepare_mode_transition(self, *_args) -> None:
        return None

    def commit_mode_transition(self, *_args) -> None:
        return None

    def rollback_mode_transition(self, *_args) -> None:
        return None


class AioBuilderExecutionProjectionAdapter:
    """Validate one builder generation and project product construction inputs."""

    def project(self, fields: Mapping[str, object], declaration: object) -> dict[str, object]:
        if getattr(declaration, "id", None) != BUILDER_PROJECTION_ID:
            raise ValueError("Unknown AIO builder projection.")
        return project_builder_generation(fields)


class AioBuilderExecutionDispatchAdapter:
    def dispatch(self, value: object, context: object, cancellation: object) -> object:
        callback = context.get("dispatch") if isinstance(context, Mapping) else None
        if not callable(callback):
            raise ValueError("AIO builder execution dispatch is unavailable.")
        checkpoint = getattr(cancellation, "checkpoint", None)
        if callable(checkpoint):
            checkpoint()
        return callback(copy.deepcopy(value))


class AioBuilderOperationAdapter:
    def __init__(self, dispatcher: Callable[[object, object], object] | None = None) -> None:
        self._dispatcher = dispatcher

    def invoke(self, payload: object, context: object) -> object:
        if not callable(self._dispatcher):
            raise ValueError("AIO protected builder operation is unavailable.")
        return self._dispatcher(payload, context)


@dataclass(slots=True)
class AioBuilderMigrationTransaction:
    """Rewrite every widget and whole-state mirror under one shared receipt."""

    workflow: object
    store: MutableMapping[str, object]
    protect_authorization: object
    reveal_authorization: object
    original: object = None
    staged: object = None

    def capture_original(self) -> object:
        self.original = copy.deepcopy(self.store)
        return copy.deepcopy(self.original)

    def stage_current(self, normalized: object) -> None:
        if not isinstance(normalized, Mapping) or set(normalized) != set(BUILDER_EXECUTION_FIELD_IDS):
            raise ValueError("AIO builder migration set is incomplete.")
        project_builder_generation(normalized)
        self.staged = {
            field_id: self.workflow.protect(
                field_id,
                normalized[field_id],
                self.protect_authorization,
            ).envelope
            for field_id in BUILDER_EXECUTION_FIELD_IDS
        }

    def stage_durable_adjuncts(self, _normalized: object) -> None:
        return None

    def commit(self) -> None:
        if not isinstance(self.staged, Mapping):
            raise ValueError("AIO builder migration has not been staged.")
        widgets = self.store.setdefault("widgets", {})
        properties = self.store.setdefault("properties", {})
        workflow = self.store.setdefault("workflow", {})
        if not all(isinstance(item, MutableMapping) for item in (widgets, properties, workflow)):
            raise ValueError("AIO builder migration store is invalid.")
        for widget_name, field_id in BUILDER_WIDGET_FIELD_IDS.items():
            widgets[widget_name] = copy.deepcopy(self.staged[field_id])
        state_envelope = copy.deepcopy(self.staged[BUILDER_STATE_FIELD_ID])
        properties[BUILDER_STATE_PROPERTY] = copy.deepcopy(state_envelope)
        workflow[BUILDER_WORKFLOW_STATE_KEY] = copy.deepcopy(state_envelope)
        workflow[BUILDER_LEGACY_WORKFLOW_STATE_KEY] = copy.deepcopy(state_envelope)

    def read_back(self) -> MigrationVerification:
        if not isinstance(self.staged, Mapping):
            raise ValueError("AIO builder migration has not been staged.")
        widgets = self.store.get("widgets", {})
        properties = self.store.get("properties", {})
        workflow = self.store.get("workflow", {})
        mirrors = (
            _mapping_get(properties, BUILDER_STATE_PROPERTY),
            _mapping_get(workflow, BUILDER_WORKFLOW_STATE_KEY),
            _mapping_get(workflow, BUILDER_LEGACY_WORKFLOW_STATE_KEY),
        )
        if any(item != mirrors[0] for item in mirrors[1:]):
            raise ValueError("AIO builder state mirrors are inconsistent.")
        normalized = {}
        current = True
        for widget_name, field_id in BUILDER_WIDGET_FIELD_IDS.items():
            envelope = _mapping_get(widgets, widget_name)
            result = self.workflow.reveal(field_id, envelope, self.reveal_authorization)
            normalized[field_id] = result.value
            current = current and _mapping_get(envelope, "schema") == AIO_BUILDER_CURRENT_SCHEMA
        state_result = self.workflow.reveal(
            BUILDER_STATE_FIELD_ID,
            mirrors[0],
            self.reveal_authorization,
        )
        normalized[BUILDER_STATE_FIELD_ID] = state_result.value
        current = current and _mapping_get(mirrors[0], "schema") == AIO_BUILDER_CURRENT_SCHEMA
        project_builder_generation(normalized)
        return MigrationVerification(normalized, current, True)

    def rollback(self, original: object) -> None:
        if not isinstance(original, Mapping):
            raise ValueError("AIO builder migration original is invalid.")
        self.store.clear()
        self.store.update(copy.deepcopy(original))
        self.staged = None

    def finalize(self, _original: object) -> None:
        self.original = None
        self.staged = None


def normalize_builder_text(value: object) -> str:
    if isinstance(value, Mapping) and set(value) == {"value"}:
        value = value["value"]
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("AIO builder text state is invalid.")
    return value


def normalize_builder_state(value: object) -> dict[str, object]:
    if isinstance(value, Mapping) and set(value) == {"value"}:
        value = value["value"]
    if not isinstance(value, Mapping):
        raise ValueError("AIO builder whole state is invalid.")
    try:
        normalized = json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        raise ValueError("AIO builder whole state is invalid.") from None
    if not isinstance(normalized, dict) or not isinstance(normalized.get("widgets", {}), dict):
        raise ValueError("AIO builder whole state is invalid.")
    normalized["widgets"] = normalized.get("widgets", {})
    for key in ("elements", "style_palette"):
        if key in normalized and not isinstance(normalized[key], list):
            raise ValueError("AIO builder whole state collection is invalid.")
    return normalized


def project_builder_generation(fields: Mapping[str, object]) -> dict[str, object]:
    """Validate and project one coherent widget plus whole-editor generation."""

    if set(fields) != set(BUILDER_EXECUTION_FIELD_IDS):
        raise ValueError("AIO builder snapshot is incomplete.")
    state = normalize_builder_state(fields[BUILDER_STATE_FIELD_ID])
    widgets = state.get("widgets")
    if not isinstance(widgets, dict):
        raise ValueError("AIO builder state widgets are unavailable.")
    if any(name not in widgets for name in BUILDER_CONTROL_WIDGETS):
        raise ValueError("AIO builder semantic controls are incomplete.")
    inputs = copy.deepcopy(widgets)
    for widget_name, field_id in BUILDER_WIDGET_FIELD_IDS.items():
        value = normalize_builder_text(fields[field_id])
        if widget_name not in widgets or normalize_builder_text(widgets[widget_name]) != value:
            raise ValueError("AIO builder field mirrors are inconsistent.")
        inputs[widget_name] = value
    if _json_list(inputs["style_palette_data"], "style_palette") != state.get(
        "style_palette", []
    ):
        raise ValueError("AIO builder palette mirrors are inconsistent.")
    if _builder_elements(inputs["elements_data"]) != state.get("elements", []):
        raise ValueError("AIO builder element mirrors are inconsistent.")
    for name in ("bg_brightness", "output_format", "coord_mode", "bbox_order"):
        if state.get(name) != inputs[name]:
            raise ValueError("AIO builder control mirrors are inconsistent.")
    state["widgets"] = inputs
    return state


def _json_list(value: object, label: str) -> list[object]:
    if value is None or value == "":
        return []
    if not isinstance(value, str):
        raise ValueError(f"AIO builder {label} state is invalid.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError(f"AIO builder {label} state is invalid.") from None
    if not isinstance(parsed, list):
        raise ValueError(f"AIO builder {label} state is invalid.")
    return parsed


def _builder_elements(value: object) -> list[object]:
    if value is None or value == "":
        return []
    if not isinstance(value, str):
        raise ValueError("AIO builder element state is invalid.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError("AIO builder element state is invalid.") from None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        elements = parsed.get("elements", parsed.get("boxes", []))
        if isinstance(elements, list):
            return elements
    raise ValueError("AIO builder element state is invalid.")


def dispatch_aio_builder_execution(reference: object, context: Mapping[str, object]) -> object:
    from .managed_privacy_execution import dispatch_aio_managed_execution

    return dispatch_aio_managed_execution(
        reference,
        BUILDER_EXECUTION_RESOURCE_ID,
        context,
        cache_result=False,
    )


def _declaration_id(declaration: object) -> str:
    field_id = getattr(declaration, "id", None)
    _field_widget(field_id, allow_state=True)
    return field_id


def _field_widget(field_id: object, *, allow_state: bool = False) -> str:
    if allow_state and field_id == BUILDER_STATE_FIELD_ID:
        return ""
    for widget_name, candidate in BUILDER_WIDGET_FIELD_IDS.items():
        if field_id == candidate:
            return widget_name
    raise ValueError("Unknown AIO builder field.")


def _mapping_get(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, Mapping) else None


def _lookup(source: object, name: str) -> object:
    if isinstance(source, Mapping):
        widgets = source.get("widgets")
        if isinstance(widgets, Mapping) and name in widgets:
            return widgets[name]
        return source.get(name)
    return getattr(source, name)


def _assign(target: object, name: str, value: object) -> None:
    if isinstance(target, MutableMapping):
        widgets = target.get("widgets")
        if isinstance(widgets, MutableMapping):
            widgets[name] = value
        else:
            target[name] = value
        return
    setattr(target, name, value)


def _write_builder_state(target: object, value: object) -> None:
    state = copy.deepcopy(value)
    if isinstance(target, MutableMapping):
        properties = target.setdefault("properties", {})
        workflow = target.setdefault("workflow", {})
    else:
        properties = getattr(target, "properties")
        workflow = getattr(target, "workflow")
    if not isinstance(properties, MutableMapping) or not isinstance(workflow, MutableMapping):
        raise ValueError("AIO builder state target is invalid.")
    properties[BUILDER_STATE_PROPERTY] = copy.deepcopy(state)
    workflow[BUILDER_WORKFLOW_STATE_KEY] = copy.deepcopy(state)
    workflow[BUILDER_LEGACY_WORKFLOW_STATE_KEY] = copy.deepcopy(state)
