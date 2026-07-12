// Browser ownership for the inactive managed Ideogram prompt-builder slice.

import {
  aioManagedNodeType,
  aioManagedOutgoingTargets,
} from "./aio_managed_privacy_graph.js";

export const AIO_BUILDER_STATE_FIELD_ID = "ideogram-builder-state";
export const AIO_BUILDER_STATE_PROPERTY = "aio_ideogram4_prompt_builder_state";
export const AIO_BUILDER_WORKFLOW_STATE_KEY = "aio_ideogram4_prompt_builder";
export const AIO_BUILDER_LEGACY_WORKFLOW_STATE_KEY = "ideo";

export const AIO_BUILDER_WIDGET_FIELD_IDS = Object.freeze({
  high_level_description: "ideogram-builder-high-level-description",
  background: "ideogram-builder-background",
  photo: "ideogram-builder-photo",
  art_style: "ideogram-builder-art-style",
  aesthetics: "ideogram-builder-aesthetics",
  lighting: "ideogram-builder-lighting",
  medium: "ideogram-builder-medium",
  style_palette_data: "ideogram-builder-style-palette-data",
  elements_data: "ideogram-builder-elements-data",
  import_json: "ideogram-builder-import-json",
});

const BUILDER_NODE = "AIOIdeogram4PromptBuilder";
const GENERATE_NODE = "AIOImageGenerate";
const SETTINGS_NODES = new Set(["AIOIdeogram4Settings", "AIOKrea2Settings"]);
const PROTECTED_VALUES = "__aioManagedBuilderProtectedValues";
const FIELD_BY_ID = Object.freeze(Object.fromEntries(
  Object.entries(AIO_BUILDER_WIDGET_FIELD_IDS).map(([widget, fieldId]) => [fieldId, { widget }]),
));
const ALL_FIELD_IDS = Object.freeze([
  ...Object.values(AIO_BUILDER_WIDGET_FIELD_IDS),
  AIO_BUILDER_STATE_FIELD_ID,
]);

function fail() {
  throw new Error("PRIVACY_AIO_BUILDER_STATE_INVALID");
}

function clone(value) {
  if (value === undefined) return undefined;
  return structuredClone(value);
}

function equal(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function connectedGenerateNodes(builder) {
  if (aioManagedNodeType(builder) !== BUILDER_NODE) return [];
  const found = [];
  const visited = new Set([builder]);
  const queue = [builder];
  while (queue.length) {
    for (const target of aioManagedOutgoingTargets(queue.shift())) {
      if (visited.has(target)) continue;
      visited.add(target);
      if (aioManagedNodeType(target) === GENERATE_NODE) {
        found.push(target);
      } else if (SETTINGS_NODES.has(aioManagedNodeType(target))) {
        queue.push(target);
      }
    }
  }
  return found;
}

function widget(node, name) {
  const result = node?.widgets?.find((item) => item?.name === name);
  if (!result) fail();
  return result;
}

function fieldFacts(context) {
  const fieldId = context?.fieldId ?? context?.id;
  if (fieldId === AIO_BUILDER_STATE_FIELD_ID) return { fieldId, state: true };
  const facts = FIELD_BY_ID[fieldId];
  if (!facts) fail();
  if (context?.location?.name !== undefined && context.location.name !== facts.widget) fail();
  return { ...facts, fieldId, state: false };
}

function editorApi(node) {
  const api = node?._aioIdeogram4EditorApi;
  if (!api || typeof api.flushManagedEdits !== "function"
      || typeof api.applyManagedState !== "function"
      || typeof api.clearManagedState !== "function"
      || typeof api.setManagedEditHandler !== "function") fail();
  return api;
}

function protectedValues(node) {
  node[PROTECTED_VALUES] ||= Object.create(null);
  return node[PROTECTED_VALUES];
}

function stateMirrors(node) {
  return [
    node?.properties?.[AIO_BUILDER_STATE_PROPERTY],
    node?._aioIdeogram4PendingWorkflowInfo?.[AIO_BUILDER_WORKFLOW_STATE_KEY],
    node?._aioIdeogram4PendingWorkflowInfo?.[AIO_BUILDER_LEGACY_WORKFLOW_STATE_KEY],
  ].filter((value) => value !== undefined && value !== null);
}

function readStateProtected(node) {
  const remembered = protectedValues(node)[AIO_BUILDER_STATE_FIELD_ID];
  const values = remembered === undefined ? stateMirrors(node) : [remembered, ...stateMirrors(node)];
  if (!values.length || values.some((value) => !equal(value, values[0]))) fail();
  return clone(values[0]);
}

function writeStateProtected(node, value) {
  node.properties ||= {};
  node._aioIdeogram4PendingWorkflowInfo ||= {};
  node.properties[AIO_BUILDER_STATE_PROPERTY] = clone(value);
  node._aioIdeogram4PendingWorkflowInfo[AIO_BUILDER_WORKFLOW_STATE_KEY] = clone(value);
  node._aioIdeogram4PendingWorkflowInfo[AIO_BUILDER_LEGACY_WORKFLOW_STATE_KEY] = clone(value);
  protectedValues(node)[AIO_BUILDER_STATE_FIELD_ID] = clone(value);
}

function restoreProtectedWidgetLocations(node) {
  const values = protectedValues(node);
  for (const [widgetName, fieldId] of Object.entries(AIO_BUILDER_WIDGET_FIELD_IDS)) {
    if (fieldId in values) widget(node, widgetName).value = clone(values[fieldId]);
  }
}

function flushState(node) {
  const state = editorApi(node).flushManagedEdits();
  if (!state || typeof state !== "object" || Array.isArray(state)
      || !state.widgets || typeof state.widgets !== "object") fail();
  const result = clone(state);
  restoreProtectedWidgetLocations(node);
  return result;
}

function applyRuntimeState(node, state) {
  editorApi(node).applyManagedState(clone(state));
  restoreProtectedWidgetLocations(node);
}

function declaredMode(node) {
  const value = widget(node, "privacy_mode").value;
  if (value === false) return "public";
  if (value === true) return "private";
  return "inherit";
}

function evidence(node) {
  const suffix = String(node?.id ?? "unknown").replace(/[^A-Za-z0-9._-]/g, "-");
  return `aio-generate-${suffix}`;
}

export function createAioBuilderModeBrowserAdapter() {
  return {
    readDeclaredMode(node) {
      if (aioManagedNodeType(node) !== BUILDER_NODE) fail();
      return declaredMode(node);
    },
    readModeFacts(node) {
      if (aioManagedNodeType(node) !== BUILDER_NODE) fail();
      return {
        upstream: connectedGenerateNodes(node).map((candidate) => ({
          sourceId: evidence(candidate),
          mode: declaredMode(candidate) === "public" ? "public" : "private",
        })),
      };
    },
    writeDeclaredMode(node, mode) {
      if (aioManagedNodeType(node) !== BUILDER_NODE || !["inherit", "private", "public"].includes(mode)) fail();
      widget(node, "privacy_mode").value = mode === "inherit" ? undefined : mode === "private";
    },
    reconcileNode(node) {
      if (aioManagedNodeType(node) !== BUILDER_NODE) fail();
    },
    reconcileNodeDefinition() {},
    onPrivacySessionChange() {},
  };
}

export function createAioBuilderWorkflowBrowserAdapter({ workflowHandle = null } = {}) {
  let locked = false;

  function markGenerationEdited(node) {
    if (typeof workflowHandle?.markEdited !== "function") fail();
    flushState(node);
    for (const fieldId of ALL_FIELD_IDS) workflowHandle.markEdited(node, fieldId);
  }

  return {
    normalize(node, context) {
      const field = fieldFacts(context);
      const state = flushState(node);
      if (field.state) {
        if (["private", "public"].includes(context?.effectiveMode)) {
          state.widgets.privacy_mode = context.effectiveMode === "private";
        }
        return state;
      }
      if (!(field.widget in state.widgets) || typeof state.widgets[field.widget] !== "string") fail();
      return { value: state.widgets[field.widget] };
    },
    readProtected(node, context) {
      const field = fieldFacts(context);
      if (field.state) return readStateProtected(node);
      const remembered = protectedValues(node)[field.fieldId];
      return clone(remembered === undefined ? widget(node, field.widget).value : remembered);
    },
    writeProtected(node, protectedValue, context) {
      if (protectedValue === undefined || protectedValue === null) fail();
      const field = fieldFacts(context);
      if (field.state) {
        writeStateProtected(node, protectedValue);
        return;
      }
      protectedValues(node)[field.fieldId] = clone(protectedValue);
      widget(node, field.widget).value = clone(protectedValue);
    },
    apply(node, value, context) {
      const field = fieldFacts(context);
      if (field.state) {
        applyRuntimeState(node, value);
        return;
      }
      const state = flushState(node);
      const plaintext = value && typeof value === "object" && Object.keys(value).length === 1
        ? value.value : value;
      if (typeof plaintext !== "string") fail();
      state.widgets[field.widget] = plaintext;
      applyRuntimeState(node, state);
    },
    clear(node, context) {
      const field = fieldFacts(context);
      if (field.state) {
        editorApi(node).clearManagedState();
        restoreProtectedWidgetLocations(node);
        return;
      }
      const state = flushState(node);
      state.widgets[field.widget] = "";
      applyRuntimeState(node, state);
    },
    reconcileNode(node) {
      if (aioManagedNodeType(node) !== BUILDER_NODE) fail();
      editorApi(node).setManagedEditHandler(() => markGenerationEdited(node));
      if (locked) {
        editorApi(node).clearManagedState();
        restoreProtectedWidgetLocations(node);
      }
    },
    serializeForWorkflow(node, output) {
      if (!output || typeof output !== "object" || !Array.isArray(output.widgets_values)) fail();
      const values = protectedValues(node);
      for (const [widgetName, fieldId] of Object.entries(AIO_BUILDER_WIDGET_FIELD_IDS)) {
        if (!(fieldId in values)) fail();
        const index = node.widgets?.indexOf(widget(node, widgetName));
        if (!Number.isInteger(index) || index < 0 || index >= output.widgets_values.length) fail();
        output.widgets_values[index] = clone(values[fieldId]);
      }
      const state = readStateProtected(node);
      output.properties ||= {};
      output.properties[AIO_BUILDER_STATE_PROPERTY] = clone(state);
      output[AIO_BUILDER_WORKFLOW_STATE_KEY] = clone(state);
      output[AIO_BUILDER_LEGACY_WORKFLOW_STATE_KEY] = clone(state);
    },
    onPrivacySessionChange(event) {
      locked = event?.state === "locked" || event?.locked === true;
    },
  };
}
