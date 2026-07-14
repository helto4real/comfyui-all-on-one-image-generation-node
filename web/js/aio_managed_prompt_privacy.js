// Browser-owned field locations for the inactive managed Generate/Krea slice.

import {
  aioManagedGraphLink,
  aioManagedGraphNodes,
  aioManagedNodeType,
} from "./aio_managed_privacy_graph.js";
import {
  createAioExternalWorkflowTransition,
  isAioCurrentModeEnvelope,
  parseAioModeTransitionStorage,
} from "./aio_managed_mode_transition.js";

export const AIO_GENERATE_POSITIVE_FIELD_ID = "generate-positive-prompt";
export const AIO_GENERATE_NEGATIVE_FIELD_ID = "generate-negative-prompt";
export const AIO_KREA_INPAINT_FIELD_ID = "krea-inpaint-positive-prompt";

const GENERATE_NODE = "AIOImageGenerate";
const KREA_NODE = "AIOKrea2Settings";
const AIO_PROMPT_SCHEMA = "helto.aio-image-generate.v2";
const PROTECTED_VALUES = "__aioManagedPromptProtectedValues";
const PLAINTEXT_VALUES = "__aioManagedPromptPlaintextValues";
const PRIVACY_STYLE_ID = "aio-managed-prompt-privacy-style";
const PROMPT_FIELD_CLASS = "aio-managed-prompt-field";
const PRIVATE_FIELD_CLASS = "aio-managed-private-field";
const MASKED_PROMPT_VALUE = "••••••••";
const IMPLEMENTATION_WIDGET_NAMES = Object.freeze([
  "privacy_mode_reference",
  "private_execution",
]);
const FIELD_FACTS = Object.freeze({
  [AIO_GENERATE_POSITIVE_FIELD_ID]: Object.freeze({
    nodeType: GENERATE_NODE,
    widget: "positive_prompt",
  }),
  [AIO_GENERATE_NEGATIVE_FIELD_ID]: Object.freeze({
    nodeType: GENERATE_NODE,
    widget: "negative_prompt",
  }),
  [AIO_KREA_INPAINT_FIELD_ID]: Object.freeze({
    nodeType: KREA_NODE,
    widget: "inpaint_positive_prompt",
  }),
});

function fail() {
  throw new Error("PRIVACY_AIO_PROMPT_STATE_INVALID");
}

function connectedGenerateNodes(kreaNode) {
  if (aioManagedNodeType(kreaNode) !== KREA_NODE) return [];
  const targets = [];
  for (const linkId of kreaNode?.outputs?.flatMap((output) => output?.links || []) || []) {
    const link = aioManagedGraphLink(kreaNode, linkId);
    const target = aioManagedGraphNodes(kreaNode).find(
      (candidate) => String(candidate?.id) === String(link?.target_id),
    );
    const inputName = target?.inputs?.[link?.target_slot]?.name;
    if (aioManagedNodeType(target) !== GENERATE_NODE || inputName !== "model_settings") continue;
    if (!targets.includes(target)) targets.push(target);
  }
  return targets;
}

function connectedKreaNodes(generateNode) {
  if (aioManagedNodeType(generateNode) !== GENERATE_NODE) return [];
  return aioManagedGraphNodes(generateNode).filter(
    (candidate) => connectedGenerateNodes(candidate).includes(generateNode),
  );
}

function generateDeclaredMode(node) {
  const value = node?.widgets?.find((item) => item?.name === "privacy_mode")?.value;
  if (value === false) return "public";
  if (value === true) return "private";
  return "inherit";
}

function modeEvidenceSource(node) {
  const suffix = String(node?.id ?? "unknown").replace(/[^A-Za-z0-9._-]/g, "-");
  return `aio-generate-${suffix}`;
}

function facts(context) {
  const declared = context?.field;
  const fieldId = declared?.id ?? context?.fieldId ?? context?.id;
  const value = FIELD_FACTS[fieldId];
  if (!value) fail();
  const location = declared?.location ?? context?.location;
  if (location?.name !== undefined && location.name !== value.widget) fail();
  return { ...value, fieldId };
}

function widget(node, context) {
  const field = facts(context);
  if (aioManagedNodeType(node) !== field.nodeType) fail();
  const found = node?.widgets?.find((item) => item?.name === field.widget);
  if (!found) fail();
  return found;
}

function serializedWidgetIndex(node, target) {
  let index = 0;
  for (const candidate of node?.widgets || []) {
    const serialized = candidate?.serialize !== false
      && candidate?.options?.serialize !== false;
    if (candidate === target) return serialized ? index : -1;
    if (serialized) index += 1;
  }
  return -1;
}

function unwrap(value) {
  if (value && typeof value === "object" && !Array.isArray(value)
      && Object.keys(value).length === 1 && "value" in value) return value.value;
  return value;
}

function normalize(value) {
  const result = unwrap(value);
  if (result === null || result === undefined) return "";
  if (typeof result !== "string") fail();
  return result;
}

function setDomText(target, value) {
  const text = normalize(value);
  for (const element of [target.inputEl, target.element, target.inputElement, target.textarea]) {
    if (!element) continue;
    if ("value" in element) element.value = text;
    else if (element.isContentEditable) element.textContent = text;
  }
}

function widgetTextElements(target) {
  return [target?.inputEl, target?.element, target?.inputElement, target?.textarea]
    .filter((element) => element && (
      typeof element.value === "string" || element.isContentEditable
    ));
}

function installPrivacyStyles() {
  if (typeof document === "undefined" || document.getElementById(PRIVACY_STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = PRIVACY_STYLE_ID;
  style.textContent = `
    .${PROMPT_FIELD_CLASS} {
      background: var(--helto-surface-2) !important;
      border-color: var(--helto-border-strong) !important;
    }
    .${PRIVATE_FIELD_CLASS} {
      background: var(--helto-surface-2) !important;
      border-color: var(--helto-border-strong) !important;
      color: transparent !important;
      -webkit-text-fill-color: transparent !important;
      caret-color: transparent !important;
      text-shadow: none !important;
    }
    .${PRIVATE_FIELD_CLASS}::placeholder {
      color: transparent !important;
      -webkit-text-fill-color: transparent !important;
    }
    .${PRIVATE_FIELD_CLASS}:hover,
    .${PRIVATE_FIELD_CLASS}:focus,
    .${PRIVATE_FIELD_CLASS}:focus-visible {
      background: var(--helto-surface-2) !important;
      border-color: var(--helto-border-strong) !important;
      color: inherit !important;
      -webkit-text-fill-color: currentColor !important;
      caret-color: auto !important;
    }
  `;
  document.head.appendChild(style);
}

function privatePresentation(node) {
  if (aioManagedNodeType(node) === GENERATE_NODE) {
    return generateDeclaredMode(node) !== "public";
  }
  if (aioManagedNodeType(node) === KREA_NODE) {
    return connectedGenerateNodes(node).some(
      (candidate) => generateDeclaredMode(candidate) !== "public",
    );
  }
  return false;
}

function nodeFieldIds(node) {
  return Object.entries(FIELD_FACTS)
    .filter(([, field]) => field.nodeType === aioManagedNodeType(node))
    .map(([fieldId]) => fieldId);
}

function hideImplementationWidgets(node) {
  for (const name of IMPLEMENTATION_WIDGET_NAMES) {
    const target = node?.widgets?.find((item) => item?.name === name);
    if (!target) continue;
    target.hidden = true;
    target.type = "hidden";
    target.computeSize = () => [0, -4];
  }
}

function patchPromptDraw(node, target) {
  if (!target || target.__aioManagedPrivacyDrawPatched) return;
  const original = target.draw;
  if (typeof original === "function") {
    target.draw = function aioManagedPromptDraw() {
      if (!privatePresentation(node)) return original.apply(this, arguments);
      const value = this.value;
      this.value = MASKED_PROMPT_VALUE;
      try {
        return original.apply(this, arguments);
      } finally {
        this.value = value;
      }
    };
  }
  target.__aioManagedPrivacyDrawPatched = true;
}

function updatePrivacyPresentation(node) {
  const masked = privatePresentation(node);
  for (const fieldId of nodeFieldIds(node)) {
    const target = widget(node, { fieldId });
    patchPromptDraw(node, target);
    for (const element of widgetTextElements(target)) {
      element.classList?.add(PROMPT_FIELD_CLASS);
      element.classList?.toggle(PRIVATE_FIELD_CLASS, masked);
      element.setAttribute?.("data-aio-private", masked ? "true" : "false");
    }
  }
  node.__aioManagedPrivacyMasked = masked;
  node.setDirtyCanvas?.(true, true);
}

function protectedValues(node) {
  node[PROTECTED_VALUES] ||= Object.create(null);
  return node[PROTECTED_VALUES];
}

function plaintextValues(node) {
  node[PLAINTEXT_VALUES] ||= Object.create(null);
  return node[PLAINTEXT_VALUES];
}

function liveText(node, context) {
  const field = facts(context);
  const remembered = plaintextValues(node)[field.fieldId];
  if (typeof remembered === "string") return remembered;
  const target = widget(node, context);
  for (const element of [target.inputEl, target.element, target.inputElement, target.textarea]) {
    if (!element) continue;
    if (typeof element.value === "string") return element.value;
    if (element.isContentEditable) return String(element.textContent || "");
  }
  return normalize(target.value);
}

export function createAioPromptModeBrowserAdapter() {
  return {
    readDeclaredMode(node) {
      if (aioManagedNodeType(node) === KREA_NODE) {
        const upstream = connectedGenerateNodes(node);
        return upstream.length && upstream.every(
          (candidate) => generateDeclaredMode(candidate) === "public",
        ) ? "public" : "inherit";
      }
      if (aioManagedNodeType(node) !== GENERATE_NODE) fail();
      return generateDeclaredMode(node);
    },
    readModeFacts(node) {
      if (aioManagedNodeType(node) !== KREA_NODE) return {};
      return {
        upstream: connectedGenerateNodes(node).map((candidate) => ({
          sourceId: modeEvidenceSource(candidate),
          mode: generateDeclaredMode(candidate) === "public" ? "public" : "private",
        })),
      };
    },
    writeDeclaredMode(node, mode) {
      if (aioManagedNodeType(node) !== GENERATE_NODE || !["inherit", "private", "public"].includes(mode)) fail();
      const target = node.widgets?.find((item) => item?.name === "privacy_mode");
      if (!target) fail();
      target.value = mode === "inherit" ? undefined : mode === "private";
    },
    reconcileNode(node) {
      if (![GENERATE_NODE, KREA_NODE].includes(aioManagedNodeType(node))) fail();
    },
    reconcileNodeDefinition() {},
    onPrivacySessionChange() {},
  };
}

export function createAioPromptWorkflowBrowserAdapter({
  workflowHandle = null,
  app = null,
} = {}) {
  let locked = true;
  const owners = new Set();

  function notifyModeChange(node) {
    updatePrivacyPresentation(node);
    for (const candidate of connectedKreaNodes(node)) updatePrivacyPresentation(candidate);
    if (typeof workflowHandle?.notifyModeChange !== "function") return;
    let settlement;
    try {
      settlement = workflowHandle.notifyModeChange();
    } catch (error) {
      node.__aioManagedPrivacyError = String(error?.code || error?.message || "PRIVACY_MODE_STATE_UNAVAILABLE");
      throw error;
    }
    node.__aioManagedPrivacyModeSettlement = Promise.resolve(settlement).then(() => {
      node.__aioManagedPrivacyError = "";
      updatePrivacyPresentation(node);
      for (const candidate of connectedKreaNodes(node)) updatePrivacyPresentation(candidate);
    }).catch((error) => {
      node.__aioManagedPrivacyError = String(error?.code || error?.message || "PRIVACY_MODE_STATE_UNAVAILABLE");
      node.setDirtyCanvas?.(true, true);
    });
  }

  function bindModeChanges(node) {
    if (aioManagedNodeType(node) !== GENERATE_NODE) return;
    const target = node.widgets?.find((item) => item?.name === "privacy_mode");
    if (!target || target.__aioManagedPrivacyModeBound) return;
    const original = target.callback;
    target.callback = function aioManagedPrivacyModeChanged() {
      const result = original?.apply(this, arguments);
      notifyModeChange(node);
      return result;
    };
    target.__aioManagedPrivacyModeBound = true;
  }

  function recordEdit(node, fieldId, candidate = undefined) {
    transition.requireMutable();
    const context = { fieldId };
    const target = widget(node, context);
    const plaintext = normalize(candidate === undefined ? liveText(node, context) : candidate);
    plaintextValues(node)[fieldId] = plaintext;
    const protectedValue = protectedValues(node)[fieldId];
    if (typeof protectedValue === "string") target.value = protectedValue;
    setDomText(target, plaintext);
    if (typeof workflowHandle?.markEdited !== "function") fail();
    return workflowHandle.markEdited(node, fieldId);
  }

  function bindEdits(node, fieldId) {
    const target = widget(node, { fieldId });
    target.__aioManagedPrivacyEditBindings ||= new Set();
    if (!target.__aioManagedPrivacyEditBindings.has(fieldId)) {
      const original = target.callback;
      target.callback = function aioManagedPromptEdited(value) {
        transition.requireMutable();
        const result = original?.apply(this, arguments);
        recordEdit(node, fieldId, value);
        return result;
      };
      target.__aioManagedPrivacyEditBindings.add(fieldId);
    }
    for (const element of widgetTextElements(target)) {
      element.__aioManagedPrivacyEditBindings ||= new Set();
      if (element.__aioManagedPrivacyEditBindings.has(fieldId)) continue;
      const onEdited = () => {
        transition.requireMutable();
        return recordEdit(
          node,
          fieldId,
          "value" in element ? element.value : element.textContent,
        );
      };
      element.addEventListener?.("input", onEdited);
      element.addEventListener?.("change", onEdited);
      element.__aioManagedPrivacyEditBindings.add(fieldId);
    }
  }

  function applyValue(node, value, context) {
    const field = facts(context);
    const target = widget(node, context);
    const plaintext = normalize(value);
    plaintextValues(node)[field.fieldId] = plaintext;
    if (!(field.fieldId in protectedValues(node))) target.value = plaintext;
    setDomText(target, plaintext);
    updatePrivacyPresentation(node);
  }

  function clearValue(node, context) {
    const field = facts(context);
    plaintextValues(node)[field.fieldId] = "";
    const target = widget(node, context);
    if (!(field.fieldId in protectedValues(node))) target.value = "";
    setDomText(target, "");
    updatePrivacyPresentation(node);
  }

  function reconcileOwner(node) {
    if (!Object.values(FIELD_FACTS).some((field) => field.nodeType === aioManagedNodeType(node))) fail();
    owners.add(node);
    transition.synchronizeOwner(node);
    node.__aioManagedPrivacyLocked = locked;
    installPrivacyStyles();
    hideImplementationWidgets(node);
    bindModeChanges(node);
    for (const [fieldId, field] of Object.entries(FIELD_FACTS)) {
      if (field.nodeType === aioManagedNodeType(node)) bindEdits(node, fieldId);
    }
    updatePrivacyPresentation(node);
    if (!locked) return;
    for (const [fieldId, field] of Object.entries(FIELD_FACTS)) {
      if (field.nodeType !== aioManagedNodeType(node)) continue;
      plaintextValues(node)[fieldId] = "";
      setDomText(widget(node, { fieldId }), "");
    }
    updatePrivacyPresentation(node);
  }

  const transition = createAioExternalWorkflowTransition({
    app,
    owners,
    registerNode: reconcileOwner,
    readStorage(node, context) {
      const field = facts(context);
      const remembered = protectedValues(node)[field.fieldId];
      return typeof remembered === "string"
        ? remembered
        : String(widget(node, context).value || "");
    },
    writeStorage(node, value, context) {
      const field = facts(context);
      protectedValues(node)[field.fieldId] = value;
      widget(node, context).value = value;
    },
    readDetachedStorage(node, serializedNode, context) {
      if (!Array.isArray(serializedNode?.widgets_values)) fail();
      const target = widget(node, context);
      const index = serializedWidgetIndex(node, target);
      if (!Number.isInteger(index) || index < 0 || index >= serializedNode.widgets_values.length) fail();
      const value = serializedNode.widgets_values[index];
      if (typeof value !== "string") fail();
      return value;
    },
    reloadRuntime(node, value, context) {
      const payload = parseAioModeTransitionStorage(value, fail);
      if (isAioCurrentModeEnvelope(payload, AIO_PROMPT_SCHEMA)) clearValue(node, context);
      else applyValue(node, payload, context);
    },
    reconcileRuntime(node) {
      node.__aioManagedPrivacyLocked = locked;
    },
    fail,
  });

  return {
    normalize(node, context) {
      return { value: liveText(node, context) };
    },
    readProtected(node, context) {
      const field = facts(context);
      const remembered = protectedValues(node)[field.fieldId];
      return typeof remembered === "string" ? remembered : String(widget(node, context).value || "");
    },
    writeProtected(node, protectedValue, context) {
      if (typeof protectedValue !== "string") fail();
      const field = facts(context);
      const target = widget(node, context);
      const plaintext = liveText(node, context);
      plaintextValues(node)[field.fieldId] = plaintext;
      protectedValues(node)[field.fieldId] = protectedValue;
      transition.withInternalMutation(() => {
        target.value = protectedValue;
      });
      setDomText(target, locked ? "" : plaintext);
      updatePrivacyPresentation(node);
    },
    writePublic(node, context) {
      if (locked) fail();
      const field = facts(context);
      const target = widget(node, context);
      const plaintext = liveText(node, context);
      plaintextValues(node)[field.fieldId] = plaintext;
      protectedValues(node)[field.fieldId] = plaintext;
      transition.withInternalMutation(() => {
        target.value = plaintext;
      });
      setDomText(target, plaintext);
      updatePrivacyPresentation(node);
      return plaintext;
    },
    writeWorkflowProjection(node, serializedNode, protectedValue, context) {
      if (!serializedNode || !Array.isArray(serializedNode.widgets_values)) fail();
      const target = widget(node, context);
      const index = serializedWidgetIndex(node, target);
      if (!Number.isInteger(index) || index < 0 || index >= serializedNode.widgets_values.length) fail();
      serializedNode.widgets_values[index] = protectedValue;
    },
    apply(node, value, context) {
      applyValue(node, value, context);
    },
    clear(node, context) {
      clearValue(node, context);
    },
    reconcileNode(node) {
      reconcileOwner(node);
    },
    reconcileNodeDefinition() {},
    onPrivacySessionChange(snapshot) {
      locked = snapshot?.state !== "ready" && snapshot?.state !== "unlocked";
      for (const owner of owners) {
        owner.__aioManagedPrivacyLocked = locked;
        updatePrivacyPresentation(owner);
      }
    },
    ...transition,
  };
}
