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
const PRIVACY_UNAVAILABLE_CLASS = "aio-managed-privacy-unavailable";
const BOOTSTRAP_MODE_BOUND = "__aioManagedPromptBootstrapModeBound";
const MASKED_PROMPT_VALUE = "••••••••";
const PRESENTATION_RECONCILE_EPOCH = "__aioManagedPrivacyPresentationEpoch";
const PRESENTATION_RECONCILE_FRAMES = 240;
const EDIT_BINDING_RECONCILE_EPOCH = "__aioManagedPrivacyEditBindingEpoch";
const PRESENTATION_BINDINGS = "__aioManagedPrivacyPresentationBindings";
const PRESENTATION_POINTER_REVEAL = "__aioManagedPrivacyPointerReveal";
const PRESENTATION_KEYBOARD_REVEAL = "__aioManagedPrivacyKeyboardReveal";
const BOOTSTRAPPED_APPS = new WeakSet();
const TRACKED_PRESENTATION_REFERENCES = new Set();
const TRACKED_PRESENTATION_REFERENCE_BY_NODE = new WeakMap();
const TRACKED_PRESENTATION_UNAVAILABLE = new WeakMap();
let presentationObserver = null;
let presentationObserverScheduled = false;
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

function vueWidgetTextElements(node, target) {
  if (typeof document === "undefined" || typeof document.querySelectorAll !== "function") {
    return [];
  }
  const nodeId = String(node?.id ?? "");
  if (!nodeId) return [];
  const root = [...document.querySelectorAll("[data-node-id]")].find(
    (candidate) => String(candidate?.dataset?.nodeId ?? "") === nodeId,
  );
  if (!root || typeof root.querySelectorAll !== "function") return [];
  const name = String(target?.name || "");
  const row = [...root.querySelectorAll('[data-testid="node-widget"]')].find(
    (candidate) => [...(candidate?.querySelectorAll?.("label") || [])].some(
      (label) => String(label?.textContent || "").trim() === name,
    ),
  );
  if (!row || typeof row.querySelectorAll !== "function") return [];
  return [...row.querySelectorAll(
    'textarea, input:not([type="hidden"]), [contenteditable="true"]',
  )];
}

function widgetTextElements(node, target) {
  const elements = [
    target?.inputEl,
    target?.element,
    target?.inputElement,
    target?.textarea,
    ...vueWidgetTextElements(node, target),
  ].filter((element) => element && (
    typeof element.value === "string" || element.isContentEditable
  ));
  return [...new Set(elements)];
}

function hasMountedTextElement(elements) {
  return elements.some((element) => element.isConnected !== false);
}

function elementText(element) {
  if (typeof element?.value === "string") return element.value;
  if (element?.isContentEditable) return String(element.textContent || "");
  return "";
}

function writeElementText(element, value) {
  const text = normalize(value);
  if (elementText(element) === text) return;
  if ("value" in element) element.value = text;
  else if (element.isContentEditable) element.textContent = text;
}

function isProtectedPromptStorage(value) {
  if (typeof value !== "string" || !value.trim()) return false;
  try {
    return isAioCurrentModeEnvelope(JSON.parse(value), AIO_PROMPT_SCHEMA);
  } catch {
    return false;
  }
}

function presentationPlaintext(node, fieldId, target, elements) {
  const remembered = plaintextValues(node)[fieldId];
  if (typeof remembered === "string") return remembered;
  const candidates = [
    ...elements.map((element) => elementText(element)),
    target?.value,
  ];
  const candidate = candidates.find(
    (value) => typeof value === "string" && value !== MASKED_PROMPT_VALUE,
  );
  const plaintext = typeof candidate === "string" && !isProtectedPromptStorage(candidate)
    ? candidate
    : "";
  plaintextValues(node)[fieldId] = plaintext;
  return plaintext;
}

function presentationRevealActive(element) {
  return element?.[PRESENTATION_POINTER_REVEAL] === true
    || element?.[PRESENTATION_KEYBOARD_REVEAL] === true
    || element?.matches?.(":hover") === true;
}

function reconcileElementPresentation(node, fieldId, target, element, unavailable) {
  const masked = privatePresentation(node);
  const fieldUnavailable = unavailable && masked;
  const plaintext = presentationPlaintext(
    node,
    fieldId,
    target,
    widgetTextElements(node, target),
  );
  const reveal = masked
    && !fieldUnavailable
    && node.__aioManagedPrivacyLocked !== true
    && presentationRevealActive(element);
  writeElementText(element, masked && !reveal ? "" : plaintext);
}

function bindPresentationEvents(node, fieldId, target, element, unavailable) {
  element[PRESENTATION_BINDINGS] ||= new Set();
  if (element[PRESENTATION_BINDINGS].has(fieldId)) return;
  const reconcile = () => reconcileElementPresentation(
    node,
    fieldId,
    target,
    element,
    node.__aioManagedPrivacyUnavailable ?? unavailable,
  );
  element.addEventListener?.("pointerenter", () => {
    element[PRESENTATION_POINTER_REVEAL] = true;
    reconcile();
  });
  element.addEventListener?.("pointerleave", () => {
    element[PRESENTATION_POINTER_REVEAL] = false;
    reconcile();
  });
  element.addEventListener?.("focus", () => {
    element[PRESENTATION_KEYBOARD_REVEAL] = element.matches?.(":focus-visible") === true;
    reconcile();
  });
  element.addEventListener?.("blur", () => {
    element[PRESENTATION_KEYBOARD_REVEAL] = false;
    reconcile();
  });
  element[PRESENTATION_BINDINGS].add(fieldId);
}

function ensurePresentationObserver() {
  if (
    presentationObserver
    || typeof MutationObserver !== "function"
    || typeof document === "undefined"
  ) return;
  const root = document.documentElement || document.body;
  if (!root) return;
  presentationObserver = new MutationObserver(() => {
    if (presentationObserverScheduled) return;
    presentationObserverScheduled = true;
    const reconcile = () => {
      presentationObserverScheduled = false;
      for (const reference of TRACKED_PRESENTATION_REFERENCES) {
        const node = reference.deref();
        if (!node) {
          TRACKED_PRESENTATION_REFERENCES.delete(reference);
          continue;
        }
        applyPrivacyPresentation(
          node,
          TRACKED_PRESENTATION_UNAVAILABLE.get(node) === true,
        );
      }
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(reconcile);
    else Promise.resolve().then(reconcile);
  });
  presentationObserver.observe(root, { childList: true, subtree: true });
}

function trackPresentationNode(node, unavailable) {
  if (!TRACKED_PRESENTATION_REFERENCE_BY_NODE.has(node)) {
    const reference = typeof WeakRef === "function"
      ? new WeakRef(node)
      : { deref: () => node };
    TRACKED_PRESENTATION_REFERENCE_BY_NODE.set(node, reference);
    TRACKED_PRESENTATION_REFERENCES.add(reference);
  }
  TRACKED_PRESENTATION_UNAVAILABLE.set(node, unavailable);
  ensurePresentationObserver();
}

function scheduleReadyPresentationReconcile(node, unavailable, epoch) {
  const reconcile = () => {
    if (node[PRESENTATION_RECONCILE_EPOCH] !== epoch) return;
    applyPrivacyPresentation(node, unavailable);
  };
  if (typeof queueMicrotask === "function") queueMicrotask(reconcile);
  else Promise.resolve().then(reconcile);
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(reconcile);
}

function applyPrivacyPresentation(node, unavailable) {
  const masked = privatePresentation(node);
  const fieldUnavailable = unavailable && masked;
  let domReady = true;
  for (const fieldId of nodeFieldIds(node)) {
    const target = widget(node, { fieldId });
    patchPromptDraw(node, target);
    const elements = widgetTextElements(node, target);
    if (!hasMountedTextElement(elements)) domReady = false;
    for (const element of elements) {
      element.classList?.add(PROMPT_FIELD_CLASS);
      element.classList?.toggle(PRIVATE_FIELD_CLASS, masked);
      element.classList?.toggle(PRIVACY_UNAVAILABLE_CLASS, fieldUnavailable);
      element.setAttribute?.("data-aio-private", masked ? "true" : "false");
      element.setAttribute?.(
        "data-aio-privacy-unavailable",
        fieldUnavailable ? "true" : "false",
      );
      bindPresentationEvents(node, fieldId, target, element, unavailable);
      reconcileElementPresentation(node, fieldId, target, element, unavailable);
    }
  }
  node.__aioManagedPrivacyUnavailable = unavailable;
  node.__aioManagedPrivacyMasked = masked;
  node.setDirtyCanvas?.(true, true);
  return domReady;
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
    .${PRIVATE_FIELD_CLASS}:hover {
      background: var(--helto-surface-2) !important;
      border-color: var(--helto-border-strong) !important;
      color: var(--helto-text, #cdd6f4) !important;
      -webkit-text-fill-color: currentColor !important;
      caret-color: auto !important;
    }
    .${PRIVACY_UNAVAILABLE_CLASS} {
      color: transparent !important;
      -webkit-text-fill-color: transparent !important;
      caret-color: transparent !important;
      text-shadow: none !important;
    }
    .${PRIVACY_UNAVAILABLE_CLASS}:hover {
      color: var(--helto-text, #cdd6f4) !important;
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
      if (
        !node.__aioManagedPrivacyMasked
        || hasMountedTextElement(widgetTextElements(node, target))
      ) {
        return original.apply(this, arguments);
      }
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

function schedulePrivacyPresentation(node, unavailable) {
  const epoch = (Number(node[PRESENTATION_RECONCILE_EPOCH]) || 0) + 1;
  node[PRESENTATION_RECONCILE_EPOCH] = epoch;
  trackPresentationNode(node, unavailable);
  let remainingFrames = PRESENTATION_RECONCILE_FRAMES;
  const reconcile = () => {
    if (node[PRESENTATION_RECONCILE_EPOCH] !== epoch) return;
    if (applyPrivacyPresentation(node, unavailable)) {
      scheduleReadyPresentationReconcile(node, unavailable, epoch);
      return;
    }
    if (remainingFrames <= 0 || typeof requestAnimationFrame !== "function") return;
    remainingFrames -= 1;
    requestAnimationFrame(reconcile);
  };
  reconcile();
}

function bindBootstrapModeChanges(node) {
  if (aioManagedNodeType(node) !== GENERATE_NODE) return;
  const target = node.widgets?.find((item) => item?.name === "privacy_mode");
  if (!target || target[BOOTSTRAP_MODE_BOUND]) return;
  const original = target.callback;
  target.callback = function aioManagedBootstrapModeChanged() {
    const result = original?.apply(this, arguments);
    reconcileAioPromptPrivacyUnavailable(node);
    for (const candidate of connectedKreaNodes(node)) {
      reconcileAioPromptPrivacyUnavailable(candidate);
    }
    return result;
  };
  target[BOOTSTRAP_MODE_BOUND] = true;
}

function updatePrivacyPresentation(node) {
  schedulePrivacyPresentation(node, false);
}

export function reconcileAioPromptPrivacyUnavailable(node) {
  if (!Object.values(FIELD_FACTS).some((field) => field.nodeType === aioManagedNodeType(node))) {
    return false;
  }
  installPrivacyStyles();
  hideImplementationWidgets(node);
  bindBootstrapModeChanges(node);
  schedulePrivacyPresentation(node, true);
  return true;
}

export function installAioPromptPrivacyBootstrap(app) {
  if (!app || typeof app.registerExtension !== "function") fail();
  if (BOOTSTRAPPED_APPS.has(app)) return;
  BOOTSTRAPPED_APPS.add(app);
  app.registerExtension({
    name: "helto.aio-image-generation.privacy-bootstrap",
    nodeCreated: reconcileAioPromptPrivacyUnavailable,
    loadedGraphNode: reconcileAioPromptPrivacyUnavailable,
  });
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
    const protectedValue = protectedValues(node)[fieldId];
    if (typeof protectedValue === "string" && plaintext === protectedValue) {
      updatePrivacyPresentation(node);
      return undefined;
    }
    plaintextValues(node)[fieldId] = plaintext;
    if (typeof protectedValue === "string") {
      transition.withInternalMutation(() => {
        target.value = protectedValue;
      });
    }
    updatePrivacyPresentation(node);
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
        if (transition.isInternalMutation()) return result;
        recordEdit(node, fieldId, value);
        return result;
      };
      target.__aioManagedPrivacyEditBindings.add(fieldId);
    }
    const widgetOwnedElements = new Set([
      target.inputEl,
      target.element,
      target.inputElement,
      target.textarea,
    ].filter(Boolean));
    for (const element of widgetTextElements(node, target)) {
      if (widgetOwnedElements.has(element)) continue;
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

  function scheduleEditBindings(node) {
    const epoch = (Number(node[EDIT_BINDING_RECONCILE_EPOCH]) || 0) + 1;
    node[EDIT_BINDING_RECONCILE_EPOCH] = epoch;
    let remainingFrames = PRESENTATION_RECONCILE_FRAMES;
    const reconcile = () => {
      if (node[EDIT_BINDING_RECONCILE_EPOCH] !== epoch) return;
      const fieldIds = nodeFieldIds(node);
      for (const fieldId of fieldIds) bindEdits(node, fieldId);
      const ready = fieldIds.every((fieldId) => {
        const target = widget(node, { fieldId });
        return hasMountedTextElement(widgetTextElements(node, target));
      });
      if (
        ready
        || remainingFrames <= 0
        || typeof requestAnimationFrame !== "function"
      ) return;
      remainingFrames -= 1;
      requestAnimationFrame(reconcile);
    };
    reconcile();
  }

  function applyValue(node, value, context) {
    const field = facts(context);
    const target = widget(node, context);
    const plaintext = normalize(value);
    plaintextValues(node)[field.fieldId] = plaintext;
    if (!(field.fieldId in protectedValues(node))) {
      transition.withInternalMutation(() => {
        target.value = plaintext;
      });
    }
    updatePrivacyPresentation(node);
  }

  function clearValue(node, context) {
    const field = facts(context);
    plaintextValues(node)[field.fieldId] = "";
    const target = widget(node, context);
    if (!(field.fieldId in protectedValues(node))) {
      transition.withInternalMutation(() => {
        target.value = "";
      });
    }
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
    scheduleEditBindings(node);
    updatePrivacyPresentation(node);
    if (!locked) return;
    for (const [fieldId, field] of Object.entries(FIELD_FACTS)) {
      if (field.nodeType !== aioManagedNodeType(node)) continue;
      plaintextValues(node)[fieldId] = "";
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
        if (locked) {
          for (const fieldId of nodeFieldIds(owner)) plaintextValues(owner)[fieldId] = "";
        }
        updatePrivacyPresentation(owner);
      }
    },
    ...transition,
  };
}
