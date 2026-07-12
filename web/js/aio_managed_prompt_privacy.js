// Browser-owned field locations for the inactive managed Generate/Krea slice.

import {
  aioManagedGraphLink,
  aioManagedGraphNodes,
  aioManagedNodeType,
} from "./aio_managed_privacy_graph.js";

export const AIO_GENERATE_POSITIVE_FIELD_ID = "generate-positive-prompt";
export const AIO_GENERATE_NEGATIVE_FIELD_ID = "generate-negative-prompt";
export const AIO_KREA_INPAINT_FIELD_ID = "krea-inpaint-positive-prompt";

const GENERATE_NODE = "AIOImageGenerate";
const KREA_NODE = "AIOKrea2Settings";
const PROTECTED_VALUES = "__aioManagedPromptProtectedValues";
const PLAINTEXT_VALUES = "__aioManagedPromptPlaintextValues";
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
  const fieldId = context?.fieldId ?? context?.id;
  const value = FIELD_FACTS[fieldId];
  if (!value) fail();
  if (context?.location?.name !== undefined && context.location.name !== value.widget) fail();
  return { ...value, fieldId };
}

function widget(node, context) {
  const field = facts(context);
  if (aioManagedNodeType(node) !== field.nodeType) fail();
  const found = node?.widgets?.find((item) => item?.name === field.widget);
  if (!found) fail();
  return found;
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

export function createAioPromptWorkflowBrowserAdapter({ workflowHandle = null } = {}) {
  let locked = false;

  function recordEdit(node, fieldId, candidate = undefined) {
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
        const result = original?.apply(this, arguments);
        recordEdit(node, fieldId, value);
        return result;
      };
      target.__aioManagedPrivacyEditBindings.add(fieldId);
    }
    for (const element of widgetTextElements(target)) {
      element.__aioManagedPrivacyEditBindings ||= new Set();
      if (element.__aioManagedPrivacyEditBindings.has(fieldId)) continue;
      const onEdited = () => recordEdit(
        node,
        fieldId,
        "value" in element ? element.value : element.textContent,
      );
      element.addEventListener?.("input", onEdited);
      element.addEventListener?.("change", onEdited);
      element.__aioManagedPrivacyEditBindings.add(fieldId);
    }
  }

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
      target.value = protectedValue;
      setDomText(target, locked ? "" : plaintext);
    },
    apply(node, value, context) {
      const field = facts(context);
      const target = widget(node, context);
      const plaintext = normalize(value);
      plaintextValues(node)[field.fieldId] = plaintext;
      if (!(field.fieldId in protectedValues(node))) target.value = plaintext;
      setDomText(target, plaintext);
    },
    clear(node, context) {
      const field = facts(context);
      plaintextValues(node)[field.fieldId] = "";
      const target = widget(node, context);
      if (!(field.fieldId in protectedValues(node))) target.value = "";
      setDomText(target, "");
    },
    reconcileNode(node) {
      if (!Object.values(FIELD_FACTS).some((field) => field.nodeType === aioManagedNodeType(node))) fail();
      for (const [fieldId, field] of Object.entries(FIELD_FACTS)) {
        if (field.nodeType === aioManagedNodeType(node)) bindEdits(node, fieldId);
      }
      if (!locked) return;
      for (const [fieldId, field] of Object.entries(FIELD_FACTS)) {
        if (field.nodeType !== aioManagedNodeType(node)) continue;
        plaintextValues(node)[fieldId] = "";
        setDomText(widget(node, { fieldId }), "");
      }
    },
    reconcileNodeDefinition() {},
    onPrivacySessionChange(snapshot) {
      locked = snapshot?.state !== "ready" && snapshot?.state !== "unlocked";
    },
  };
}
