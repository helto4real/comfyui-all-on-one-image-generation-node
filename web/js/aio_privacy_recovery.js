import {
  PRIVACY_SCHEMA,
  acceptsAioRecoveryEnvelope,
  encryptState,
  encryptValue,
  getSharedPrivacyUi,
  showPrivacyRecoveryDialog,
} from "./aio_privacy.js";

export const AIO_GENERATE_NODE_NAME = "AIOImageGenerate";
export const AIO_GENERATE_NODE_DISPLAY_NAME = "AIO Image Generate";
export const AIO_KREA_SETTINGS_NODE_NAME = "AIOKrea2Settings";
export const AIO_KREA_SETTINGS_NODE_DISPLAY_NAME = "Krea 2 Settings";
export const AIO_PROMPT_BUILDER_NODE_NAME = "AIOIdeogram4PromptBuilder";
export const AIO_PROMPT_BUILDER_NODE_DISPLAY_NAME = "Ideogram 4 Prompt Builder";
export const DEFAULT_GENERATE_PROMPT = "A luminous studio portrait, crisp details, natural color, soft light";
export const PROMPT_BUILDER_STATE_PROPERTY = "aio_ideogram4_prompt_builder_state";
export const PROMPT_BUILDER_WORKFLOW_STATE_KEY = "aio_ideogram4_prompt_builder";
export const PROMPT_BUILDER_LEGACY_WORKFLOW_STATE_KEY = "ideo";
export const AIO_PRIVACY_MENU_LABEL = "Privacy Recovery...";

export const PROMPT_BUILDER_SENSITIVE_WIDGET_NAMES = [
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
];

let descriptorsRegistered = false;
let lastRegistration = null;

function widgetByName(node, name) {
  return node?.widgets?.find?.((widget) => widget?.name === name) || null;
}

function nodeTypeCandidates(node) {
  return [
    node?.type,
    node?.comfyClass,
    node?.class_type,
    node?.constructor?.type,
    node?.constructor?.comfyClass,
    node?.title,
  ].map((value) => String(value || "")).filter(Boolean);
}

function nodeMatches(node, names) {
  const candidates = nodeTypeCandidates(node);
  return names.some((name) => candidates.includes(name));
}

async function reencryptValue(plaintext) {
  return encryptValue(plaintext ?? "");
}

async function reencryptState(plaintext) {
  let state = plaintext;
  if (typeof plaintext === "string") {
    try {
      state = JSON.parse(plaintext);
    } catch {
      state = { value: plaintext };
    }
  }
  return encryptState(state && typeof state === "object" ? state : { value: state ?? "" });
}

function clearWidgetMemo(node, field) {
  const widget = widgetByName(node, field?.name);
  if (widget) delete widget.__aioPrivacyEnvelopeMemo;
}

function clearGenerateRuntimeState(node, context) {
  clearWidgetMemo(node, context?.field);
  if (node) delete node._aioPrivacyStatus;
}

function clearPromptBuilderRuntimeState(node, context) {
  clearWidgetMemo(node, context?.field);
  if (!node) return;
  delete node._aioIdeogram4LastPrivatePayload;
  if (node._aioIdeogram4PendingWorkflowInfo) {
    delete node._aioIdeogram4PendingWorkflowInfo[PROMPT_BUILDER_WORKFLOW_STATE_KEY];
    delete node._aioIdeogram4PendingWorkflowInfo[PROMPT_BUILDER_LEGACY_WORKFLOW_STATE_KEY];
  }
  if (context?.field?.kind === "property" && node.properties) {
    delete node.properties[PROMPT_BUILDER_STATE_PROPERTY];
  }
  node._aioIdeogram4RecoveryReset?.();
}

function privacyFieldDefault() {
  return false;
}

export function aioPrivacyRecoveryDescriptors() {
  return [
    {
      id: "aio-image-generate:main-prompts",
      nodeTypes: [AIO_GENERATE_NODE_NAME],
      label: AIO_GENERATE_NODE_DISPLAY_NAME,
      schema: PRIVACY_SCHEMA,
      privacy: { widget: "privacy_mode", default: privacyFieldDefault() },
      acceptsEnvelope: acceptsAioRecoveryEnvelope,
      reencrypt: reencryptValue,
      clearRuntimeState: clearGenerateRuntimeState,
      fields: [
        {
          kind: "widget",
          name: "positive_prompt",
          label: "Positive prompt",
          defaultValue: DEFAULT_GENERATE_PROMPT,
          sensitive: true,
          resetOnlyForLegacy: true,
        },
        {
          kind: "widget",
          name: "negative_prompt",
          label: "Negative prompt",
          defaultValue: "",
          sensitive: true,
          resetOnlyForLegacy: true,
        },
      ],
    },
    {
      id: "aio-image-generate:krea-inpaint-prompt",
      nodeTypes: [AIO_KREA_SETTINGS_NODE_NAME],
      label: AIO_KREA_SETTINGS_NODE_DISPLAY_NAME,
      schema: PRIVACY_SCHEMA,
      acceptsEnvelope: acceptsAioRecoveryEnvelope,
      reencrypt: reencryptValue,
      clearRuntimeState: clearGenerateRuntimeState,
      fields: [
        {
          kind: "widget",
          name: "inpaint_positive_prompt",
          label: "Inpaint positive prompt",
          defaultValue: "",
          sensitive: true,
          resetOnlyForLegacy: true,
        },
      ],
    },
    {
      id: "aio-image-generate:prompt-builder",
      nodeTypes: [AIO_PROMPT_BUILDER_NODE_NAME],
      label: AIO_PROMPT_BUILDER_NODE_DISPLAY_NAME,
      schema: PRIVACY_SCHEMA,
      privacy: { widget: "privacy_mode", default: privacyFieldDefault() },
      acceptsEnvelope: acceptsAioRecoveryEnvelope,
      reencrypt: reencryptValue,
      clearRuntimeState: clearPromptBuilderRuntimeState,
      fields: [
        ...PROMPT_BUILDER_SENSITIVE_WIDGET_NAMES.map((name) => ({
          kind: "widget",
          name,
          label: name.replaceAll("_", " "),
          defaultValue: "",
          sensitive: true,
          resetOnlyForLegacy: true,
        })),
        {
          kind: "property",
          name: PROMPT_BUILDER_STATE_PROPERTY,
          label: "Prompt builder state",
          defaultValue: "",
          sensitive: true,
          resetOnlyForLegacy: true,
          reencrypt: reencryptState,
          clearRuntimeState: clearPromptBuilderRuntimeState,
        },
      ],
    },
  ];
}

export async function registerAioPrivacyRecoveryDescriptors() {
  if (descriptorsRegistered) return lastRegistration;
  const privacy = await getSharedPrivacyUi();
  if (!privacy?.registerPrivacyRecoveryDescriptors) return null;
  lastRegistration = privacy.registerPrivacyRecoveryDescriptors(
    "aio-image-generate",
    aioPrivacyRecoveryDescriptors(),
  );
  descriptorsRegistered = true;
  return lastRegistration;
}

export function appendPrivacyRecoveryMenuOption(node, options) {
  if (!Array.isArray(options)) return;
  if (!nodeMatches(node, [AIO_GENERATE_NODE_NAME, AIO_GENERATE_NODE_DISPLAY_NAME, AIO_KREA_SETTINGS_NODE_NAME, AIO_KREA_SETTINGS_NODE_DISPLAY_NAME, AIO_PROMPT_BUILDER_NODE_NAME, AIO_PROMPT_BUILDER_NODE_DISPLAY_NAME])) return;
  if (options.some((item) => item?.content === AIO_PRIVACY_MENU_LABEL)) return;
  options.push({
    content: AIO_PRIVACY_MENU_LABEL,
    callback: () => showPrivacyRecoveryDialog({ mode: "manual" }),
  });
}
