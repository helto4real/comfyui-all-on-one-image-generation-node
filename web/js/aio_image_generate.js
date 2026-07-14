import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import "./aio_managed_privacy.js";
import {
  applyHeltoLiteGraphWidgetTheme,
  applyHeltoNodeTheme,
  ensureHeltoTokens,
  HELTO,
  restoreHeltoLiteGraphWidgetTheme,
  withHeltoLiteGraphWidgetTheme,
} from "./aio_helto_theme.js";
import {
  escapeHtml,
  escapedValueMarkup,
  imageInfoFieldMarkup,
  safeUrl,
} from "./aio_lora_info_markup.js";
import {
  normalizeFluxSettingsWidgetValues,
  normalizeZImageSettingsWidgetValues,
} from "./aio_settings_migration.js";

const NODE_NAME = "AIOLoraConfiguration";
const NODE_DISPLAY_NAME = "AIO LoRA Configuration";
const GENERATE_NODE_NAME = "AIOImageGenerate";
const GENERATE_NODE_DISPLAY_NAME = "AIO Image Generate";
const FLUX_SETTINGS_NODE_NAME = "AIOFlux2Klein9BSettings";
const Z_IMAGE_SETTINGS_NODE_NAME = "AIOZImageTurboSettings";
const AIO_THEME_NODE_TYPES = new Set([
  "AIOImageGenerate",
  "AIOZImageTurboSettings",
  "AIOFlux2Klein9BSettings",
  "AIOIdeogram4PromptBuilder",
  "AIOIdeogram4Settings",
  "AIOKrea2Settings",
  "AIOInpaint",
  "AIOLoraConfiguration",
  "AIOLoadPipelineModels",
  "AIOModelInfo",
  "AIOPIDInfo",
  "AIOInpaintInfo",
]);
const ROW_PREFIX = "lora_";
const HEADER_NAME = "aio_lora_header";
const ADD_BUTTON_LABEL = "+ Add LoRA";
const FIXED_SEED_BUTTON_LABEL = "Generate fixed seed";
const MIN_NODE_WIDTH = 560;
const MAX_SIDE_MIN = 256;
const MAX_SIDE_MAX = 4096;
const SEED_MAX = Number.MAX_SAFE_INTEGER;
const SEED_CONTROL_MODES = ["fixed", "increment", "decrement", "randomize"];
const AIO_SEED_QUEUE_WRAPPER_KEY = "__aioGenerateSeedQueuePromptWrapper";
const AIO_SEED_QUEUE_INSTALL_KEY = "__aioGenerateSeedQueuePromptInstallScheduled";
const AIO_SEED_QUEUE_INSTALL_ATTEMPT_LIMIT = 80;
const AIO_PROGRESS_TEXT_CLEANUP_KEY = "__aioGenerateProgressTextCleanupInstalled";
const AIO_RUNTIME_PHASE_BRIDGE_KEY = "__aioGenerateRuntimePhaseBridgeInstalled";
const AIO_WIDGET_THEME_BRIDGE_KEY = "__aioHeltoLiteGraphWidgetThemeBridgeInstalled";
const AIO_WIDGET_THEME_FALLBACK_KEY = "__aioHeltoLiteGraphWidgetThemeFallbackInstalled";
const AIO_WIDGET_THEME_SNAPSHOT_KEY = "__aioHeltoLiteGraphWidgetThemeSnapshot";
const AIO_RUNTIME_PHASE_STYLE_ID = "aio-generate-runtime-phase-style";
const AIO_RUNTIME_PHASE_LABEL_CLASS = "aio-generate-runtime-phase-label";
const AIO_RUNTIME_PHASE_NODE_KEY = "__aioGenerateRuntimePhase";
const PROGRESS_TEXT_WIDGET_NAME = "$$node-text-preview";
const LORA_HEADER_TOOLTIP = "Toggle every configured LoRA row on or off.";
const LORA_ROW_TOOLTIP = "LoRA row: choose a LoRA, toggle it, inspect metadata, and adjust strength.";
const ADD_LORA_TOOLTIP = "Add a LoRA row filtered by the match field.";
const FIXED_SEED_BUTTON_TOOLTIP = "Generate a new fixed random seed and write it into the seed field.";
const USE_ZERO_NEGATIVE_CONDITIONING_WIDGET_NAME = "use_zero_negative_conditioning";
const BATCH_COUNT_WIDGET_NAME = "batch_count";

const DEFAULT_ROW = {
  on: true,
  lora: null,
  strength: 1,
  strengthTwo: null,
};

const NUMBER_WIDTH_TOTAL = 9 + 3 + 32 + 3 + 9;

let loraListPromise = null;

function widgetValue(node, name, fallback = "") {
  const widget = node.widgets?.find((item) => item.name === name);
  return widget?.value ?? fallback;
}

function widgetByName(node, name) {
  return node.widgets?.find((item) => item.name === name);
}

function isAioLoraNodeData(nodeData) {
  return nodeData?.name === NODE_NAME || nodeData?.display_name === NODE_DISPLAY_NAME;
}

function isAioGenerateNodeData(nodeData) {
  return nodeData?.name === GENERATE_NODE_NAME || nodeData?.display_name === GENERATE_NODE_DISPLAY_NAME;
}

function aioNodeTypeCandidates(node) {
  return [
    node?.type,
    node?.comfyClass,
    node?.class_type,
    node?.constructor?.type,
    node?.constructor?.comfyClass,
    node?.title,
  ].map((value) => String(value || "")).filter(Boolean);
}

function isAioThemedNodeData(nodeData) {
  return AIO_THEME_NODE_TYPES.has(String(nodeData?.name || ""));
}

function isAioThemedNode(node) {
  return aioNodeTypeCandidates(node).some((candidate) => AIO_THEME_NODE_TYPES.has(candidate));
}

function liteGraphCanvasPrototype() {
  return [
    globalThis.LGraphCanvas?.prototype,
    globalThis.LiteGraph?.LGraphCanvas?.prototype,
    app.canvas?.constructor?.prototype,
  ].find((prototype) => typeof prototype?.drawNodeWidgets === "function") || null;
}

function installAioWidgetThemeBridge() {
  const prototype = liteGraphCanvasPrototype();
  if (!prototype) {
    return false;
  }
  if (prototype[AIO_WIDGET_THEME_BRIDGE_KEY]) {
    return true;
  }
  const originalDrawNodeWidgets = prototype.drawNodeWidgets;
  prototype[AIO_WIDGET_THEME_BRIDGE_KEY] = true;
  prototype.drawNodeWidgets = function (node) {
    if (isAioThemedNode(node)) {
      return withHeltoLiteGraphWidgetTheme(() => originalDrawNodeWidgets.apply(this, arguments));
    }
    return originalDrawNodeWidgets.apply(this, arguments);
  };
  return true;
}

function ensureAioWidgetThemeFallback(node) {
  if (!node || node[AIO_WIDGET_THEME_FALLBACK_KEY]) {
    return;
  }
  node[AIO_WIDGET_THEME_FALLBACK_KEY] = true;

  const originalDrawBackground = node.onDrawBackground;
  node.onDrawBackground = function () {
    restoreHeltoLiteGraphWidgetTheme(this[AIO_WIDGET_THEME_SNAPSHOT_KEY]);
    this[AIO_WIDGET_THEME_SNAPSHOT_KEY] = applyHeltoLiteGraphWidgetTheme();
    try {
      return originalDrawBackground?.apply(this, arguments);
    } catch (error) {
      restoreHeltoLiteGraphWidgetTheme(this[AIO_WIDGET_THEME_SNAPSHOT_KEY]);
      this[AIO_WIDGET_THEME_SNAPSHOT_KEY] = null;
      throw error;
    }
  };

  const originalDrawForeground = node.onDrawForeground;
  node.onDrawForeground = function () {
    try {
      return originalDrawForeground?.apply(this, arguments);
    } finally {
      restoreHeltoLiteGraphWidgetTheme(this[AIO_WIDGET_THEME_SNAPSHOT_KEY]);
      this[AIO_WIDGET_THEME_SNAPSHOT_KEY] = null;
    }
  };
}

function applyAioNodeTheme(node) {
  if (!isAioThemedNode(node)) {
    return false;
  }
  if (!installAioWidgetThemeBridge()) {
    ensureAioWidgetThemeFallback(node);
  }
  return applyHeltoNodeTheme(node);
}

function isAioGenerateNode(node) {
  return (
    node?.type === GENERATE_NODE_NAME ||
    node?.comfyClass === GENERATE_NODE_NAME ||
    node?.constructor?.type === GENERATE_NODE_NAME ||
    node?.constructor?.comfyClass === GENERATE_NODE_NAME ||
    node?.title === GENERATE_NODE_DISPLAY_NAME
  );
}

function isAioLoraNode(node) {
  return (
    node?.type === NODE_NAME ||
    node?.comfyClass === NODE_NAME ||
    node?.constructor?.type === NODE_NAME ||
    node?.constructor?.comfyClass === NODE_NAME ||
    node?.title === NODE_DISPLAY_NAME
  );
}

function multipleValueStep(value) {
  const numeric = Number(value);
  return [8, 16, 32].includes(numeric) ? numeric : 1;
}

function clampMaxSide(value) {
  return Math.min(MAX_SIDE_MAX, Math.max(MAX_SIDE_MIN, value));
}

function snapMaxSide(value, step) {
  const numeric = Number(value);
  const rounded = Number.isFinite(numeric) ? Math.round(numeric) : 1024;
  if (step <= 1) {
    return clampMaxSide(rounded);
  }
  return clampMaxSide(Math.round(rounded / step) * step);
}

function markNodeDirty(node) {
  if (typeof node?.setDirtyCanvas === "function") {
    node.setDirtyCanvas(true, true);
  } else {
    app.graph?.setDirtyCanvas?.(true, true);
  }
  node?.graph?.setDirtyCanvas?.(true, true);
  app.canvas?.setDirty?.(true, true);
}

function installAioGenerateRuntimePhaseStyles() {
  ensureHeltoTokens();
  if (document.getElementById(AIO_RUNTIME_PHASE_STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = AIO_RUNTIME_PHASE_STYLE_ID;
  style.textContent = `
    .${AIO_RUNTIME_PHASE_LABEL_CLASS} {
      position: absolute;
      top: 34px;
      right: 10px;
      z-index: 8;
      max-width: calc(100% - 20px);
      min-height: 18px;
      padding: 2px 8px;
      border: 1px solid var(--helto-border-strong);
      border-radius: 9px;
      background: var(--helto-bg);
      color: var(--helto-text);
      font: 600 11px/14px var(--helto-font-sans, system-ui, sans-serif);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      pointer-events: none;
      box-shadow: var(--helto-shadow);
    }
    .lg-node[data-collapsed] > .${AIO_RUNTIME_PHASE_LABEL_CLASS} {
      top: 8px;
    }
  `;
  document.head.appendChild(style);
}

function randomUnit53() {
  if (globalThis.crypto?.getRandomValues) {
    const values = new Uint32Array(2);
    globalThis.crypto.getRandomValues(values);
    return (((values[0] & 0x1fffff) * 0x100000000) + values[1]) / 0x20000000000000;
  }
  return Math.random();
}

function randomFixedSeed() {
  return Math.floor(randomUnit53() * SEED_MAX) + 1;
}

function validSeedControlMode(value) {
  return SEED_CONTROL_MODES.includes(value) ? value : null;
}

function isSeedControlWidget(widget, seedWidget = null) {
  const values = widget?.options?.values;
  const seedName = String(seedWidget?.name || "seed");
  return (
    widget?.name === "control_after_generate" ||
    widget?.name === `${seedName}.control_after_generate` ||
    widget?.name === `${seedName}_control_after_generate` ||
    (Array.isArray(values) && SEED_CONTROL_MODES.every((value) => values.includes(value)))
  );
}

function seedControlWidget(node, seedWidget = widgetByName(node, "seed")) {
  for (const widget of seedWidget?.linkedWidgets || []) {
    if (isSeedControlWidget(widget, seedWidget)) {
      return widget;
    }
  }
  return node?.widgets?.find((widget) => widget !== seedWidget && isSeedControlWidget(widget, seedWidget)) || null;
}

function liveSeedControlMode(node) {
  const seedWidget = widgetByName(node, "seed");
  const controlWidget = seedControlWidget(node, seedWidget);
  return (
    validSeedControlMode(controlWidget?.value) ??
    validSeedControlMode(seedWidget?.control_after_generate) ??
    validSeedControlMode(seedWidget?.options?.control_after_generate)
  );
}

function writeWidgetValue(node, widget, value) {
  if (!node || !widget) {
    return false;
  }
  const previousValue = widget.value;
  widget.value = value;
  widget.callback?.(value, app.canvas, node, widget);
  node.onWidgetChanged?.(widget.name ?? "", value, previousValue, widget);
  node.graph?.incrementVersion?.();
  markNodeDirty(node);
  return true;
}

function writeSerializedWidgetValue(node, widget, value) {
  const index = serializedWidgetIndex(node, widget);
  for (const values of [node?.widgets_values, node?.last_serialization?.widgets_values]) {
    if (Array.isArray(values) && index >= 0 && index < values.length) {
      values[index] = value;
    }
  }
}

function writeAioSeedValue(node, seed) {
  const seedWidget = widgetByName(node, "seed");
  if (!writeWidgetValue(node, seedWidget, seed)) {
    return false;
  }
  writeSerializedWidgetValue(node, seedWidget, seed);
  return true;
}

function writeAioSeedControlMode(node, mode) {
  const seedWidget = widgetByName(node, "seed");
  const controlWidget = seedControlWidget(node, seedWidget);
  if (!writeWidgetValue(node, controlWidget, mode)) {
    return false;
  }
  writeSerializedWidgetValue(node, controlWidget, mode);
  return true;
}

function ensureAioGenerateSeedButton(node) {
  if (!isAioGenerateNode(node)) {
    return;
  }
  const seedWidget = widgetByName(node, "seed");
  const stepsWidget = widgetByName(node, "steps");
  if (!seedWidget || !stepsWidget) {
    return;
  }

  let button = node.widgets?.find((widget) => widget._aioGenerateSeedButton === true);
  if (!button) {
    button = node.addWidget("button", FIXED_SEED_BUTTON_LABEL, null, () => {
      const seed = randomFixedSeed();
      writeAioSeedValue(node, seed);
      writeAioSeedControlMode(node, "fixed");
      markNodeDirty(node);
    });
    button._aioGenerateSeedButton = true;
  }

  button.serialize = false;
  button.options ||= {};
  button.options.serialize = false;
  button.tooltip = FIXED_SEED_BUTTON_TOOLTIP;
  button.options.tooltip = FIXED_SEED_BUTTON_TOOLTIP;

  const widgets = node.widgets || [];
  const buttonIndex = widgets.indexOf(button);
  if (buttonIndex >= 0) {
    widgets.splice(buttonIndex, 1);
  }
  const stepsIndex = widgets.indexOf(stepsWidget);
  widgets.splice(stepsIndex >= 0 ? stepsIndex : widgets.length, 0, button);
}

function defaultGraph() {
  return app.rootGraph || app.graph;
}

function graphNodes(graph = defaultGraph()) {
  const nodes = [];
  const seenNodes = new Set();
  const seenGraphs = new Set();

  function visit(currentGraph) {
    if (!currentGraph || seenGraphs.has(currentGraph)) {
      return;
    }
    seenGraphs.add(currentGraph);
    for (const node of currentGraph.nodes || currentGraph._nodes || []) {
      if (!node || seenNodes.has(node)) {
        continue;
      }
      seenNodes.add(node);
      nodes.push(node);
      if (node.subgraph) {
        visit(node.subgraph);
      }
    }
    for (const subgraph of currentGraph.subgraphs?.values?.() || []) {
      visit(subgraph);
    }
  }

  visit(graph);
  return nodes;
}

function removeAioGenerateProgressTextWidget(node) {
  if (!isAioGenerateNode(node) || !Array.isArray(node.widgets)) {
    return false;
  }
  let removed = false;
  for (let index = node.widgets.length - 1; index >= 0; index -= 1) {
    if (node.widgets[index]?.name !== PROGRESS_TEXT_WIDGET_NAME) {
      continue;
    }
    const [widget] = node.widgets.splice(index, 1);
    widget?.onRemove?.();
    removed = true;
  }
  if (removed) {
    markNodeDirty(node);
  }
  return removed;
}

function aioGenerateRuntimePhaseText(text) {
  const phase = String(text ?? "").trim();
  if (!phase) {
    return "";
  }
  return phase.length > 96 ? `${phase.slice(0, 93)}...` : phase;
}

function aioGenerateNodeIdCandidates(nodeId) {
  const value = String(nodeId ?? "").trim();
  if (!value) {
    return new Set();
  }
  const candidates = new Set([value]);
  const parts = value.split(":").filter(Boolean);
  if (parts.length > 1) {
    candidates.add(parts[parts.length - 1]);
  }
  return candidates;
}

function findAioGenerateNodeById(nodeId) {
  const candidates = aioGenerateNodeIdCandidates(nodeId);
  if (!candidates.size) {
    return null;
  }
  for (const node of graphNodes()) {
    if (isAioGenerateNode(node) && candidates.has(String(node.id))) {
      return node;
    }
  }
  return null;
}

function aioGenerateVueNodeElement(node) {
  const nodeId = String(node?.id ?? "");
  if (!nodeId) {
    return null;
  }
  for (const element of document.querySelectorAll(".lg-node[data-node-id]")) {
    if (element.dataset?.nodeId === nodeId) {
      return element;
    }
  }
  return null;
}

function aioGenerateRuntimePhaseLabel(nodeElement) {
  if (!nodeElement) {
    return null;
  }
  for (const child of nodeElement.children || []) {
    if (child.classList?.contains(AIO_RUNTIME_PHASE_LABEL_CLASS)) {
      return child;
    }
  }
  return null;
}

function updateAioGenerateRuntimePhaseDom(node) {
  const nodeElement = aioGenerateVueNodeElement(node);
  if (!nodeElement) {
    return;
  }
  const phase = node?.[AIO_RUNTIME_PHASE_NODE_KEY] || "";
  let label = aioGenerateRuntimePhaseLabel(nodeElement);
  if (!phase) {
    label?.remove();
    return;
  }
  installAioGenerateRuntimePhaseStyles();
  if (!label) {
    label = document.createElement("div");
    label.className = AIO_RUNTIME_PHASE_LABEL_CLASS;
    nodeElement.appendChild(label);
  }
  label.textContent = phase;
  label.title = phase;
}

function scheduleAioGenerateRuntimePhaseDomUpdate(node) {
  updateAioGenerateRuntimePhaseDom(node);
  const update = () => updateAioGenerateRuntimePhaseDom(node);
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(update);
  } else {
    setTimeout(update, 0);
  }
}

function setAioGenerateRuntimePhase(node, text) {
  if (!isAioGenerateNode(node)) {
    return false;
  }
  const phase = aioGenerateRuntimePhaseText(text);
  if (!phase) {
    return false;
  }
  node[AIO_RUNTIME_PHASE_NODE_KEY] = phase;
  scheduleAioGenerateRuntimePhaseDomUpdate(node);
  markNodeDirty(node);
  return true;
}

function clearAioGenerateRuntimePhase(node) {
  if (!isAioGenerateNode(node) || !node[AIO_RUNTIME_PHASE_NODE_KEY]) {
    return false;
  }
  delete node[AIO_RUNTIME_PHASE_NODE_KEY];
  scheduleAioGenerateRuntimePhaseDomUpdate(node);
  markNodeDirty(node);
  return true;
}

function clearAioGenerateRuntimePhases() {
  let removed = false;
  for (const node of graphNodes()) {
    removed = clearAioGenerateRuntimePhase(node) || removed;
  }
  for (const label of document.querySelectorAll(`.${AIO_RUNTIME_PHASE_LABEL_CLASS}`)) {
    label.remove();
  }
  return removed;
}

function drawAioGenerateRuntimePhase(ctx, node) {
  const phase = node?.[AIO_RUNTIME_PHASE_NODE_KEY];
  if (!phase || !isAioGenerateNode(node)) {
    return;
  }
  const width = Number(node.size?.[0] || 0);
  if (!width || width < 96) {
    return;
  }
  const paddingX = 8;
  const height = 18;
  const maxWidth = Math.max(60, width - 20);
  ctx.save();
  ctx.font = "600 11px sans-serif";
  const text = fitString(ctx, phase, maxWidth - paddingX * 2);
  const textWidth = Math.min(ctx.measureText(text).width, maxWidth - paddingX * 2);
  const boxWidth = Math.ceil(textWidth + paddingX * 2);
  const x = Math.max(10, width - boxWidth - 10);
  const y = 4;
  ctx.globalAlpha = app.canvas?.editor_alpha ?? 1;
  ctx.fillStyle = HELTO.bg;
  ctx.strokeStyle = HELTO.borderStrong;
  ctx.beginPath();
  ctx.roundRect(x, y, boxWidth, height, [height * 0.5]);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = HELTO.text;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + paddingX, y + height * 0.5);
  ctx.restore();
}

function handleAioGenerateProgressText(event) {
  const { nodeId, text } = event?.detail || {};
  const node = findAioGenerateNodeById(nodeId);
  if (!node) {
    return;
  }
  setAioGenerateRuntimePhase(node, text);
}

function installAioGenerateRuntimePhaseBridge() {
  if (api[AIO_RUNTIME_PHASE_BRIDGE_KEY]) {
    return;
  }
  api[AIO_RUNTIME_PHASE_BRIDGE_KEY] = true;
  api.addEventListener?.("progress_text", handleAioGenerateProgressText);
}

function ensureAioGenerateRuntimePhaseUi(node) {
  if (!isAioGenerateNode(node)) {
    return;
  }
  installAioGenerateRuntimePhaseStyles();
  if (node._aioGenerateRuntimePhaseUiInstalled) {
    scheduleAioGenerateRuntimePhaseDomUpdate(node);
    return;
  }
  node._aioGenerateRuntimePhaseUiInstalled = true;
  const originalDrawForeground = node.onDrawForeground;
  node.onDrawForeground = function (ctx) {
    const result = originalDrawForeground?.apply(this, arguments);
    drawAioGenerateRuntimePhase(ctx, this);
    return result;
  };
  scheduleAioGenerateRuntimePhaseDomUpdate(node);
}

function clearAioGenerateProgressTextWidgets() {
  let removed = false;
  for (const node of graphNodes()) {
    removed = removeAioGenerateProgressTextWidget(node) || removed;
  }
  return removed;
}

function scheduleAioGenerateProgressTextCleanup() {
  const cleanup = () => {
    clearAioGenerateProgressTextWidgets();
    clearAioGenerateRuntimePhases();
  };
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(cleanup);
  } else {
    setTimeout(cleanup, 0);
  }
}

function installAioGenerateProgressTextCleanup() {
  if (api[AIO_PROGRESS_TEXT_CLEANUP_KEY]) {
    return;
  }
  api[AIO_PROGRESS_TEXT_CLEANUP_KEY] = true;
  api.addEventListener?.("execution_success", scheduleAioGenerateProgressTextCleanup);
  api.addEventListener?.("execution_error", scheduleAioGenerateProgressTextCleanup);
  api.addEventListener?.("execution_interrupted", scheduleAioGenerateProgressTextCleanup);
}

function suspendSeedControlCallbacks(controlWidget) {
  if (!controlWidget) {
    return null;
  }
  const beforeQueued = controlWidget.beforeQueued;
  const afterQueued = controlWidget.afterQueued;
  const beforeQueuedNoop = () => {};
  const afterQueuedNoop = () => {};
  controlWidget.beforeQueued = beforeQueuedNoop;
  controlWidget.afterQueued = afterQueuedNoop;
  return {
    controlWidget,
    beforeQueued,
    afterQueued,
    beforeQueuedNoop,
    afterQueuedNoop,
  };
}

function restoreSeedControlCallbacks(suspended) {
  for (const item of suspended) {
    if (item.controlWidget.beforeQueued === item.beforeQueuedNoop) {
      item.controlWidget.beforeQueued = item.beforeQueued;
    }
    if (item.controlWidget.afterQueued === item.afterQueuedNoop) {
      item.controlWidget.afterQueued = item.afterQueued;
    }
  }
}

function randomizeAioSeedsBeforeQueue() {
  const queuedSeeds = [];
  for (const node of graphNodes()) {
    if (!isAioGenerateNode(node) || liveSeedControlMode(node) !== "randomize") {
      continue;
    }
    const seedWidget = widgetByName(node, "seed");
    const controlWidget = seedControlWidget(node, seedWidget);
    const seed = randomFixedSeed();
    if (!writeAioSeedValue(node, seed)) {
      continue;
    }
    node._aioGenerateQueuedSeed = { seed, at: Date.now() };
    queuedSeeds.push({
      node,
      seed,
      suspended: suspendSeedControlCallbacks(controlWidget),
    });
  }
  return queuedSeeds;
}

function restoreQueuedAioSeeds(queuedSeeds) {
  restoreSeedControlCallbacks(queuedSeeds.map((item) => item.suspended).filter(Boolean));
  for (const { node, seed } of queuedSeeds) {
    const queuedSeed = node?._aioGenerateQueuedSeed;
    if (!queuedSeed || queuedSeed.seed !== seed || Date.now() - queuedSeed.at > 10000) {
      continue;
    }
    if (Number(widgetByName(node, "seed")?.value) !== Number(seed)) {
      writeAioSeedValue(node, seed);
    }
  }
}

function installAioSeedQueuePatch(source = "install") {
  if (typeof app.queuePrompt !== "function") {
    return false;
  }
  if (app.queuePrompt[AIO_SEED_QUEUE_WRAPPER_KEY]) {
    return true;
  }

  const originalQueuePrompt = app.queuePrompt;
  const wrappedQueuePrompt = async function (...args) {
    const queuedSeeds = randomizeAioSeedsBeforeQueue();
    try {
      return await originalQueuePrompt.apply(this, args);
    } finally {
      restoreQueuedAioSeeds(queuedSeeds);
    }
  };
  Object.defineProperty(wrappedQueuePrompt, AIO_SEED_QUEUE_WRAPPER_KEY, {
    value: true,
    configurable: true,
  });
  app.queuePrompt = wrappedQueuePrompt;
  return true;
}

function scheduleAioSeedQueuePatch(source = "top-level") {
  if (globalThis[AIO_SEED_QUEUE_INSTALL_KEY]) {
    installAioSeedQueuePatch(`${source}:resync`);
    return;
  }
  globalThis[AIO_SEED_QUEUE_INSTALL_KEY] = true;

  let attempts = 0;
  function attempt() {
    attempts += 1;
    installAioSeedQueuePatch(`${source}:${attempts}`);
    if (attempts < AIO_SEED_QUEUE_INSTALL_ATTEMPT_LIMIT) {
      setTimeout(attempt, 250);
    }
  }
  attempt();
}

function widgetSerializesToWorkflow(widget) {
  return Boolean(widget) && widget.serialize !== false && widget.options?.serialize !== false;
}

function serializedWidgetIndex(node, targetWidget) {
  let serializedIndex = 0;
  for (const widget of node?.widgets || []) {
    if (widget === targetWidget) {
      return widgetSerializesToWorkflow(widget) ? serializedIndex : -1;
    }
    if (widgetSerializesToWorkflow(widget)) {
      serializedIndex += 1;
    }
  }
  return -1;
}

function serializedWidgetCount(node) {
  let count = 0;
  for (const widget of node?.widgets || []) {
    if (widgetSerializesToWorkflow(widget)) {
      count += 1;
    }
  }
  return count;
}

function removeAioGenerateSeedButton(node) {
  const widgets = node?.widgets;
  if (!Array.isArray(widgets)) {
    return;
  }
  for (let index = widgets.length - 1; index >= 0; index -= 1) {
    if (widgets[index]?._aioGenerateSeedButton === true) {
      widgets.splice(index, 1);
    }
  }
}

function seedButtonWidgetValueIndex(node, values) {
  if (!Array.isArray(values)) {
    return -1;
  }
  const widgets = node?.widgets || [];
  const buttonIndex = widgets.findIndex((widget) => widget?._aioGenerateSeedButton === true);
  if (buttonIndex >= 0 && values.length === widgets.length) {
    return buttonIndex;
  }

  const stepsIndex = widgets.findIndex((widget) => widget?.name === "steps");
  if (stepsIndex >= 0 && values.length === widgets.length + 1) {
    return stepsIndex;
  }
  return -1;
}

function valuesWithoutSeedButtonSlot(node, values) {
  const index = seedButtonWidgetValueIndex(node, values);
  if (index < 0) {
    return values;
  }
  const normalized = values.slice();
  normalized.splice(index, 1);
  return normalized;
}

function valuesWithMissingBatchCountSlot(node, values) {
  if (!Array.isArray(values)) {
    return values;
  }
  const seedWidget = widgetByName(node, "seed");
  const batchWidget = widgetByName(node, BATCH_COUNT_WIDGET_NAME);
  const seedIndex = serializedWidgetIndex(node, seedWidget);
  const batchIndex = serializedWidgetIndex(node, batchWidget);
  const count = serializedWidgetCount(node);
  if (
    !seedWidget ||
    !batchWidget ||
    seedIndex < 0 ||
    batchIndex !== seedIndex + 1 ||
    values.length !== count - 1
  ) {
    return values;
  }
  const normalized = values.slice();
  normalized.splice(batchIndex, 0, 1);
  return normalized;
}

function valuesWithMissingZeroNegativeConditioningSlot(node, values) {
  if (!Array.isArray(values)) {
    return values;
  }
  const negativePromptWidget = widgetByName(node, "negative_prompt");
  const zeroNegativeWidget = widgetByName(node, USE_ZERO_NEGATIVE_CONDITIONING_WIDGET_NAME);
  const negativePromptIndex = serializedWidgetIndex(node, negativePromptWidget);
  const zeroNegativeIndex = serializedWidgetIndex(node, zeroNegativeWidget);
  const count = serializedWidgetCount(node);
  if (
    !negativePromptWidget ||
    !zeroNegativeWidget ||
    negativePromptIndex < 0 ||
    zeroNegativeIndex !== negativePromptIndex + 1 ||
    (values.length !== count - 1 && values.length !== count - 2)
  ) {
    return values;
  }
  const normalized = values.slice();
  normalized.splice(zeroNegativeIndex, 0, true);
  return normalized;
}

function normalizedAioGenerateWidgetValues(node, values) {
  const withoutSeedButton = valuesWithoutSeedButtonSlot(node, values);
  const withZeroNegativeConditioning = valuesWithMissingZeroNegativeConditioningSlot(node, withoutSeedButton);
  return valuesWithMissingBatchCountSlot(node, withZeroNegativeConditioning);
}

function configureInfoWithNormalizedGenerateWidgets(node, info) {
  if (!info || !Array.isArray(info.widgets_values)) {
    return info;
  }
  const values = normalizedAioGenerateWidgetValues(node, info.widgets_values);
  return values === info.widgets_values ? info : { ...info, widgets_values: values };
}

function patchWidgetCallback(widget, patchKey, callback) {
  if (!widget || widget[patchKey]) {
    return;
  }
  const originalCallback = widget.callback;
  widget.callback = function () {
    const result = originalCallback?.apply(this, arguments);
    callback();
    return result;
  };
  widget[patchKey] = true;
}

function updateAioMaxSideStep(node) {
  const maxSideWidget = widgetByName(node, "max side");
  const multipleWidget = widgetByName(node, "multiple value");
  if (!maxSideWidget || !multipleWidget) {
    return;
  }

  const step = multipleValueStep(multipleWidget.value);
  maxSideWidget.options ||= {};
  maxSideWidget.options.min = MAX_SIDE_MIN;
  maxSideWidget.options.max = MAX_SIDE_MAX;
  maxSideWidget.options.step = step * 10;
  maxSideWidget.options.step2 = step;
  maxSideWidget.options.precision = 0;

  const snapped = snapMaxSide(maxSideWidget.value, step);
  if (maxSideWidget.value !== snapped) {
    maxSideWidget.value = snapped;
    markNodeDirty(node);
  }
}

function ensureAioGenerateSizingUi(node) {
  if (!isAioGenerateNode(node)) {
    return;
  }

  const maxSideWidget = widgetByName(node, "max side");
  const multipleWidget = widgetByName(node, "multiple value");
  if (!maxSideWidget || !multipleWidget) {
    return;
  }

  patchWidgetCallback(multipleWidget, "_aioMultipleValueCallbackPatched", () => updateAioMaxSideStep(node));
  patchWidgetCallback(maxSideWidget, "_aioMaxSideCallbackPatched", () => updateAioMaxSideStep(node));
  updateAioMaxSideStep(node);
}

function showSeparateStrengths(node) {
  return widgetValue(node, "show_strengths", "single") === "separate";
}

function dynamicWidgets(node) {
  return (node.widgets || []).filter(
    (widget) =>
      String(widget.name).startsWith(ROW_PREFIX) ||
      widget.name === HEADER_NAME ||
      widget._aioLoraAddButton === true,
  );
}

function rowWidgets(node) {
  return (node.widgets || []).filter((widget) => String(widget.name).startsWith(ROW_PREFIX));
}

function moveArrayItem(array, item, index) {
  const current = array.indexOf(item);
  if (current < 0 || index < 0 || index >= array.length) {
    return;
  }
  array.splice(current, 1);
  array.splice(index, 0, item);
}

function removeArrayItem(array, item) {
  const index = array.indexOf(item);
  if (index >= 0) {
    array.splice(index, 1);
  }
}

function nextRowName(node) {
  let max = 0;
  for (const widget of node.widgets || []) {
    const match = String(widget.name || "").match(/^lora_(\d+)$/);
    if (match) {
      max = Math.max(max, Number(match[1]));
    }
  }
  return `${ROW_PREFIX}${max + 1}`;
}

function fitString(ctx, str, maxWidth) {
  str = String(str);
  if (ctx.measureText(str).width <= maxWidth) {
    return str;
  }
  const ellipsis = "...";
  let low = 0;
  let high = str.length;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (ctx.measureText(str.slice(0, mid) + ellipsis).width <= maxWidth) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  return str.slice(0, low) + ellipsis;
}

function isLowQuality() {
  return (app.canvas?.ds?.scale || 1) <= 0.5;
}

function drawRoundedRectangle(ctx, { pos, size, borderRadius = null }) {
  const radius = isLowQuality() ? 0 : borderRadius ?? size[1] * 0.5;
  ctx.save();
  ctx.strokeStyle = HELTO.borderStrong;
  ctx.fillStyle = HELTO.surface2;
  ctx.beginPath();
  ctx.roundRect(pos[0], pos[1], size[0], size[1], [radius]);
  ctx.fill();
  if (!isLowQuality()) {
    ctx.stroke();
  }
  ctx.restore();
}

function drawTogglePart(ctx, { posX, posY, height, value }) {
  const lowQuality = isLowQuality();
  const toggleRadius = height * 0.36;
  const toggleBgWidth = height * 1.5;
  ctx.save();
  if (!lowQuality) {
    ctx.beginPath();
    ctx.roundRect(posX + 4, posY + 4, toggleBgWidth - 8, height - 8, [height * 0.5]);
    ctx.globalAlpha = app.canvas.editor_alpha * 0.25;
    ctx.fillStyle = HELTO.surface3;
    ctx.fill();
    ctx.globalAlpha = app.canvas.editor_alpha;
  }
  ctx.fillStyle = value === true ? HELTO.accent : HELTO.textFaint;
  const toggleX =
    lowQuality || value === false ? posX + height * 0.5 : value === true ? posX + height : posX + height * 0.75;
  ctx.beginPath();
  ctx.arc(toggleX, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  return [posX, posY, toggleBgWidth, height];
}

function drawNumberWidgetPart(ctx, { posX, posY, height, value, direction = -1, textColor }) {
  const arrowWidth = 9;
  const arrowHeight = 10;
  const innerMargin = 3;
  const numberWidth = 32;
  let x = direction === -1 ? posX - NUMBER_WIDTH_TOTAL : posX;
  const midY = posY + height / 2;

  ctx.save();
  ctx.fillStyle = HELTO.textDim;
  ctx.fill(new Path2D(`M ${x} ${midY} l ${arrowWidth} ${arrowHeight / 2} l 0 -${arrowHeight} L ${x} ${midY} z`));
  const left = [x, posY, arrowWidth, height];
  x += arrowWidth + innerMargin;

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  if (textColor) {
    ctx.fillStyle = textColor;
  }
  ctx.fillText(fitString(ctx, Number(value ?? 1).toFixed(2), numberWidth), x + numberWidth / 2, midY);
  const text = [x, posY, numberWidth, height];
  x += numberWidth + innerMargin;

  ctx.fillStyle = HELTO.textDim;
  ctx.fill(new Path2D(`M ${x} ${midY - arrowHeight / 2} l ${arrowWidth} ${arrowHeight / 2} l -${arrowWidth} ${arrowHeight / 2} v -${arrowHeight} z`));
  const right = [x, posY, arrowWidth, height];
  ctx.restore();
  return [left, text, right, [left[0], posY, right[0] + right[2] - left[0], height]];
}

function drawInfoIcon(ctx, x, y, size, treatment = "GRAYED") {
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(x, y, size, size, [size * 0.1]);
  // GRAYED = no info (faint); OUTLINED/FILLED = info available -> gold accent.
  ctx.fillStyle = treatment === "GRAYED" ? HELTO.textFaint : HELTO.accent;
  ctx.strokeStyle = ctx.fillStyle;
  if (treatment === "FILLED") {
    ctx.fill();
  } else {
    ctx.stroke();
  }
  ctx.strokeStyle = treatment === "FILLED" ? HELTO.bg : HELTO.textDim;
  ctx.lineWidth = 2;
  const midX = x + size / 2;
  const serif = size * 0.175;
  ctx.stroke(
    new Path2D(`
      M ${midX} ${y + size * 0.15}
      v 2
      M ${midX - serif} ${y + size * 0.45}
      h ${serif}
      v ${size * 0.325}
      h ${serif}
      h -${serif * 2}
    `),
  );
  ctx.restore();
}

function inArea(pos, area) {
  return (
    area &&
    pos[0] >= area[0] &&
    pos[0] <= area[0] + area[2] &&
    pos[1] >= area[1] &&
    pos[1] <= area[1] + area[3]
  );
}

async function getLoras(force = false) {
  if (force) {
    loraListPromise = null;
  }
  if (!loraListPromise) {
    loraListPromise = api
      .fetchApi("/aio-image-gen/api/loras?format=details", { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error("No AIO loras route"))))
      .then((data) => data.map((item) => item.file ?? item))
      .catch(() =>
        api
          .fetchApi("/object_info/LoraLoader", { cache: "no-store" })
          .then((response) => response.json())
          .then((data) => data?.LoraLoader?.input?.required?.lora_name?.[0] || [])
          .catch(() => []),
      );
  }
  return loraListPromise;
}

async function fetchLoraInfo(file, { refresh = false, light = false } = {}) {
  const endpoint = refresh ? "/aio-image-gen/api/loras/info/refresh" : "/aio-image-gen/api/loras/info";
  const params = new URLSearchParams({ files: file });
  if (light) {
    params.set("light", "1");
  }
  const response = await api.fetchApi(`${endpoint}?${params.toString()}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`AIO LoRA info request failed: ${response.status}`);
  }
  const payload = await response.json();
  return payload?.data?.[0] ?? null;
}

function filteredChooserData(node, loras) {
  let filtered = [...loras];
  let prefix = "";
  const match = String(widgetValue(node, "match", "") || "");
  if (match) {
    try {
      const regex = new RegExp(match);
      filtered = filtered.filter((lora) => regex.test(lora));
    } catch {
      filtered = [...loras];
    }
  }

  if (filtered.length > 0) {
    prefix = filtered[0];
    for (const lora of filtered) {
      let common = "";
      for (let index = 0; prefix[index] && prefix[index] === lora[index]; index++) {
        common += prefix[index];
      }
      prefix = common;
      if (!prefix) {
        break;
      }
    }
    if (prefix) {
      filtered = filtered.map((lora) => lora.replace(prefix, ""));
    }
  }

  return { prefix, choices: filtered };
}

async function showLoraChooser(event, node, onChoose) {
  const { prefix, choices } = filteredChooserData(node, await getLoras());
  new LiteGraph.ContextMenu(["None", ...choices], {
    event,
    title: "Choose LoRA",
    className: "dark",
    callback: (value) => {
      if (typeof value === "string" && value !== "None") {
        onChoose(prefix + value);
      }
      node.setDirtyCanvas(true, true);
    },
  });
}

function showFallbackInfo(file, error = null) {
  const message = error ? `Could not load LoRA/Civitai info for ${file}: ${error.message}` : `LoRA: ${file}`;
  if (app.extensionManager?.toast) {
    app.extensionManager.toast.add({ severity: error ? "warn" : "info", summary: "LoRA", detail: message });
    return;
  }
  console.info(`[AIO LoRA Configuration] ${message}`);
}

const CIVITAI_LOGO = `<svg class="logo-civitai" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M7.2 3.8 12 1l4.8 2.8v5.5l4.8 2.7v5.6L12 23l-9.6-5.4V12l4.8-2.7V3.8Zm1.6 1v5.4L4 13v3.6l8 4.5 8-4.5V13l-4.8-2.8V4.8L12 3 8.8 4.8Zm1.6 7.3L12 11l1.6 1.1v2.1L12 15.2l-1.6-1v-2.1Z"/></svg>`;
const EXTERNAL_ICON = `<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M10 2h4v4h-1.5V4.6L7 10.1 5.9 9 11.4 3.5H10V2ZM3.5 4h4v1.5h-4v7h7v-4H12v4.5a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h.5Z"/></svg>`;
const EDIT_ICON = `<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M11.9 1.7 14.3 4 5.5 12.8 2.6 13.4l.6-2.9 8.7-8.8Zm-.9 2.1-6.4 6.4-.2.9.9-.2 6.4-6.4-.7-.7Z"/></svg>`;
const SAVE_ICON = `<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M2 2h10.5L14 3.5V14H2V2Zm2 1.5v3h7v-3H4Zm0 9h8v-4H4v4Z"/></svg>`;

async function saveLoraInfoPartial(file, partial) {
  const body = new FormData();
  body.append("json", JSON.stringify(partial));
  const response = await api.fetchApi(`/aio-image-gen/api/loras/info?file=${encodeURIComponent(file)}`, {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw new Error(`Save failed: ${response.status}`);
  }
  const payload = await response.json();
  return payload?.data ?? null;
}

function infoTableRow(label, value, help = "", editableFieldName = "") {
  if (value == null || value === "") {
    return "";
  }
  return `
    <tr class="${editableFieldName ? "editable" : ""}" ${editableFieldName ? `data-field-name="${editableFieldName}"` : ""}>
      <td><span>${escapeHtml(label)} ${help ? `<span class="-help" title="${escapeHtml(help)}"></span>` : ""}<span></td>
      <td ${editableFieldName ? "" : 'colspan="2"'}>${escapedValueMarkup(value)}</td>
      ${
        editableFieldName
          ? `<td style="width: 24px;"><button class="rgthree-button-reset rgthree-button-edit" data-action="edit-row">${EDIT_ICON}${SAVE_ICON}</button></td>`
          : ""
      }
    </tr>`;
}

function infoTableMarkupRow(label, markup, help = "") {
  if (!markup) {
    return "";
  }
  return `
    <tr>
      <td><span>${escapeHtml(label)} ${help ? `<span class="-help" title="${escapeHtml(help)}"></span>` : ""}</span></td>
      <td colspan="2">${markup}</td>
    </tr>`;
}

function trainedWordsMarkup(words) {
  if (!words?.length) {
    return "";
  }
  return `<ul class="rgthree-info-trained-words-list">${words
    .map((item) => {
      const word = item.word ?? item;
      return `<li title="${escapeHtml(word)}" data-word="${escapeHtml(word)}" class="rgthree-info-trained-words-list-item" data-action="toggle-trained-word">
        <span>${escapeHtml(word)}</span>
        ${item.civitai ? CIVITAI_LOGO : ""}
        ${item.count != null ? `<small>${escapeHtml(item.count)}</small>` : ""}
      </li>`;
    })
    .join("")}</ul>`;
}

function imagesMarkup(images) {
  if (!images?.length) {
    return "";
  }
  return `<ul class="rgthree-info-images">${images
    .map((image) => {
      const mediaUrl = safeUrl(image.url, { allowRelative: true });
      if (!mediaUrl) {
        return "";
      }
      const media =
        image.type === "video"
          ? `<video src="${escapeHtml(mediaUrl)}" autoplay loop muted></video>`
          : `<img src="${escapeHtml(mediaUrl)}" alt="">`;
      const civitaiUrl = safeUrl(image.civitaiUrl);
      return `<li><figure>${media}<figcaption>
        ${civitaiUrl ? imgInfoMarkup(`<a href="${escapeHtml(civitaiUrl)}" target="_blank" rel="noreferrer">civitai${EXTERNAL_ICON}</a>`) : ""}
        ${imgInfoField("seed", image.seed)}
        ${imgInfoField("steps", image.steps)}
        ${imgInfoField("cfg", image.cfg)}
        ${imgInfoField("sampler", image.sampler)}
        ${imgInfoField("model", image.model)}
        ${imgInfoField("positive", image.positive)}
        ${imgInfoField("negative", image.negative)}
      </figcaption></figure></li>`;
    })
    .join("")}</ul>`;
}

function imgInfoField(label, value) {
  return imageInfoFieldMarkup(label, value);
}

function imgInfoMarkup(markup) {
  return markup ? `<span>${markup}</span>` : "";
}

function renderInfoDialogContent(container, info, file, isLoading = false) {
  const civitaiLink = safeUrl(info?.links?.find((link) => String(link).includes("civitai.com/models")));
  const civitaiError = info?.raw?.civitai?.error;
  const civitaiValue = civitaiLink
    ? `<a href="${escapeHtml(civitaiLink)}" target="_blank" rel="noreferrer">${CIVITAI_LOGO}View on Civitai</a>`
    : civitaiError
      ? String(civitaiError) === "Model not found"
        ? `<i>Model not found</i> <span class="-help" title="The model was not found on civitai with the sha256 hash. It is possible the model was removed, re-uploaded, or was never on civitai to begin with."></span>`
        : escapeHtml(civitaiError)
      : !info?.raw?.civitai
        ? `<button type="button" class="rgthree-button" data-action="fetch-civitai">Fetch info from civitai</button>`
        : "";
  const trainedWords = trainedWordsMarkup(info?.trainedWords);
  const metadata = info?.raw?.metadata || {};
  const title = info?.name || info?.file || file || "Unknown";
  container.innerHTML = `
    <div class="rgthree-info-dialog">
      <div class="aio-rgthree-dialog-title">
        <h2>${escapeHtml(title)}</h2>
        <button type="button" class="aio-lora-close" aria-label="Close">x</button>
      </div>
      <div class="aio-rgthree-dialog-content">
        ${isLoading ? `<div class="aio-lora-loading">Loading...</div>` : ""}
        <ul class="rgthree-info-area">
          <li title="Type" class="rgthree-info-tag -type -type-${escapeHtml((info?.type || "").toLowerCase())}"><span>${escapeHtml(info?.type || "")}</span></li>
          <li title="Base Model" class="rgthree-info-tag -basemodel -basemodel-${escapeHtml((info?.baseModel || "").toLowerCase())}"><span>${escapeHtml(info?.baseModel || "")}</span></li>
          <li class="rgthree-info-menu"></li>
        </ul>
        <table class="rgthree-info-table">
          ${infoTableRow("File", info?.file || file)}
          ${infoTableRow("Hash (sha256)", info?.sha256)}
          ${infoTableMarkupRow("Civitai", civitaiValue)}
          ${infoTableRow("Name", info?.name || metadata.ss_output_name || "", "The name for display.", "name")}
          ${!info?.baseModelFile && !info?.baseModel ? "" : infoTableRow("Base Model", `${info?.baseModel || ""}${info?.baseModelFile ? ` (${info.baseModelFile})` : ""}`)}
          ${trainedWords ? infoTableMarkupRow("Trained Words", trainedWords, "Trained words from the metadata and/or civitai. Click to select for copy.") : ""}
          ${!metadata.ss_clip_skip || metadata.ss_clip_skip === "None" ? "" : infoTableRow("Clip Skip", metadata.ss_clip_skip)}
          ${infoTableRow("Strength Min", info?.strengthMin ?? "", "The recommended minimum strength. In the Power Lora Loader node, strength will signal when it is below this threshold.", "strengthMin")}
          ${infoTableRow("Strength Max", info?.strengthMax ?? "", "The recommended maximum strength. In the Power Lora Loader node, strength will signal when it is above this threshold.", "strengthMax")}
          ${infoTableRow("Additional Notes", info?.userNote ?? "", "Additional notes you'd like to keep and reference in the info dialog.", "userNote")}
        </table>
        ${imagesMarkup(info?.images)}
      </div>
    </div>
  `;
}

function ensureDialogStyles() {
  ensureHeltoTokens();
  if (document.getElementById("aio-lora-info-styles")) {
    return;
  }
  const style = document.createElement("style");
  style.id = "aio-lora-info-styles";
  style.textContent = `
    /* ---- Overlay + modal card (Helto modal recipe) ---- */
    .aio-lora-info-overlay {
      position: fixed;
      inset: 0;
      z-index: 10000;
      display: grid;
      place-items: center;
      padding: 12px;
      background: color-mix(in srgb, var(--helto-bg) 72%, transparent);
      backdrop-filter: blur(4px);
      animation: aio-lora-fade 0.2s ease;
    }
    @keyframes aio-lora-fade { from { opacity: 0; } to { opacity: 1; } }
    .rgthree-info-dialog {
      width: 90vw;
      max-width: 960px;
      max-height: calc(100vh - 48px);
      overflow: hidden;
      border: 1px solid var(--helto-border-strong);
      border-radius: var(--helto-radius-lg);
      background: linear-gradient(135deg, var(--helto-surface-2), var(--helto-bg));
      color: var(--helto-text);
      box-shadow: var(--helto-shadow-pop);
      backdrop-filter: blur(15px);
      font: var(--helto-font-size)/var(--helto-line) var(--helto-font-sans);
      -webkit-font-smoothing: antialiased;
      animation: aio-lora-rise 0.2s var(--helto-ease-spring);
    }
    @keyframes aio-lora-rise { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
    .rgthree-info-dialog *, .rgthree-info-dialog *::before, .rgthree-info-dialog *::after { box-sizing: border-box; }
    .rgthree-info-dialog ::-webkit-scrollbar { width: 6px; height: 6px; }
    .rgthree-info-dialog ::-webkit-scrollbar-track { background: transparent; }
    .rgthree-info-dialog ::-webkit-scrollbar-thumb { background: var(--helto-border-strong); border-radius: 3px; }
    .rgthree-info-dialog ::-webkit-scrollbar-thumb:hover { background: var(--helto-text-faint); }
    .aio-rgthree-dialog-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 48px;
      padding: 10px 14px 10px 16px;
      border-bottom: 1px solid var(--helto-border);
      color: var(--helto-text);
      font-weight: 700;
    }
    .aio-rgthree-dialog-title h2 {
      margin: 0;
      font-size: 15px;
      line-height: 1.2;
      letter-spacing: 0.02em;
      color: var(--helto-text);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .aio-lora-close {
      flex: 0 0 auto;
      width: 28px;
      height: 28px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--helto-border-strong);
      border-radius: var(--helto-radius-sm);
      background: linear-gradient(180deg, var(--helto-surface-3), var(--helto-surface-2));
      color: var(--helto-text-dim);
      font-size: 18px;
      line-height: 1;
      cursor: pointer;
      transition: background var(--helto-transition), border-color var(--helto-transition), color var(--helto-transition);
    }
    .aio-lora-close:hover {
      background: linear-gradient(180deg, var(--helto-surface-hover), var(--helto-surface-3));
      border-color: var(--helto-border-hover);
      color: var(--helto-text);
    }
    .aio-lora-close:focus-visible { outline: none; border-color: var(--helto-focus); box-shadow: var(--helto-focus-ring); }
    .aio-rgthree-dialog-content {
      padding: 14px 16px 16px;
      max-height: calc(100vh - 96px);
      overflow: auto;
    }
    .aio-lora-loading {
      padding: 7px 10px;
      margin-bottom: 12px;
      color: var(--helto-text-dim);
      background: var(--helto-surface-2);
      border: 1px solid var(--helto-border);
      border-radius: var(--helto-radius);
    }
    .rgthree-button,
    .rgthree-button-reset {
      font: inherit;
      color: inherit;
    }
    .rgthree-button {
      height: 24px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--helto-border-strong);
      border-radius: var(--helto-radius-sm);
      background: linear-gradient(180deg, var(--helto-surface-3), var(--helto-surface-2));
      color: var(--helto-text);
      padding: 0 12px;
      cursor: pointer;
      transition: background var(--helto-transition), border-color var(--helto-transition), color var(--helto-transition);
    }
    .rgthree-button:hover {
      background: linear-gradient(180deg, var(--helto-surface-hover), var(--helto-surface-3));
      border-color: var(--helto-border-hover);
      color: var(--helto-text);
    }
    .rgthree-button:focus-visible { outline: none; border-color: var(--helto-focus); box-shadow: var(--helto-focus-ring); }
    .rgthree-button-reset {
      border: 0;
      padding: 0;
      background: transparent;
      cursor: pointer;
      color: var(--helto-text-dim);
      transition: color var(--helto-transition);
    }
    .rgthree-button-reset:hover { color: var(--helto-accent-strong); }
    .rgthree-info-dialog .rgthree-info-area {
      list-style: none;
      padding: 0;
      margin: 0;
      display: flex;
      align-items: center;
    }
    .rgthree-info-dialog .rgthree-info-area > li {
      display: inline-flex;
      margin: 0;
      vertical-align: top;
    }
    .rgthree-info-dialog .rgthree-info-area > li + li {
      margin-left: 6px;
    }
    /* Type / base-model badges = Helto info pills. */
    .rgthree-info-dialog .rgthree-info-area > li.rgthree-info-tag > * {
      min-height: 24px;
      border-radius: 999px;
      line-height: 1;
      color: var(--helto-text-dim);
      background: var(--helto-surface-2);
      border: 1px solid var(--helto-border-strong);
      font-size: 12px;
      font-weight: 600;
      text-decoration: none;
      display: flex;
      height: 1.7em;
      padding: 0 0.7em 0.1em;
      align-content: center;
      justify-content: center;
      align-items: center;
    }
    .rgthree-info-dialog .rgthree-info-area > li.rgthree-info-tag > *:empty {
      display: none;
    }
    .rgthree-info-dialog .rgthree-info-area > li.-type > * {
      background: var(--helto-bg);
      border-color: var(--helto-info);
      color: var(--helto-info);
    }
    .rgthree-info-dialog .rgthree-info-area > li.rgthree-info-menu {
      margin-left: auto;
    }
    .rgthree-info-dialog .rgthree-info-table {
      border-collapse: collapse;
      margin: 16px 0;
      width: 100%;
      font-size: 12px;
    }
    .rgthree-info-dialog .rgthree-info-table tr.editable button {
      display: flex;
      width: 28px;
      height: 28px;
      align-items: center;
      justify-content: center;
    }
    .rgthree-info-dialog .rgthree-info-table tr.editable button svg + svg {
      display: none;
    }
    .rgthree-info-dialog .rgthree-info-table tr.editable.-rgthree-editing button svg {
      display: none;
    }
    .rgthree-info-dialog .rgthree-info-table tr.editable.-rgthree-editing button svg + svg {
      display: inline-block;
    }
    .rgthree-info-dialog .rgthree-info-table td {
      position: relative;
      border: 1px solid var(--helto-border);
      padding: 0;
      vertical-align: top;
    }
    .rgthree-info-dialog .rgthree-info-table td:first-child {
      background: var(--helto-surface-2);
      width: 10px;
      color: var(--helto-text-dim);
    }
    .rgthree-info-dialog .rgthree-info-table td:first-child > *:first-child {
      white-space: nowrap;
      padding-right: 32px;
    }
    .rgthree-info-dialog .rgthree-info-table td:first-child small {
      display: block;
      margin-top: 2px;
      color: var(--helto-text-faint);
    }
    .rgthree-info-dialog .rgthree-info-table td:first-child small > [data-action] {
      color: var(--helto-accent);
      text-decoration: underline;
      cursor: pointer;
    }
    .rgthree-info-dialog .rgthree-info-table td:first-child small > [data-action]:hover {
      text-decoration: none;
    }
    .rgthree-info-dialog .rgthree-info-table td a {
      color: var(--helto-accent);
    }
    .rgthree-info-dialog .rgthree-info-table td a:hover { color: var(--helto-accent-strong); }
    .rgthree-info-dialog .rgthree-info-table td a:visited { color: var(--helto-accent); }
    .rgthree-info-dialog .rgthree-info-table td svg {
      width: 1.3333em;
      height: 1.3333em;
      vertical-align: -0.285em;
    }
    .rgthree-info-dialog .rgthree-info-table td svg.logo-civitai {
      margin-right: 0.3333em;
    }
    .rgthree-info-dialog .rgthree-info-table td > *:first-child {
      display: block;
      padding: 6px 10px;
    }
    .rgthree-info-dialog .rgthree-info-table td > input,
    .rgthree-info-dialog .rgthree-info-table td > textarea {
      padding: 5px 10px;
      border: 0;
      box-shadow: inset 0 0 0 1px var(--helto-border-strong);
      font: inherit;
      appearance: none;
      background: var(--helto-bg);
      color: var(--helto-text);
      resize: vertical;
    }
    .rgthree-info-dialog .rgthree-info-table td > input:focus,
    .rgthree-info-dialog .rgthree-info-table td > textarea:focus {
      outline: none;
      box-shadow: inset 0 0 0 1px var(--helto-focus), var(--helto-focus-ring);
    }
    .rgthree-info-dialog .rgthree-info-table td > input:only-child,
    .rgthree-info-dialog .rgthree-info-table td > textarea:only-child {
      width: 100%;
      box-sizing: border-box;
    }
    .rgthree-info-dialog .rgthree-info-table td .-help {
      border: 1px solid currentColor;
      position: absolute;
      right: 5px;
      top: 6px;
      line-height: 1;
      font-size: 11px;
      width: 12px;
      height: 12px;
      border-radius: 8px;
      display: flex;
      align-content: center;
      justify-content: center;
      color: var(--helto-text-faint);
      cursor: help;
    }
    .rgthree-info-dialog .rgthree-info-table td .-help::before {
      content: "?";
    }
    .rgthree-info-dialog .rgthree-info-table td > ul.rgthree-info-trained-words-list {
      list-style: none;
      padding: 2px 8px;
      margin: 0;
      display: flex;
      flex-direction: row;
      flex-wrap: wrap;
      max-height: 15vh;
      overflow: auto;
    }
    /* Trained-word chips: pill default, GOLD when selected. */
    .rgthree-info-dialog .rgthree-info-table td > ul.rgthree-info-trained-words-list > li {
      display: inline-flex;
      margin: 2px;
      vertical-align: top;
      border-radius: 999px;
      line-height: 1;
      color: var(--helto-text-dim);
      background: var(--helto-surface-2);
      border: 1px solid var(--helto-border-strong);
      font-size: 1.1em;
      font-weight: 600;
      text-decoration: none;
      height: 1.7em;
      align-content: center;
      justify-content: center;
      align-items: center;
      cursor: pointer;
      white-space: nowrap;
      max-width: 183px;
      transition: background var(--helto-transition), border-color var(--helto-transition), color var(--helto-transition);
    }
    .rgthree-info-dialog .rgthree-info-table td > ul.rgthree-info-trained-words-list > li:hover {
      background: var(--helto-surface-hover);
      border-color: var(--helto-border-hover);
      color: var(--helto-text);
    }
    .rgthree-info-dialog .rgthree-info-table td > ul.rgthree-info-trained-words-list > li > svg {
      width: auto;
      height: 1.2em;
    }
    .rgthree-info-dialog .rgthree-info-table td > ul.rgthree-info-trained-words-list > li > span {
      padding-left: 0.6em;
      padding-right: 0.6em;
      padding-bottom: 0.1em;
      text-overflow: ellipsis;
      overflow: hidden;
    }
    .rgthree-info-dialog .rgthree-info-table td > ul.rgthree-info-trained-words-list > li > small {
      align-self: stretch;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 0.6em;
      background: var(--helto-surface);
    }
    .rgthree-info-dialog .rgthree-info-table td > ul.rgthree-info-trained-words-list > li.-rgthree-is-selected {
      background: var(--helto-accent-bg);
      border-color: var(--helto-accent-border);
      color: var(--helto-accent-strong);
      box-shadow: var(--helto-shadow-glow);
    }
    .rgthree-info-dialog .rgthree-info-images {
      list-style: none;
      padding: 0;
      margin: 0;
      scroll-snap-type: x mandatory;
      display: flex;
      flex-direction: row;
      overflow: auto;
    }
    .rgthree-info-dialog .rgthree-info-images > li {
      scroll-snap-align: start;
      max-width: 90%;
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      overflow: hidden;
      padding: 0;
      margin: 6px;
      font-size: 0;
      position: relative;
      border: 1px solid var(--helto-border);
      border-radius: var(--helto-radius);
      background: var(--helto-bg);
    }
    .rgthree-info-dialog .rgthree-info-images > li figure {
      margin: 0;
      position: static;
    }
    .rgthree-info-dialog .rgthree-info-images > li figure video,
    .rgthree-info-dialog .rgthree-info-images > li figure img {
      max-height: 45vh;
      max-width: 100%;
    }
    .rgthree-info-dialog .rgthree-info-images > li figure figcaption {
      position: absolute;
      left: 0;
      width: 100%;
      bottom: 0;
      padding: 12px;
      font-size: 12px;
      background: color-mix(in srgb, var(--helto-bg) 85%, transparent);
      opacity: 0;
      transform: translateY(50px);
      transition: all 0.25s ease-in-out;
      box-sizing: border-box;
    }
    .rgthree-info-dialog .rgthree-info-images > li figure figcaption > span {
      display: inline-block;
      padding: 2px 5px;
      margin: 2px;
      border-radius: var(--helto-radius-sm);
      border: 1px solid var(--helto-border-strong);
      color: var(--helto-text);
      word-break: break-word;
    }
    .rgthree-info-dialog .rgthree-info-images > li figure figcaption > span label {
      display: inline;
      padding: 0;
      margin: 0;
      color: var(--helto-text-faint);
      pointer-events: none;
      user-select: none;
    }
    .rgthree-info-dialog .rgthree-info-images > li figure figcaption > span a {
      color: var(--helto-accent);
      text-decoration: underline;
    }
    .rgthree-info-dialog .rgthree-info-images > li figure figcaption:empty {
      text-align: center;
    }
    .rgthree-info-dialog .rgthree-info-images > li figure figcaption:empty::before {
      content: "No data.";
      color: var(--helto-text-faint);
    }
    .rgthree-info-dialog .rgthree-info-images > li:hover figure figcaption {
      opacity: 1;
      transform: translateY(0);
    }
  `;
  document.head.appendChild(style);
}

function showInfoToast(message, severity = "info") {
  if (app.extensionManager?.toast) {
    app.extensionManager.toast.add({ severity, summary: "LoRA", detail: message });
    return;
  }
  console.info(`[AIO LoRA Configuration] ${message}`);
}

function selectedWordElements(tr) {
  return Array.from(tr?.querySelectorAll(".-rgthree-is-selected") || []);
}

function updateSelectedWordsSummary(tr) {
  const labelSpan = tr?.querySelector("td:first-child > *");
  if (!labelSpan) {
    return;
  }
  let small = labelSpan.querySelector("small");
  if (!small) {
    small = document.createElement("small");
    labelSpan.appendChild(small);
  }
  const count = selectedWordElements(tr).length;
  small.innerHTML = count
    ? `${count} selected | <span role="button" data-action="copy-trained-words">Copy</span>`
    : "";
}

async function copySelectedWords(target) {
  const tr = target.closest("tr");
  const words = selectedWordElements(tr).map((el) => el.getAttribute("data-word")).filter(Boolean);
  await navigator.clipboard.writeText(words.join(", "));
  showInfoToast(`Successfully copied ${words.length} key word${words.length === 1 ? "" : "s"}.`, "success");
}

async function saveEditableRow(info, file, tr, saving = true) {
  const fieldName = tr?.dataset?.fieldName;
  const td = tr?.querySelector("td:nth-child(2)");
  const input = td?.querySelector("input,textarea");
  if (!fieldName || !td) {
    return false;
  }

  let newValue = info?.[fieldName] ?? "";
  let modified = false;
  if (saving && input) {
    newValue = input.value;
    if (fieldName.startsWith("strength")) {
      if (Number.isNaN(Number(newValue))) {
        alert(`You must enter a number into the ${fieldName} field.`);
        return false;
      }
      newValue = (Math.round(Number(newValue) * 100) / 100).toFixed(2);
    }
    const saved = await saveLoraInfoPartial(file, { [fieldName]: newValue });
    Object.assign(info, saved || { [fieldName]: newValue });
    modified = true;
  }

  tr.classList.remove("-rgthree-editing");
  td.replaceChildren();
  const span = document.createElement("span");
  span.textContent = newValue;
  td.appendChild(span);
  return modified;
}

function beginEditableRow(info, file, tr) {
  const fieldName = tr?.dataset?.fieldName;
  const td = tr?.querySelector("td:nth-child(2)");
  if (!fieldName || !td) {
    return;
  }
  tr.classList.add("-rgthree-editing");
  const isTextarea = fieldName === "userNote";
  const input = document.createElement(isTextarea ? "textarea" : "input");
  if (!isTextarea) {
    input.type = "text";
  }
  input.value = td.textContent || info?.[fieldName] || "";
  input.addEventListener("keydown", async (event) => {
    if (!isTextarea && event.key === "Enter") {
      event.preventDefault();
      event.stopPropagation();
      await saveEditableRow(info, file, tr, true);
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      await saveEditableRow(info, file, tr, false);
    }
  });
  td.replaceChildren(input);
  input.focus();
}

async function showLoraInfoDialog(file, row = null) {
  ensureDialogStyles();
  const overlay = document.createElement("div");
  overlay.className = "aio-lora-info-overlay";
  document.body.appendChild(overlay);

  const close = () => overlay.remove();
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay || event.target.closest(".aio-lora-close")) {
      close();
    }
  });

  let info = null;
  try {
    renderInfoDialogContent(overlay, null, file, true);
    info = await fetchLoraInfo(file);
    renderInfoDialogContent(overlay, info, file, false);
    row?.setLoraInfo?.(info);
  } catch (error) {
    close();
    showFallbackInfo(file, error);
    return;
  }

  overlay.addEventListener("click", async (event) => {
    const target = event.target.closest("[data-action]");
    const action = target?.getAttribute("data-action");
    if (!target || !action) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();

    if (action === "fetch-civitai") {
      renderInfoDialogContent(overlay, info, file, true);
      try {
        info = await fetchLoraInfo(file, { refresh: true });
        renderInfoDialogContent(overlay, info, file, false);
        row?.setLoraInfo?.(info);
      } catch (error) {
        renderInfoDialogContent(overlay, info, file, false);
        showFallbackInfo(file, error);
      }
    } else if (action === "toggle-trained-word") {
      target.classList.toggle("-rgthree-is-selected");
      updateSelectedWordsSummary(target.closest("tr"));
    } else if (action === "copy-trained-words") {
      await copySelectedWords(target);
    } else if (action === "edit-row") {
      const tr = target.closest("tr");
      if (tr?.querySelector("input,textarea")) {
        await saveEditableRow(info, file, tr, true);
        row?.setLoraInfo?.(info);
      } else {
        beginEditableRow(info, file, tr);
      }
    }
  });
}

class LoraHeaderWidget {
  constructor() {
    this.name = HEADER_NAME;
    this.type = "custom";
    this.value = { type: HEADER_NAME };
    this.tooltip = LORA_HEADER_TOOLTIP;
    this.last_y = 0;
    this.hitAreas = {};
  }

  computeSize(width) {
    return [width, LiteGraph.NODE_WIDGET_HEIGHT];
  }

  serializeValue() {
    return this.value;
  }

  draw(ctx, node, width, posY, height) {
    const nodeWidth = node.size?.[0] ?? width;
    this.last_y = posY;
    if (!rowWidgets(node).length) {
      return;
    }
    const margin = 10;
    const innerMargin = margin * 0.33;
    const midY = posY + height / 2;
    let posX = margin;
    const separate = showSeparateStrengths(node);

    ctx.save();
    this.hitAreas.toggle = drawTogglePart(ctx, {
      posX,
      posY: posY + 2,
      height,
      value: allLorasState(node),
    });
    if (!isLowQuality()) {
      posX += this.hitAreas.toggle[2] + innerMargin;
      ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
      ctx.fillStyle = HELTO.textDim;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText("Toggle All", posX, midY);

      let rightX = nodeWidth - margin - innerMargin - innerMargin;
      ctx.textAlign = "center";
      ctx.fillText(separate ? "Clip" : "Strength", rightX - NUMBER_WIDTH_TOTAL / 2, midY);
      if (separate) {
        rightX = rightX - NUMBER_WIDTH_TOTAL - innerMargin * 2;
        ctx.fillText("Model", rightX - NUMBER_WIDTH_TOTAL / 2, midY);
      }
    }
    ctx.restore();
  }

  mouse(event, pos, node) {
    if (event.type === "pointerdown" && inArea(pos, this.hitAreas.toggle)) {
      toggleAll(node);
      return true;
    }
    return false;
  }
}

class LoraRowWidget {
  constructor(name, value = null) {
    this.name = name;
    this.type = "custom";
    this.value = { ...DEFAULT_ROW, ...(value || {}) };
    this.tooltip = LORA_ROW_TOOLTIP;
    this.last_y = 0;
    this.hitAreas = {};
    this.showModelAndClip = null;
    this.haveMouseMovedStrength = false;
    this.activeStrengthKey = null;
    this.loraInfo = null;
    this.loraInfoPromise = null;
    this.getLoraInfo();
  }

  computeSize(width) {
    return [width, LiteGraph.NODE_WIDGET_HEIGHT];
  }

  serializeValue(node) {
    const value = { ...this.value };
    if (!showSeparateStrengths(node)) {
      delete value.strengthTwo;
    } else {
      value.strengthTwo = value.strengthTwo ?? value.strength ?? 1;
    }
    return value;
  }

  setLora(lora) {
    this.value.lora = lora;
    this.loraInfo = null;
    this.loraInfoPromise = null;
    this.getLoraInfo(true);
  }

  setLoraInfo(info) {
    this.loraInfo = info;
    this.loraInfoPromise = Promise.resolve(info);
  }

  draw(ctx, node, width, posY, height) {
    const nodeWidth = node.size?.[0] ?? width;
    this.last_y = posY;
    const currentShowModelAndClip = showSeparateStrengths(node);
    if (this.showModelAndClip !== currentShowModelAndClip) {
      const oldShowModelAndClip = this.showModelAndClip;
      this.showModelAndClip = currentShowModelAndClip;
      if (this.showModelAndClip) {
        if (oldShowModelAndClip != null) {
          this.value.strengthTwo = this.value.strength ?? 1;
        }
      } else {
        this.value.strengthTwo = null;
      }
    }

    const margin = 10;
    const innerMargin = margin * 0.33;
    const midY = posY + height / 2;
    let posX = margin;

    ctx.save();
    drawRoundedRectangle(ctx, {
      pos: [posX, posY],
      size: [nodeWidth - margin * 2, height],
      borderRadius: height * 0.5,
    });
    this.hitAreas.toggle = drawTogglePart(ctx, { posX, posY, height, value: this.value.on });
    posX += this.hitAreas.toggle[2] + innerMargin;

    if (isLowQuality()) {
      ctx.restore();
      return;
    }

    if (!this.value.on) {
      ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
    }

    let rightX = nodeWidth - margin - innerMargin - innerMargin;
    const clipStrength = this.showModelAndClip ? this.value.strengthTwo ?? 1 : this.value.strength ?? 1;
    const clipParts = drawNumberWidgetPart(ctx, {
      posX: rightX,
      posY,
      height,
      value: clipStrength,
      direction: -1,
      textColor: this.strengthTextColor(clipStrength),
    });
    this.hitAreas.strengthTwoDec = this.showModelAndClip ? clipParts[0] : null;
    this.hitAreas.strengthTwoVal = this.showModelAndClip ? clipParts[1] : null;
    this.hitAreas.strengthTwoInc = this.showModelAndClip ? clipParts[2] : null;
    this.hitAreas.strengthTwoAny = this.showModelAndClip ? clipParts[3] : null;
    this.hitAreas.strengthDec = this.showModelAndClip ? null : clipParts[0];
    this.hitAreas.strengthVal = this.showModelAndClip ? null : clipParts[1];
    this.hitAreas.strengthInc = this.showModelAndClip ? null : clipParts[2];
    this.hitAreas.strengthAny = this.showModelAndClip ? null : clipParts[3];
    rightX = clipParts[0][0] - innerMargin;

    if (this.showModelAndClip) {
      rightX -= innerMargin;
      const modelStrength = this.value.strength ?? 1;
      const modelParts = drawNumberWidgetPart(ctx, {
        posX: rightX,
        posY,
        height,
        value: modelStrength,
        direction: -1,
        textColor: this.strengthTextColor(modelStrength),
      });
      this.hitAreas.strengthDec = modelParts[0];
      this.hitAreas.strengthVal = modelParts[1];
      this.hitAreas.strengthInc = modelParts[2];
      this.hitAreas.strengthAny = modelParts[3];
      rightX = modelParts[0][0] - innerMargin;
    }

    const infoSize = height * 0.66;
    const infoWidth = infoSize + innerMargin + innerMargin;
    if (this.value.lora) {
      rightX -= innerMargin;
      drawInfoIcon(ctx, rightX - infoSize, posY + (height - infoSize) / 2, infoSize, this.infoTreatment());
      this.hitAreas.info = [rightX - infoSize, posY, infoWidth, height];
      rightX = rightX - infoSize - innerMargin;
    } else {
      this.hitAreas.info = null;
    }

    const loraWidth = rightX - posX;
    this.hitAreas.lora = [posX, posY, loraWidth, height];
    ctx.fillStyle = HELTO.text;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(fitString(ctx, this.value.lora || "None", loraWidth), posX, midY);
    ctx.restore();
  }

  mouse(event, pos, node) {
    if (event.type === "pointerdown") {
      if (event.button === 2) {
        this.showMenu(event, node);
        return true;
      }
      if (inArea(pos, this.hitAreas.toggle)) {
        this.value.on = !this.value.on;
        node.setDirtyCanvas(true, true);
        return true;
      }
      if (inArea(pos, this.hitAreas.info)) {
        this.showLoraInfoDialog();
        return true;
      }
      if (inArea(pos, this.hitAreas.lora)) {
        showLoraChooser(event, node, (value) => this.setLora(value));
        return true;
      }
      if (this.handleNumberPointerDown(event, pos, node)) {
        return true;
      }
    }

    if (event.type === "pointermove" && this.activeStrengthKey) {
      const delta = event.deltaX ?? event.movementX ?? 0;
      if (delta) {
        this.haveMouseMovedStrength = true;
        this.value[this.activeStrengthKey] = (this.value[this.activeStrengthKey] ?? 1) + delta * 0.05;
        node.setDirtyCanvas(true, true);
      }
      return true;
    }

    if (event.type === "pointerup" && this.activeStrengthKey) {
      if (!this.haveMouseMovedStrength) {
        this.promptStrength(event, this.activeStrengthKey);
      }
      this.haveMouseMovedStrength = false;
      this.activeStrengthKey = null;
      return true;
    }

    return false;
  }

  handleNumberPointerDown(event, pos, node) {
    const specs = [
      ["strength", this.hitAreas.strengthDec, -1],
      ["strength", this.hitAreas.strengthInc, 1],
      ["strengthTwo", this.hitAreas.strengthTwoDec, -1],
      ["strengthTwo", this.hitAreas.strengthTwoInc, 1],
    ];
    for (const [key, area, direction] of specs) {
      if (inArea(pos, area)) {
        this.stepStrength(key, direction);
        node.setDirtyCanvas(true, true);
        return true;
      }
    }

    if (inArea(pos, this.hitAreas.strengthAny)) {
      this.activeStrengthKey = "strength";
      this.haveMouseMovedStrength = false;
      return true;
    }
    if (inArea(pos, this.hitAreas.strengthTwoAny)) {
      this.activeStrengthKey = "strengthTwo";
      this.haveMouseMovedStrength = false;
      return true;
    }
    return false;
  }

  stepStrength(key, direction) {
    const current = this.value[key] ?? 1;
    this.value[key] = Math.round((current + 0.05 * direction) * 100) / 100;
  }

  promptStrength(event, key) {
    app.canvas.prompt(
      "Value",
      this.value[key] ?? 1,
      (value) => {
        const parsed = Number(value);
        if (!Number.isNaN(parsed)) {
          this.value[key] = parsed;
        }
      },
      event,
    );
  }

  strengthTextColor(value) {
    if (this.loraInfo?.strengthMax != null && value > this.loraInfo.strengthMax) {
      return HELTO.danger;
    }
    if (this.loraInfo?.strengthMin != null && value < this.loraInfo.strengthMin) {
      return HELTO.danger;
    }
    return undefined;
  }

  infoTreatment() {
    if (this.loraInfo?.raw?.civitai) {
      return "FILLED";
    }
    if (this.loraInfo?.hasInfoFile) {
      return "OUTLINED";
    }
    return "GRAYED";
  }

  async getLoraInfo(force = false) {
    if (!this.value.lora || this.value.lora === "None") {
      this.loraInfo = null;
      return null;
    }
    if (!this.loraInfoPromise || force) {
      this.loraInfoPromise = fetchLoraInfo(this.value.lora, { refresh: force, light: true })
        .then((info) => (this.loraInfo = info))
        .catch(() => null);
    }
    return this.loraInfoPromise;
  }

  async showLoraInfoDialog() {
    if (!this.value.lora || this.value.lora === "None") {
      return;
    }
    await showLoraInfoDialog(this.value.lora, this);
  }

  showMenu(event, node) {
    new LiteGraph.ContextMenu(rowMenuItems(node, this), {
      event,
      title: "LoRA",
      className: "dark",
    });
  }
}

function rowMenuItems(node, row) {
  const rows = rowWidgets(node);
  const index = rows.indexOf(row);
  return [
    { content: "Show Info", callback: () => row.showLoraInfoDialog() },
    null,
    {
      content: row.value.on ? "Toggle Off" : "Toggle On",
      callback: () => {
        row.value.on = !row.value.on;
        node.setDirtyCanvas(true, true);
      },
    },
    {
      content: "Move Up",
      disabled: index <= 0,
      callback: () => moveRow(node, row, -1),
    },
    {
      content: "Move Down",
      disabled: index < 0 || index >= rows.length - 1,
      callback: () => moveRow(node, row, 1),
    },
    {
      content: "Remove",
      callback: () => removeRow(node, row),
    },
  ];
}

function allLorasState(node) {
  const rows = rowWidgets(node);
  if (!rows.length) {
    return false;
  }
  const allOn = rows.every((row) => row.value.on === true);
  const allOff = rows.every((row) => row.value.on === false);
  if (!allOn && !allOff) {
    return null;
  }
  return allOn;
}

function toggleAll(node) {
  const rows = rowWidgets(node);
  const toggledTo = !allLorasState(node);
  for (const row of rows) {
    row.value.on = toggledTo;
  }
  node.setDirtyCanvas(true, true);
}

function removeDynamicWidgets(node) {
  node.widgets = (node.widgets || []).filter((widget) => !dynamicWidgets(node).includes(widget));
}

function addHeader(node) {
  const header = new LoraHeaderWidget();
  node.addCustomWidget(header);
  return header;
}

function addControls(node) {
  removeArrayItem(node.widgets, node.widgets.find((widget) => widget._aioLoraAddButton === true));
  const button = node.addWidget("button", ADD_BUTTON_LABEL, null, async (...args) => {
    const event = args.find((arg) => arg instanceof Event) || window.event;
    await showLoraChooser(event, node, (value) => addRow(node, value));
  });
  button._aioLoraAddButton = true;
  button.tooltip = ADD_LORA_TOOLTIP;
  button.options ||= {};
  button.options.tooltip = ADD_LORA_TOOLTIP;
}

function applyLoraNodeSize(node, mode = "restore", savedSize = null) {
  const currentSize = node.size || node.computeSize();
  const nextSize = [Math.max(Number(currentSize[0]) || 0, MIN_NODE_WIDTH), Number(currentSize[1]) || 0];

  if (Array.isArray(savedSize) && Number.isFinite(Number(savedSize[0]))) {
    nextSize[0] = Math.max(Number(savedSize[0]), MIN_NODE_WIDTH);
  }
  if (mode === "interactive") {
    nextSize[1] = Math.max(nextSize[1], node.computeSize()[1]);
  } else if (Array.isArray(savedSize) && Number.isFinite(Number(savedSize[1]))) {
    nextSize[1] = Number(savedSize[1]);
  }

  if (typeof node.setSize === "function") {
    node.setSize(nextSize);
  } else {
    node.size = nextSize;
  }
}

function scheduleLoraNodeSizeRestore(node, savedSize) {
  if (!Array.isArray(savedSize)) {
    return;
  }
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      applyLoraNodeSize(node, "restore", savedSize);
      node.setDirtyCanvas?.(true, true);
    });
  });
}

function addRow(node, lora = null, value = null, { resize = true, dirty = true } = {}) {
  const widget = new LoraRowWidget(nextRowName(node), value);
  if (lora) {
    widget.setLora(lora);
  }
  const buttonIndex = node.widgets.findIndex((item) => item._aioLoraAddButton === true);
  if (buttonIndex >= 0) {
    node.widgets.splice(buttonIndex, 0, widget);
  } else {
    node.addCustomWidget(widget);
  }
  applyLoraNodeSize(node, resize ? "interactive" : "restore");
  if (dirty) {
    node.setDirtyCanvas(true, true);
  }
  return widget;
}

function moveRow(node, row, direction) {
  const rows = rowWidgets(node);
  const rowIndex = rows.indexOf(row);
  const sibling = rows[rowIndex + direction];
  if (!sibling) {
    return;
  }
  moveArrayItem(node.widgets, row, node.widgets.indexOf(sibling));
  node.setDirtyCanvas(true, true);
}

function removeRow(node, row) {
  removeArrayItem(node.widgets, row);
  node.setDirtyCanvas(true, true);
}

function restoreRows(node, info) {
  const values = (info?.widgets_values || []).filter((value) => value && typeof value.lora === "string");
  removeDynamicWidgets(node);
  addHeader(node);
  for (const value of values) {
    addRow(node, null, value, { resize: false, dirty: false });
  }
  addControls(node);
  applyLoraNodeSize(node, "restore", info?.size);
  scheduleLoraNodeSizeRestore(node, info?.size);
}

function ensureLoraUi(node) {
  if (!isAioLoraNode(node)) {
    return;
  }
  node.serialize_widgets = true;
  const hasHeader = node.widgets?.some((widget) => widget.name === HEADER_NAME);
  const hasButton = node.widgets?.some((widget) => widget._aioLoraAddButton === true);
  if (!hasHeader) {
    addHeader(node);
  }
  if (!hasButton) {
    addControls(node);
  }
  applyLoraNodeSize(node, "restore");
  node.setDirtyCanvas?.(true, true);
}

function patchLoraNodeType(nodeType) {
  if (nodeType.prototype.__aioLoraConfigurationPatched) {
    return;
  }
  nodeType.prototype.__aioLoraConfigurationPatched = true;

  const originalCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    originalCreated?.apply(this, arguments);
    this.serialize_widgets = true;
    removeDynamicWidgets(this);
    addHeader(this);
    addControls(this);
    applyLoraNodeSize(this, "interactive");
    this.setDirtyCanvas(true, true);
  };

  const originalConfigure = nodeType.prototype.configure;
  nodeType.prototype.configure = function (info) {
    originalConfigure?.apply(this, arguments);
    this.serialize_widgets = true;
    restoreRows(this, info);
  };

  const originalRefreshCombo = nodeType.prototype.refreshComboInNode;
  nodeType.prototype.refreshComboInNode = function () {
    loraListPromise = null;
    return originalRefreshCombo?.apply(this, arguments);
  };

  const originalMenu = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
    originalMenu?.apply(this, arguments);
    options.push({
      content: "Toggle All LoRAs",
      callback: () => toggleAll(this),
    });
    options.push({
      content: "Refresh LoRA List",
      callback: () => getLoras(true),
    });
  };

  const originalGetSlot = nodeType.prototype.getSlotInPosition;
  nodeType.prototype.getSlotInPosition = function (canvasX, canvasY) {
    const slot = originalGetSlot?.apply(this, arguments);
    if (slot) {
      return slot;
    }
    const localY = canvasY - this.pos[1];
    for (const widget of this.widgets || []) {
      if (
        String(widget.name).startsWith(ROW_PREFIX) &&
        localY >= widget.last_y &&
        localY <= widget.last_y + LiteGraph.NODE_WIDGET_HEIGHT
      ) {
        return { widget, output: { type: "AIO LORA ROW" } };
      }
    }
    return undefined;
  };

  const originalSlotMenu = nodeType.prototype.getSlotMenuOptions;
  nodeType.prototype.getSlotMenuOptions = function (slot) {
    if (String(slot?.widget?.name || "").startsWith(ROW_PREFIX)) {
      return rowMenuItems(this, slot.widget);
    }
    return originalSlotMenu?.apply(this, arguments);
  };
}

function patchAioThemeNodeType(nodeType) {
  if (nodeType.prototype.__aioHeltoNodeThemePatched) {
    return;
  }
  nodeType.prototype.__aioHeltoNodeThemePatched = true;

  const originalCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = originalCreated?.apply(this, arguments);
    applyAioNodeTheme(this);
    return result;
  };

  const originalConfigure = nodeType.prototype.configure;
  nodeType.prototype.configure = function () {
    const result = originalConfigure?.apply(this, arguments);
    applyAioNodeTheme(this);
    return result;
  };
}

function patchRemovedSettingsWidgets(nodeType, { marker, normalize }) {
  if (nodeType.prototype[marker]) {
    return;
  }
  nodeType.prototype[marker] = true;

  const originalConfigure = nodeType.prototype.configure;
  nodeType.prototype.configure = function (info) {
    const normalizedInfo = info && Array.isArray(info.widgets_values)
      ? {
          ...info,
          widgets_values: normalize(info.widgets_values),
        }
      : info;
    const args = arguments.length ? [normalizedInfo, ...Array.prototype.slice.call(arguments, 1)] : arguments;
    return originalConfigure?.apply(this, args);
  };
}

function patchAioGenerateNodeType(nodeType) {
  if (nodeType.prototype.__aioGenerateConfigurePatched) {
    return;
  }
  nodeType.prototype.__aioGenerateConfigurePatched = true;

  const originalConfigure = nodeType.prototype.configure;
  nodeType.prototype.configure = function () {
    const normalizedInfo = configureInfoWithNormalizedGenerateWidgets(this, arguments[0]);
    const args = arguments.length ? [normalizedInfo, ...Array.prototype.slice.call(arguments, 1)] : arguments;
    removeAioGenerateSeedButton(this);
    try {
      return originalConfigure?.apply(this, args);
    } finally {
      ensureAioGenerateSizingUi(this);
      ensureAioGenerateSeedButton(this);
      ensureAioGenerateRuntimePhaseUi(this);
    }
  };
}

scheduleAioSeedQueuePatch();

app.registerExtension({
  name: "aio.image.generate",
  setup() {
    installAioWidgetThemeBridge();
    scheduleAioSeedQueuePatch("setup");
    installAioGenerateProgressTextCleanup();
    installAioGenerateRuntimePhaseBridge();
    requestAnimationFrame(() => {
      for (const node of app.graph?._nodes || []) {
        applyAioNodeTheme(node);
        ensureLoraUi(node);
        ensureAioGenerateSizingUi(node);
        ensureAioGenerateSeedButton(node);
        ensureAioGenerateRuntimePhaseUi(node);
      }
    });
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (isAioLoraNodeData(nodeData)) {
      patchLoraNodeType(nodeType);
    }
    if (isAioGenerateNodeData(nodeData)) {
      patchAioGenerateNodeType(nodeType);
    }
    if (nodeData?.name === FLUX_SETTINGS_NODE_NAME) {
      patchRemovedSettingsWidgets(nodeType, {
        marker: "__aioFluxSettingsWidgetMigrationPatched",
        normalize: normalizeFluxSettingsWidgetValues,
      });
    }
    if (nodeData?.name === Z_IMAGE_SETTINGS_NODE_NAME) {
      patchRemovedSettingsWidgets(nodeType, {
        marker: "__aioZImageSettingsWidgetMigrationPatched",
        normalize: normalizeZImageSettingsWidgetValues,
      });
    }
    if (isAioThemedNodeData(nodeData)) {
      patchAioThemeNodeType(nodeType);
    }
  },
  nodeCreated(node) {
    applyAioNodeTheme(node);
    ensureLoraUi(node);
    ensureAioGenerateSizingUi(node);
    ensureAioGenerateSeedButton(node);
    ensureAioGenerateRuntimePhaseUi(node);
  },
  loadedGraphNode(node) {
    applyAioNodeTheme(node);
    ensureLoraUi(node);
    ensureAioGenerateSizingUi(node);
    ensureAioGenerateSeedButton(node);
    ensureAioGenerateRuntimePhaseUi(node);
  },
});
