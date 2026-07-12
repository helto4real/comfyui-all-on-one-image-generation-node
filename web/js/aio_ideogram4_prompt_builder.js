import { app } from "/scripts/app.js";
import {
  assertSupportedPrivacyPayload,
  decryptState,
  decryptValue,
  encryptStateSync,
  encryptValueSync,
  getSharedPrivacyUi,
  isAnyAioPrivacyPayload,
  isEncryptedPrivacyPayload,
  isLegacyPrivacyPayload,
  parsePrivacyPayload,
  privacyFetchHeaders,
} from "./aio_privacy.js";
import {
  appendPrivacyRecoveryMenuOption,
  registerAioPrivacyRecoveryDescriptors,
} from "./aio_privacy_recovery.js";
import { applyHeltoNodeTheme, ensureHeltoTokens, HELTO } from "./aio_helto_theme.js";

const NODE_NAME = "AIOIdeogram4PromptBuilder";
const MIN_WIDTH = 620;
const EDITOR_HEIGHT = 520;
const EDITOR_MIN_HEIGHT = EDITOR_HEIGHT;
const EDITOR_NODE_MARGIN = 10;
const MIN_NODE_WIDTH = MIN_WIDTH + EDITOR_NODE_MARGIN * 2;
const EDITOR_INITIAL_NODE_HEIGHT = 760 + EDITOR_NODE_MARGIN * 2;
// Display-only bbox palette default kept for prompt data compatibility.
const DEFAULT_COLOR = "#8ca8ff";
const DEFAULT_COLOR_UPPER = DEFAULT_COLOR.toUpperCase();
// Selection highlight on the canvas = Helto gold accent (selection/active).
const ACTIVE_COLOR = HELTO.accent;
const LIBRARY_ROUTE = "/aio_image_generate/ideogram4_prompt_library";
const WORKFLOW_STATE_KEY = "aio_ideogram4_prompt_builder";
const STATE_PROPERTY = "aio_ideogram4_prompt_builder_state";
const LIBRARY_ITEM_PROPERTY = "aio_ideogram4_prompt_library_item_id";
const PRIVACY_WIDGET_NAME = "privacy_mode";
const SENSITIVE_WIDGET_NAMES = [
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
const STATE_WIDGET_NAMES = [
  "max side",
  "aspect ratio",
  "multiple value",
  "privacy_mode",
  "high_level_description",
  "background",
  "style",
  "photo",
  "art_style",
  "aesthetics",
  "lighting",
  "medium",
  "import_mode",
  "output_format",
  "coord_mode",
  "bbox_order",
  "style_palette_data",
  "elements_data",
  "bg_brightness",
  "import_json",
];

function editorWidgetHeight(editorHeight = EDITOR_HEIGHT) {
  return Math.ceil(editorHeight + EDITOR_NODE_MARGIN * 2);
}

const ICONS = {
  library: `<svg viewBox="0 0 24 24"><path d="M5 5h6v14H5z"/><path d="M13 5h6v14h-6z"/><path d="M7 8h2M15 8h2M7 12h2M15 12h2"/></svg>`,
  save: `<svg viewBox="0 0 24 24"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8"/><path d="M7 3v5h8"/></svg>`,
  load: `<svg viewBox="0 0 24 24"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>`,
  copy: `<svg viewBox="0 0 24 24"><path d="M8 8h11v11H8z"/><path d="M5 16H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h11a1 1 0 0 1 1 1v1"/></svg>`,
  paste: `<svg viewBox="0 0 24 24"><rect x="8" y="3" width="8" height="4" rx="1"/><path d="M16 5h2a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h2"/></svg>`,
  clear: `<svg viewBox="0 0 24 24"><path d="M5 16 14 7l4 4-7 7H8z"/><path d="m5 16 3 3"/><path d="M12 20h8"/></svg>`,
  text: `<svg viewBox="0 0 24 24"><path d="M4 6h11"/><path d="M9.5 6v12"/><path d="M18 13v6"/><path d="M15 16h6"/></svg>`,
  obj: `<svg viewBox="0 0 24 24"><rect x="3" y="5" width="11" height="11" rx="1"/><path d="M18 13v6"/><path d="M15 16h6"/></svg>`,
  delete: `<svg viewBox="0 0 24 24"><path d="M6 7h12M10 7V5h4v2M9 10v7M15 10v7M8 7l1 12h6l1-12"/></svg>`,
  edit: `<svg viewBox="0 0 24 24"><path d="m4 20 4-1 11-11-3-3L5 16z"/><path d="m14 6 3 3"/></svg>`,
  close: `<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>`,
  search: `<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m16 16 4 4"/></svg>`,
};

function widgetByName(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function stopEvent(event) {
  event.stopPropagation();
}

function parseJsonList(value) {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function parseElementsPayload(value) {
  if (!value) return { elements: [], widgets: {} };
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed)) return { elements: parsed, widgets: {} };
    if (parsed && typeof parsed === "object") {
      return {
        elements: Array.isArray(parsed.elements) ? parsed.elements : Array.isArray(parsed.boxes) ? parsed.boxes : [],
        widgets: parsed.widgets && typeof parsed.widgets === "object" && !Array.isArray(parsed.widgets) ? parsed.widgets : {},
      };
    }
  } catch {
    return { elements: [], widgets: {} };
  }
  return { elements: [], widgets: {} };
}

function parseStatePayload(value) {
  if (!value || isAnyAioPrivacyPayload(value)) return null;
  if (typeof value === "object") return value;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function parseWorkflowStatePayload(value) {
  const state = parseStatePayload(value);
  if (state) return state;
  if (value && typeof value === "object" && Array.isArray(value.boxes)) {
    return {
      version: 1,
      elements: value.boxes,
      style_palette: Array.isArray(value.palette) ? value.palette : [],
      output_format: value.outputFormat,
      coord_mode: value.coordMode,
      bbox_order: value.bboxOrder,
      widgets: {
        import_mode: value.importMode,
        output_format: value.outputFormat,
        coord_mode: value.coordMode,
        bbox_order: value.bboxOrder,
      },
    };
  }
  return null;
}

function cloneJson(value, fallback) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return fallback;
  }
}

async function fetchLibraryJson(url, options = {}, retry = true) {
  const privacy = await getSharedPrivacyUi();
  privacy?.ensureStoredPrivacyTokenCookie?.();
  const requestOptions = {
    ...options,
    headers: privacyFetchHeaders(options.headers || {}),
  };
  const response = await fetch(url, requestOptions);
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text || response.statusText || `HTTP ${response.status}`);
  }
  if (!response.ok || data.ok === false || data.error) {
    const error = new Error(data.error || response.statusText || `HTTP ${response.status}`);
    const unlockRequired = Boolean(
      privacy?.isPrivacyUnlockRequiredError?.(error) ||
        privacy?.isPrivacyLockedError?.(error)
    );
    if (retry && unlockRequired) {
      const unlocked = await privacy.showPrivacyKeystoreDialog?.("auto");
      privacy.ensureStoredPrivacyTokenCookie?.();
      if (unlocked) return fetchLibraryJson(url, options, false);
    }
    throw error;
  }
  return data;
}

function iconButton(label, iconName, title) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "aio-ideo-icon-btn";
  button.title = title || label;
  button.setAttribute("aria-label", title || label);
  button.innerHTML = ICONS[iconName] || "";
  return button;
}

function textButton(label, iconName = "") {
  const button = document.createElement("button");
  button.type = "button";
  if (iconName) {
    button.innerHTML = `${ICONS[iconName] || ""}<span>${label}</span>`;
  } else {
    button.textContent = label;
  }
  return button;
}

function normalizedColor(value) {
  const text = String(value || "").trim();
  if (/^#[0-9a-fA-F]{6}$/.test(text)) return text.toUpperCase();
  return DEFAULT_COLOR_UPPER;
}

function palette(colors) {
  if (!colors) return [];
  const values = Array.isArray(colors) ? colors : Object.values(colors);
  return values.filter(Boolean).map((color) => String(color).toUpperCase());
}

function promptPalette(colors) {
  const values = palette(colors);
  return values.length === 1 && values[0] === DEFAULT_COLOR_UPPER ? [] : values;
}

function widgetDomTextValue(widget) {
  const elements = [];
  for (const candidate of [widget?.inputEl, widget?.element, widget?.inputElement, widget?.textarea, widget?.textElement]) {
    if (candidate instanceof HTMLElement) elements.push(candidate);
  }
  for (const candidate of [...elements]) {
    elements.push(...(candidate.querySelectorAll?.("textarea,input,[contenteditable='true']") || []));
  }
  for (const element of [...new Set(elements)]) {
    const tagName = String(element.tagName || "").toLowerCase();
    if (tagName === "textarea") return element.value;
    if (tagName === "input") {
      const type = String(element.type || "text").toLowerCase();
      if (["", "text", "search", "url", "email", "password", "number"].includes(type)) return element.value;
    }
    if (element.isContentEditable) return element.textContent || "";
  }
  return null;
}

function normBBox(box) {
  const clamp = (value) => Math.max(0, Math.min(1000, Math.round(value * 1000)));
  let ymin = clamp(box.y || 0);
  let xmin = clamp(box.x || 0);
  let ymax = clamp((box.y || 0) + (box.h || 0));
  let xmax = clamp((box.x || 0) + (box.w || 0));
  if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
  if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
  return [ymin, xmin, ymax, xmax];
}

function captionBBox(box, { coordMode = "normalized", bboxOrder = "yx", width = 1000, height = 1000 } = {}) {
  const absolute = coordMode === "absolute";
  const xy = bboxOrder === "xy";
  const sx = absolute ? Math.max(1, Number(width) || 1) : 1000;
  const sy = absolute ? Math.max(1, Number(height) || 1) : 1000;
  const clampX = (value) => Math.max(0, Math.min(sx, Math.round(value * sx)));
  const clampY = (value) => Math.max(0, Math.min(sy, Math.round(value * sy)));
  let ymin = clampY(box.y || 0);
  let xmin = clampX(box.x || 0);
  let ymax = clampY((box.y || 0) + (box.h || 0));
  let xmax = clampX((box.x || 0) + (box.w || 0));
  if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
  if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
  return xy ? [xmin, ymin, xmax, ymax] : [ymin, xmin, ymax, xmax];
}

function pyJson(value, level = 0) {
  const pad = "    ".repeat(level + 1);
  const end = "    ".repeat(level);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    if (value.every((item) => typeof item !== "object" || item === null)) {
      return "[" + value.map((item) => pyJson(item, level)).join(", ") + "]";
    }
    return "[\n" + value.map((item) => pad + pyJson(item, level + 1)).join(",\n") + "\n" + end + "]";
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) return "{}";
    return "{\n" + entries.map(([key, val]) => pad + JSON.stringify(key) + ": " + pyJson(val, level + 1)).join(",\n") + "\n" + end + "}";
  }
  return JSON.stringify(value);
}

function captionToBoxes(caption, { coordMode = "normalized", bboxOrder = "yx", width = 1000, height = 1000 } = {}) {
  const cd = caption?.compositional_deconstruction || {};
  const boxes = [];
  const absolute = coordMode === "absolute";
  const xy = bboxOrder === "xy";
  const sx = absolute ? Math.max(1, Number(width) || 1) : 1000;
  const sy = absolute ? Math.max(1, Number(height) || 1) : 1000;
  for (const element of cd.elements || []) {
    if (!element || typeof element !== "object") continue;
    const box = {
      type: element.type === "text" ? "text" : "obj",
      text: element.text || "",
      desc: element.desc || "",
      palette: [...(element.color_palette || [])],
    };
    const bbox = element.bbox;
    if (Array.isArray(bbox) && bbox.length === 4) {
      const [a, b, c, d] = bbox;
      const [xmin, ymin, xmax, ymax] = xy ? [a, b, c, d] : [b, a, d, c];
      box.x = xmin / sx;
      box.y = ymin / sy;
      box.w = (xmax - xmin) / sx;
      box.h = (ymax - ymin) / sy;
    } else {
      box.x = 0.03;
      box.y = 0.03;
      box.w = 0.22;
      box.h = 0.14;
      box.nobbox = true;
    }
    boxes.push(box);
  }
  return boxes;
}

function buildCaption(node, boxes, stylePalette) {
  const value = (name, fallback = "") => {
    const widget = widgetByName(node, name);
    const domValue = widgetDomTextValue(widget);
    return domValue ?? widget?.value ?? fallback;
  };
  const kind = String(value("style", "none"));
  const coordMode = value("coord_mode", "normalized") === "absolute" ? "absolute" : "normalized";
  const bboxOrder = value("bbox_order", "yx") === "xy" ? "xy" : "yx";
  const [width, height] = resolveDims(node);
  const caption = {};
  const highLevel = String(value("high_level_description", ""));
  if (highLevel.trim()) caption.high_level_description = highLevel;

  if (kind !== "none") {
    const styleDescription = {
      aesthetics: String(value("aesthetics", "")),
      lighting: String(value("lighting", "")),
    };
    if (kind === "photo") {
      styleDescription.photo = String(value("photo", ""));
      styleDescription.medium = String(value("medium", ""));
    } else {
      styleDescription.medium = String(value("medium", ""));
      styleDescription.art_style = String(value("art_style", ""));
    }
    const styleColors = palette(stylePalette);
    if (styleColors.length) styleDescription.color_palette = styleColors;
    caption.style_description = styleDescription;
  }

  const elements = [];
  for (const box of boxes) {
    if (!box || typeof box !== "object") continue;
    const type = box.type === "text" ? "text" : "obj";
    const element = { type };
    if (!box.nobbox) element.bbox = captionBBox(box, { coordMode, bboxOrder, width, height });
    if (type === "text") element.text = box.text || "";
    element.desc = box.desc || "";
    const colors = promptPalette(box.palette).slice(0, 5);
    if (colors.length) element.color_palette = colors;
    elements.push(element);
  }

  caption.compositional_deconstruction = {
    background: String(value("background", "")),
    elements,
  };
  return caption;
}

function formatCaption(node, caption) {
  return widgetByName(node, "output_format")?.value === "pretty" ? pyJson(caption) : JSON.stringify(caption);
}

function resolveDims(node) {
  const maxSide = Math.max(1, Number(widgetByName(node, "max side")?.value || 1024));
  const ratio = String(widgetByName(node, "aspect ratio")?.value || "1:1").split(":");
  const rw = Math.max(1, Number(ratio[0] || 1));
  const rh = Math.max(1, Number(ratio[1] || 1));
  const multipleValue = String(widgetByName(node, "multiple value")?.value || "none");
  const multiple = multipleValue === "none" ? null : Number(multipleValue);
  const roundValue = (value) => {
    if (!multiple) return Math.max(1, Math.round(value));
    return Math.max(multiple, Math.round(value / multiple) * multiple);
  };
  const side = roundValue(maxSide);
  if (rw >= rh) return [side, roundValue((side * rh) / rw)];
  return [roundValue((side * rw) / rh), side];
}

function installStyles() {
  ensureHeltoTokens();
  if (document.getElementById("aio-ideo-style")) return;
  const style = document.createElement("style");
  style.id = "aio-ideo-style";
  style.textContent = `
    /* ---- Editor shell (Helto panel) ---- */
    .aio-ideo-wrap{position:relative;box-sizing:border-box;display:flex;flex-direction:column;gap:8px;width:var(--aio-ideo-editor-width, 100%);min-width:${MIN_WIDTH}px;height:var(--aio-ideo-editor-height, ${EDITOR_HEIGHT}px);padding:9px;color:var(--helto-text);font:var(--helto-font-size)/var(--helto-line) var(--helto-font-sans);-webkit-font-smoothing:antialiased;background:var(--helto-surface);border:1px solid var(--helto-border);border-radius:var(--helto-radius);box-shadow:var(--helto-shadow);overflow:hidden}
    .aio-ideo-wrap *,.aio-ideo-wrap *::before,.aio-ideo-wrap *::after{box-sizing:border-box}
    .aio-ideo-wrap ::-webkit-scrollbar{width:6px;height:6px}
    .aio-ideo-wrap ::-webkit-scrollbar-track{background:transparent}
    .aio-ideo-wrap ::-webkit-scrollbar-thumb{background:var(--helto-border-strong);border-radius:3px}
    .aio-ideo-wrap ::-webkit-scrollbar-thumb:hover{background:var(--helto-text-faint)}

    /* ---- Toolbar (gradient strip + inset hairline) ---- */
    .aio-ideo-toolbar{display:flex;align-items:center;gap:6px;flex:0 0 auto;min-height:34px;padding:5px;border-radius:var(--helto-radius);background:linear-gradient(180deg,var(--helto-surface-2),var(--helto-surface));box-shadow:inset 0 0 0 1px var(--helto-border)}

    /* ---- Buttons: raised gradient, hover one step lighter (covers toolbar,
       rows, list and standalone side-panel buttons so none fall back to the
       native gray control). ---- */
    .aio-ideo-wrap button{min-width:28px;height:24px;display:inline-flex;align-items:center;justify-content:center;gap:6px;background:linear-gradient(180deg,var(--helto-surface-3),var(--helto-surface-2));color:var(--helto-text);border:1px solid var(--helto-border-strong);border-radius:var(--helto-radius-sm);padding:0 8px;font:inherit;white-space:nowrap;cursor:pointer;transition:background var(--helto-transition),border-color var(--helto-transition),color var(--helto-transition),box-shadow var(--helto-transition),transform .03s ease}
    .aio-ideo-wrap button:hover:not(:disabled){background:linear-gradient(180deg,var(--helto-surface-hover),var(--helto-surface-3));border-color:var(--helto-border-hover);color:var(--helto-text)}
    .aio-ideo-wrap button:active:not(:disabled){transform:translateY(1px)}
    .aio-ideo-wrap button:disabled{opacity:.4;cursor:not-allowed}
    .aio-ideo-wrap button:focus-visible{outline:none;border-color:var(--helto-focus);box-shadow:var(--helto-focus-ring)}
    /* Constructive "+ add" actions = GOLD (affirmative) gradient. */
    .aio-ideo-wrap button.aio-ideo-accent{border-color:var(--helto-accent-border);background:linear-gradient(180deg,#4f3a2a,#3d2d20);color:var(--helto-accent-strong)}
    .aio-ideo-wrap button.aio-ideo-accent:hover:not(:disabled){background:linear-gradient(180deg,#5d4531,#493626);color:var(--helto-accent-strong)}
    /* Destructive actions = red gradient. */
    .aio-ideo-wrap button.aio-ideo-danger{border-color:var(--helto-danger-border);background:linear-gradient(180deg,#5c2c3d,#482331);color:var(--helto-danger)}
    .aio-ideo-wrap button.aio-ideo-danger:hover:not(:disabled){border-color:var(--helto-danger);background:linear-gradient(180deg,#6e3549,#5a2a3c);color:var(--helto-danger)}
    /* Compact list reorder button. */
    .aio-ideo-item button{min-width:22px;height:22px;padding:0 5px}
    .aio-ideo-toolbar .aio-ideo-icon-btn{width:28px;min-width:28px;height:28px;padding:0}
    .aio-ideo-icon-btn svg,.aio-ideo-library svg{width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
    /* Linked-to-library = GOLD active state. */
    .aio-ideo-toolbar .aio-ideo-library-linked{border-color:var(--helto-accent-border);background:linear-gradient(180deg,#4f3a2a,#3d2d20);color:var(--helto-accent-strong);box-shadow:inset 0 0 0 1px var(--helto-accent-border)}
    .aio-ideo-toolbar .aio-ideo-library-linked:hover{background:linear-gradient(180deg,#5d4531,#493626);color:var(--helto-accent-strong)}
    .aio-ideo-count{margin-left:auto;color:var(--helto-text-dim);white-space:nowrap;font-variant-numeric:tabular-nums}
    .aio-ideo-toolbar label{color:var(--helto-text-dim)}
    .aio-ideo-toolbar input[type="checkbox"]{accent-color:var(--helto-accent)}

    /* ---- Main split + canvas inset ---- */
    .aio-ideo-main{display:grid;grid-template-columns:minmax(260px,1fr) 340px;gap:8px;width:100%;min-width:0;min-height:0;flex:1 1 auto}
    .aio-ideo-canvasBox{position:relative;display:flex;align-items:center;justify-content:center;min-height:0;background:var(--helto-bg);border:1px solid var(--helto-border);border-radius:var(--helto-radius);box-shadow:inset 0 0 0 1px var(--helto-border);overflow:hidden}
    .aio-ideo-canvas{background:var(--helto-bg);outline:none;max-width:100%;max-height:100%;border-radius:4px}
    .aio-ideo-side{display:flex;flex-direction:column;gap:6px;min-height:0;overflow:hidden}
    .aio-ideo-list{flex:1 1 auto;min-height:92px;overflow:auto;border:1px solid var(--helto-border);border-radius:var(--helto-radius);background:var(--helto-bg);padding:4px}
    .aio-ideo-item{display:grid;grid-template-columns:18px 1fr auto;gap:6px;align-items:center;padding:4px 6px;border-radius:var(--helto-radius-sm);cursor:pointer;border:1px solid transparent;transition:background var(--helto-transition),border-color var(--helto-transition)}
    .aio-ideo-item:hover{background:var(--helto-surface-2)}
    /* Selected region = GOLD tint + accent border. */
    .aio-ideo-item.active{background:var(--helto-accent-bg);border-color:var(--helto-accent-border);color:var(--helto-accent-strong)}
    .aio-ideo-item span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

    /* ---- Side form fields ---- */
    .aio-ideo-row{display:flex;gap:5px;align-items:center;flex:0 0 auto}
    .aio-ideo-row input,.aio-ideo-row textarea,.aio-ideo-row select{width:100%;box-sizing:border-box;background:var(--helto-surface-2);color:var(--helto-text);border:1px solid var(--helto-border-strong);border-radius:var(--helto-radius-sm);padding:0 8px;height:26px;font:inherit;transition:border-color var(--helto-transition),box-shadow var(--helto-transition)}
    .aio-ideo-row textarea{min-height:56px;height:auto;padding:7px 9px;line-height:var(--helto-line);resize:vertical}
    .aio-ideo-row input:focus,.aio-ideo-row textarea:focus,.aio-ideo-row select:focus{outline:none;border-color:var(--helto-focus);box-shadow:var(--helto-focus-ring)}
    .aio-ideo-row input::placeholder,.aio-ideo-row textarea::placeholder{color:var(--helto-text-faint)}
    .aio-ideo-row input[type="number"]{font-variant-numeric:tabular-nums}
    .aio-ideo-swatches{display:flex;gap:4px;flex-wrap:wrap;align-items:center}
    .aio-ideo-swatch{width:18px;height:18px;padding:0;border:1px solid var(--helto-border-strong);border-radius:var(--helto-radius-sm);cursor:pointer;background:var(--helto-surface-2)}
    input.aio-ideo-swatch{height:18px}
    .aio-ideo-help{color:var(--helto-text-faint);line-height:1.3}

    /* ---- Privacy / hide-mode (text masked while concealed) ---- */
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) .aio-ideo-count,
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) .aio-ideo-item span,
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) input[type="text"],
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) input:not([type]),
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) input[type="number"],
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) textarea {
      color: transparent !important;
      -webkit-text-fill-color: transparent !important;
      text-shadow: none !important;
      caret-color: transparent !important;
    }
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) input::placeholder,
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) textarea::placeholder {
      color: transparent !important;
    }
    .aio-ideo-private-field {
      background: var(--helto-surface-2) !important;
      border-color: var(--helto-surface-2) !important;
      color: transparent !important;
      -webkit-text-fill-color: transparent !important;
      text-shadow: none !important;
      caret-color: transparent !important;
    }
    .aio-ideo-private-field::placeholder {
      color: transparent !important;
    }
    /* Privacy status banner (warm, communicates concealed state). */
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) input[type="text"],
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) input:not([type]),
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) input[type="number"],
    .aio-ideo-wrap.is-private:not(.is-privacy-revealed) textarea {
      background: var(--helto-surface-2) !important;
      border-color: var(--helto-surface-2) !important;
    }
    .aio-ideo-privacy-status{position:absolute;left:9px;right:9px;bottom:9px;z-index:4;padding:7px 10px;border:1px solid var(--helto-accent-border);border-radius:var(--helto-radius);background:var(--helto-accent-bg);color:var(--helto-accent-strong);box-shadow:var(--helto-shadow-pop)}

    /* ---- Prompt Library (modal/overlay) ---- */
    .aio-ideo-library{position:fixed;inset:0;z-index:10020;display:flex;align-items:center;justify-content:center;padding:28px;box-sizing:border-box;background:color-mix(in srgb,var(--helto-bg) 72%,transparent);backdrop-filter:blur(4px);color:var(--helto-text);font:var(--helto-font-size)/var(--helto-line) var(--helto-font-sans)}
    .aio-ideo-library *,.aio-ideo-library *::before,.aio-ideo-library *::after{box-sizing:border-box}
    .aio-ideo-library-panel{width:min(980px,calc(100vw - 56px));height:min(700px,calc(100vh - 56px));min-height:480px;display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;border:1px solid var(--helto-border-strong);border-radius:var(--helto-radius-lg);background:linear-gradient(135deg,var(--helto-surface-2),var(--helto-bg));box-shadow:var(--helto-shadow-pop);backdrop-filter:blur(15px);overflow:hidden;animation:aio-ideo-rise .2s var(--helto-ease-spring)}
    @keyframes aio-ideo-rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
    .aio-ideo-library-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;border-bottom:1px solid var(--helto-border)}
    .aio-ideo-library-title{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:16px;font-weight:700;letter-spacing:.02em;color:var(--helto-text)}
    .aio-ideo-library-controls{display:grid;grid-template-columns:minmax(220px,1fr) 160px auto;gap:10px;align-items:center;padding:12px 16px}
    .aio-ideo-library-search{height:32px;display:grid;grid-template-columns:20px minmax(0,1fr);align-items:center;gap:5px;padding:0 8px;border:1px solid var(--helto-border-strong);border-radius:var(--helto-radius-sm);background:var(--helto-surface-2);color:var(--helto-text-faint);transition:border-color var(--helto-transition),box-shadow var(--helto-transition)}
    .aio-ideo-library-search:focus-within{border-color:var(--helto-focus);box-shadow:var(--helto-focus-ring)}
    .aio-ideo-library-search input,.aio-ideo-library select,.aio-ideo-library input,.aio-ideo-library textarea{min-width:0;box-sizing:border-box;background:var(--helto-surface-2);color:var(--helto-text);border:1px solid var(--helto-border-strong);border-radius:var(--helto-radius-sm);font:inherit}
    .aio-ideo-library input::placeholder,.aio-ideo-library textarea::placeholder{color:var(--helto-text-faint)}
    .aio-ideo-library input:focus,.aio-ideo-library select:focus,.aio-ideo-library textarea:focus{outline:none;border-color:var(--helto-focus);box-shadow:var(--helto-focus-ring)}
    .aio-ideo-library-search input{height:30px;border:0;background:transparent;outline:0;color:var(--helto-text)}
    .aio-ideo-library select{height:32px;padding:0 8px;cursor:pointer}
    .aio-ideo-library-body{min-height:0;display:grid;grid-template-columns:minmax(320px,1fr) 320px;border-top:1px solid var(--helto-border);overflow:hidden}
    .aio-ideo-library-grid{min-height:0;overflow:auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));grid-auto-rows:min-content;gap:12px;padding:16px}
    .aio-ideo-library-details{min-height:0;overflow:auto;display:flex;flex-direction:column;gap:12px;padding:16px 18px;border-left:1px solid var(--helto-border)}
    /* Cards = tiles: 2px transparent border -> strong on hover -> GOLD + glow when selected. */
    .aio-ideo-library-card{min-width:0;display:flex;flex-direction:column;gap:8px;padding:10px;border:2px solid transparent;border-radius:var(--helto-radius);background:var(--helto-surface-2);color:var(--helto-text);text-align:left;cursor:pointer;transition:border-color var(--helto-transition),box-shadow var(--helto-transition),background var(--helto-transition)}
    .aio-ideo-library-card:hover{border-color:var(--helto-border-strong);background:var(--helto-surface-3)}
    .aio-ideo-library-card.is-selected{border-color:var(--helto-accent);box-shadow:var(--helto-shadow-glow)}
    .aio-ideo-library-card:focus-visible{outline:none;border-color:var(--helto-focus);box-shadow:var(--helto-focus-ring)}
    .aio-ideo-library-card-title,.aio-ideo-library-detail-title{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--helto-text);font-weight:700}
    .aio-ideo-library-detail-title{font-size:15px}
    .aio-ideo-library-card-meta,.aio-ideo-library-preview,.aio-ideo-library-detail-meta,.aio-ideo-library-status{min-width:0;overflow:hidden;text-overflow:ellipsis;color:var(--helto-text-dim)}
    .aio-ideo-library-preview{display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;white-space:normal;min-height:72px;padding:8px;border:1px solid var(--helto-border);border-radius:var(--helto-radius-sm);background:var(--helto-bg);color:var(--helto-text-dim);font:11px/1.35 var(--helto-font-mono)}
    .aio-ideo-library-actions,.aio-ideo-library-card-actions{display:flex;align-items:center;justify-content:flex-end;gap:7px;margin-top:auto}
    .aio-ideo-library button{min-width:28px;height:28px;min-height:28px;display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:0 10px;border:1px solid var(--helto-border-strong);border-radius:var(--helto-radius-sm);background:linear-gradient(180deg,var(--helto-surface-3),var(--helto-surface-2));color:var(--helto-text);font:inherit;cursor:pointer;transition:background var(--helto-transition),border-color var(--helto-transition),color var(--helto-transition),box-shadow var(--helto-transition)}
    .aio-ideo-library button:hover{background:linear-gradient(180deg,var(--helto-surface-hover),var(--helto-surface-3));border-color:var(--helto-border-hover);color:var(--helto-text)}
    .aio-ideo-library button:active{transform:translateY(1px)}
    .aio-ideo-library button:focus-visible{outline:none;border-color:var(--helto-focus);box-shadow:var(--helto-focus-ring)}
    /* Affirmative action = GOLD gradient (not a solid block). */
    .aio-ideo-library button.primary,.aio-ideo-library button.positive{border-color:var(--helto-accent-border);background:linear-gradient(180deg,#4f3a2a,#3d2d20);color:var(--helto-accent-strong)}
    .aio-ideo-library button.primary:hover,.aio-ideo-library button.positive:hover{background:linear-gradient(180deg,#5d4531,#493626);color:var(--helto-accent-strong)}
    /* Destructive = red gradient. */
    .aio-ideo-library button.danger{border-color:var(--helto-danger-border);background:linear-gradient(180deg,#5c2c3d,#482331);color:var(--helto-danger)}
    .aio-ideo-library button.danger:hover{border-color:var(--helto-danger);background:linear-gradient(180deg,#6e3549,#5a2a3c);color:var(--helto-danger)}
    .aio-ideo-library .aio-ideo-icon-btn{width:34px;padding:0}
    .aio-ideo-library-status{min-height:18px;padding:0 16px 12px;color:var(--helto-text-dim)}
    .aio-ideo-library-empty{grid-column:1/-1;padding:28px 8px;text-align:center;color:var(--helto-text-faint)}
    .aio-ideo-library-save-form{display:grid;gap:8px;padding:10px;border:1px solid var(--helto-border);border-radius:var(--helto-radius);background:var(--helto-surface)}
    .aio-ideo-library-save-form input,.aio-ideo-library-save-form textarea{width:100%;padding:7px 9px}
    .aio-ideo-library-save-form textarea{min-height:70px;resize:vertical;font-family:var(--helto-font-mono)}
    .aio-ideo-library-private-row{display:flex;gap:7px;align-items:center;color:var(--helto-text-dim)}
    .aio-ideo-library-private-row input[type="checkbox"]{accent-color:var(--helto-accent)}
    .aio-ideo-library.privacy-mode:not(.is-revealed) .aio-ideo-library-preview,
    .aio-ideo-library.privacy-mode:not(.is-revealed) .aio-ideo-library-card-meta,
    .aio-ideo-library.privacy-mode:not(.is-revealed) .aio-ideo-library-detail-meta { color: transparent !important; -webkit-text-fill-color: transparent !important; text-shadow: none !important; }
    .aio-ideo-library.privacy-mode.is-revealed .aio-ideo-library-preview,
    .aio-ideo-library.privacy-mode.is-revealed .aio-ideo-library-card-meta,
    .aio-ideo-library.privacy-mode.is-revealed .aio-ideo-library-detail-meta { color: var(--helto-text-dim); }
  `;
  document.head.appendChild(style);
}

function createEditor(node) {
  installStyles();
  const elementsWidget = widgetByName(node, "elements_data");
  const stylePaletteWidget = widgetByName(node, "style_palette_data");
  const brightnessWidget = widgetByName(node, "bg_brightness");
  const outputFormatWidget = widgetByName(node, "output_format");
  const coordModeWidget = widgetByName(node, "coord_mode");
  const bboxOrderWidget = widgetByName(node, "bbox_order");
  const privacyWidget = widgetByName(node, PRIVACY_WIDGET_NAME);
  let privacyReveal = false;
  const privacyRevealSources = {
    editor: false,
    field: false,
  };
  const nativePrivacyHoveredElements = new Set();
  const nativePrivacyFocusedElements = new Set();
  const nativePrivacyElements = new Set();
  for (const widget of [elementsWidget, stylePaletteWidget, brightnessWidget, outputFormatWidget, coordModeWidget, bboxOrderWidget]) {
    if (!widget) continue;
    widget.hidden = true;
    widget.serialize = true;
    widget.options ||= {};
    widget.options.serialize = true;
    widget.computeSize = () => [0, -4];
  }

  function privacyEnabled() {
    return Boolean(privacyWidget?.value);
  }

  function setStatus(message = "") {
    privacyStatus.textContent = message;
    privacyStatus.style.display = message ? "" : "none";
  }

  function nativeSensitiveWidgetsMasked() {
    return privacyEnabled() && !privacyReveal;
  }

  function refreshPrivacyRevealState() {
    privacyRevealSources.field = nativePrivacyHoveredElements.size > 0 || nativePrivacyFocusedElements.size > 0;
    privacyReveal = Object.values(privacyRevealSources).some(Boolean);
    return privacyReveal;
  }

  function setPrivacyRevealSource(source, revealed) {
    privacyRevealSources[source] = Boolean(revealed);
    refreshPrivacyRevealState();
    updatePrivacyClasses();
    draw();
  }

  function setNativeFieldReveal(source, element, revealed) {
    const elements = source === "focus" ? nativePrivacyFocusedElements : nativePrivacyHoveredElements;
    if (revealed) elements.add(element);
    else elements.delete(element);
    setPrivacyRevealSource("field", nativePrivacyHoveredElements.size > 0 || nativePrivacyFocusedElements.size > 0);
  }

  function sensitiveWidgetDomElements(widget) {
    const elements = [];
    for (const candidate of [widget?.inputEl, widget?.element, widget?.inputElement, widget?.textarea, widget?.textElement]) {
      if (candidate instanceof HTMLElement) elements.push(candidate);
    }
    for (const candidate of [...elements]) {
      elements.push(...(candidate.querySelectorAll?.("textarea,input,[contenteditable='true']") || []));
    }
    return [...new Set(elements)].filter((element) => element instanceof HTMLElement);
  }

  function patchSensitiveWidgetDomElement(element, widget) {
    if (!element || element._aioIdeoPrivacyRevealNode === node) return;
    element._aioIdeoPrivacyRevealCleanup?.();

    const onPointerEnter = () => setNativeFieldReveal("hover", element, true);
    const onPointerLeave = () => setNativeFieldReveal("hover", element, false);
    const onFocusIn = () => setNativeFieldReveal("focus", element, true);
    const onFocusOut = () => setNativeFieldReveal("focus", element, false);
    const onInput = () => {
      syncingExecutionWidgets = true;
      try {
        syncLiveWidgetTextValue(widget);
        setExecutionWidgetValue(elementsWidget, serializedElementsValue());
      } finally {
        syncingExecutionWidgets = false;
      }
      updateCount();
      app.graph?.setDirtyCanvas?.(true, true);
    };

    element.addEventListener("pointerenter", onPointerEnter);
    element.addEventListener("pointerleave", onPointerLeave);
    element.addEventListener("focusin", onFocusIn);
    element.addEventListener("focusout", onFocusOut);
    element.addEventListener("input", onInput);
    element.addEventListener("change", onInput);
    element.addEventListener("blur", onInput);
    nativePrivacyElements.add(element);
    element._aioIdeoPrivacyRevealNode = node;
    element._aioIdeoPrivacyRevealCleanup = () => {
      element.removeEventListener("pointerenter", onPointerEnter);
      element.removeEventListener("pointerleave", onPointerLeave);
      element.removeEventListener("focusin", onFocusIn);
      element.removeEventListener("focusout", onFocusOut);
      element.removeEventListener("input", onInput);
      element.removeEventListener("change", onInput);
      element.removeEventListener("blur", onInput);
      element.classList.remove("aio-ideo-private-field");
      element.removeAttribute("data-aio-ideo-private");
      nativePrivacyHoveredElements.delete(element);
      nativePrivacyFocusedElements.delete(element);
      nativePrivacyElements.delete(element);
      delete element._aioIdeoPrivacyRevealNode;
      delete element._aioIdeoPrivacyRevealCleanup;
    };
  }

  function pruneDisconnectedNativePrivacyElements(currentElements = null) {
    const keep = currentElements || new Set();
    for (const element of [...nativePrivacyHoveredElements]) {
      if (!element.isConnected || (currentElements && !keep.has(element))) nativePrivacyHoveredElements.delete(element);
    }
    for (const element of [...nativePrivacyFocusedElements]) {
      if (!element.isConnected || (currentElements && !keep.has(element))) nativePrivacyFocusedElements.delete(element);
    }
  }

  function updateSensitiveWidgetDomPrivacy() {
    const currentElements = new Set();
    for (const name of SENSITIVE_WIDGET_NAMES) {
      const widget = widgetByName(node, name);
      for (const element of sensitiveWidgetDomElements(widget)) {
        currentElements.add(element);
        patchSensitiveWidgetDomElement(element, widget);
        if (element === document.activeElement || element.contains(document.activeElement)) {
          nativePrivacyFocusedElements.add(element);
        }
      }
    }
    pruneDisconnectedNativePrivacyElements(currentElements);
    refreshPrivacyRevealState();
    const masked = nativeSensitiveWidgetsMasked();
    for (const element of currentElements) {
      element.classList.toggle("aio-ideo-private-field", masked);
      element.setAttribute("data-aio-ideo-private", masked ? "true" : "false");
    }
  }

  function cleanupSensitiveWidgetDomPrivacy() {
    for (const element of [...nativePrivacyElements]) {
      element._aioIdeoPrivacyRevealCleanup?.();
    }
    nativePrivacyHoveredElements.clear();
    nativePrivacyFocusedElements.clear();
    nativePrivacyElements.clear();
    refreshPrivacyRevealState();
  }

  function directWidgetValue(widget) {
    const value = widget?.value;
    if (value == null) return "";
    if (typeof value === "string") return value;
    if (isAnyAioPrivacyPayload(value)) return JSON.stringify(parsePrivacyPayload(value));
    return "";
  }

  function liveWidgetValue(widget) {
    const domValue = widgetDomTextValue(widget);
    if (domValue != null) return domValue;
    return directWidgetValue(widget);
  }

  function captionWidgetValues() {
    const values = {};
    for (const name of SENSITIVE_WIDGET_NAMES) {
      if (["elements_data", "style_palette_data", "import_json"].includes(name)) continue;
      values[name] = liveWidgetValue(widgetByName(node, name));
    }
    return values;
  }

  function serializedElementsValue() {
    const widgets = captionWidgetValues();
    const hasWidgetText = Object.values(widgets).some((value) => String(value || "").trim());
    if (!boxes.length && !hasWidgetText) return "";
    if (!hasWidgetText) return JSON.stringify(boxes);
    return JSON.stringify({
      version: 1,
      elements: boxes,
      widgets,
    });
  }

  function serializedStylePaletteValue() {
    return stylePalette.length ? JSON.stringify(stylePalette) : "";
  }

  function serializedOutputFormatValue() {
    return compact.checked ? "compact" : "pretty";
  }

  function serializedCoordModeValue() {
    return absolute.checked ? "absolute" : "normalized";
  }

  function serializedBboxOrderValue() {
    return xyOrder.checked ? "xy" : "yx";
  }

  function setExecutionWidgetValue(widget, value) {
    if (!widget || widget.value === value) return;
    widget.value = value;
  }

  function syncLiveWidgetTextValue(widget) {
    const domValue = widgetDomTextValue(widget);
    if (domValue == null || isAnyAioPrivacyPayload(widget?.value)) return;
    setExecutionWidgetValue(widget, domValue);
  }

  function syncLiveWidgetTextValues() {
    for (const name of SENSITIVE_WIDGET_NAMES) {
      if (name === "elements_data" || name === "style_palette_data") continue;
      syncLiveWidgetTextValue(widgetByName(node, name));
    }
  }

  function syncExecutionWidgets() {
    if (privacyRestorePending || privacyRestoreFailed) return;
    syncingExecutionWidgets = true;
    try {
      syncLiveWidgetTextValues();
      setExecutionWidgetValue(elementsWidget, serializedElementsValue());
      setExecutionWidgetValue(stylePaletteWidget, serializedStylePaletteValue());
      setExecutionWidgetValue(outputFormatWidget, serializedOutputFormatValue());
      setExecutionWidgetValue(coordModeWidget, serializedCoordModeValue());
      setExecutionWidgetValue(bboxOrderWidget, serializedBboxOrderValue());
    } finally {
      syncingExecutionWidgets = false;
    }
  }

  function serializePrivateValue(widget, value) {
    if (!privacyEnabled()) return value;
    assertSupportedPrivacyPayload(widget?.value);
    if (isEncryptedPrivacyPayload(widget?.value)) {
      return typeof widget.value === "string" ? widget.value : JSON.stringify(parsePrivacyPayload(widget.value));
    }
    try {
      return encryptValueSync(value ?? "");
    } catch (error) {
      setStatus(`Privacy encryption failed: ${error.message}`);
      throw error;
    }
  }

  function existingEncryptedWorkflowPayload() {
    const candidates = [
      node._aioIdeogram4LastPrivatePayload,
      node.properties?.[STATE_PROPERTY],
      node._aioIdeogram4PendingWorkflowInfo?.[WORKFLOW_STATE_KEY],
      node._aioIdeogram4PendingWorkflowInfo?.ideo,
    ];
    for (const candidate of candidates) {
      assertSupportedPrivacyPayload(candidate);
      if (isEncryptedPrivacyPayload(candidate)) return parsePrivacyPayload(candidate);
    }
    return null;
  }

  function widgetValues() {
    syncExecutionWidgets();
    const values = {};
    for (const name of STATE_WIDGET_NAMES) {
      const widget = widgetByName(node, name);
      if (!widget) continue;
      values[name] = SENSITIVE_WIDGET_NAMES.includes(name) ? liveWidgetValue(widget) : widget.value;
    }
    return values;
  }

  function currentState() {
    return {
      version: 1,
      widgets: widgetValues(),
      elements: cloneJson(boxes, []),
      style_palette: cloneJson(stylePalette, []),
      bg_brightness: brightnessWidget?.value ?? 25,
      output_format: outputFormatWidget?.value ?? "compact",
      coord_mode: coordModeWidget?.value === "absolute" ? "absolute" : "normalized",
      bbox_order: bboxOrderWidget?.value === "xy" ? "xy" : "yx",
      active,
    };
  }

  function applyState(state, { restorePrivacyMode = false } = {}) {
    if (!state || typeof state !== "object") return;
    for (const [name, value] of Object.entries(state.widgets || {})) {
      const widget = widgetByName(node, name);
      if (widget && (restorePrivacyMode || name !== PRIVACY_WIDGET_NAME)) widget.value = value;
    }
    if (Array.isArray(state.elements)) {
      boxes = cloneJson(state.elements, []);
    } else if (typeof state.widgets?.elements_data === "string") {
      boxes = parseElementsPayload(state.widgets.elements_data).elements;
    }
    if (Array.isArray(state.style_palette)) {
      stylePalette = cloneJson(state.style_palette, []);
    } else if (typeof state.widgets?.style_palette_data === "string") {
      stylePalette = parseJsonList(state.widgets.style_palette_data);
    }
    if (state.bg_brightness != null && brightnessWidget) brightnessWidget.value = state.bg_brightness;
    if (state.output_format != null && outputFormatWidget) outputFormatWidget.value = state.output_format;
    if (state.coord_mode != null && coordModeWidget) coordModeWidget.value = state.coord_mode === "absolute" ? "absolute" : "normalized";
    if (state.bbox_order != null && bboxOrderWidget) bboxOrderWidget.value = state.bbox_order === "xy" ? "xy" : "yx";
    active = Math.max(-1, Math.min(boxes.length - 1, Number(state.active ?? active)));
  }

  function workflowStatePayload() {
    if (privacyEnabled() && (privacyRestorePending || privacyRestoreFailed)) {
      const existing = existingEncryptedWorkflowPayload();
      if (existing) return existing;
      throw new Error("Private prompt builder is locked and no encrypted workflow state is available to preserve.");
    }
    const state = currentState();
    if (!privacyEnabled()) {
      return state;
    }
    try {
      setStatus("");
      const envelope = encryptStateSync(state);
      node._aioIdeogram4LastPrivatePayload = envelope;
      return envelope;
    } catch (error) {
      setStatus(`Privacy encryption failed: ${error.message}`);
      console.error("[AIO Ideogram 4 Prompt Builder] privacy encryption failed", error);
      const existing = existingEncryptedWorkflowPayload();
      if (existing) return existing;
      throw error;
    }
  }

  function writeStateProperty() {
    node.properties ||= {};
    const payload = workflowStatePayload();
    if (payload) node.properties[STATE_PROPERTY] = payload;
  }

  function patchSensitiveWidget(widget) {
    if (!widget || widget._aioIdeoPrivacyPatched) return;
    widget.serialize = true;
    widget.options ||= {};
    widget.options.serialize = true;
    widget.serializeValue = function () {
      return serializePrivateValue(this, liveWidgetValue(this));
    };
    widget._aioIdeoPrivacyPatched = true;
  }

  for (const name of SENSITIVE_WIDGET_NAMES) {
    patchSensitiveWidget(widgetByName(node, name));
  }
  if (elementsWidget) {
    elementsWidget.serializeValue = function () {
      return serializePrivateValue(this, serializedElementsValue());
    };
  }
  if (stylePaletteWidget) {
    stylePaletteWidget.serializeValue = function () {
      return serializePrivateValue(this, serializedStylePaletteValue());
    };
  }
  if (outputFormatWidget) {
    outputFormatWidget.serializeValue = function () {
      return serializedOutputFormatValue();
    };
  }
  if (coordModeWidget) {
    coordModeWidget.serializeValue = function () {
      return serializedCoordModeValue();
    };
  }
  if (bboxOrderWidget) {
    bboxOrderWidget.serializeValue = function () {
      return serializedBboxOrderValue();
    };
  }

  let boxes = parseElementsPayload(elementsWidget?.value).elements;
  let stylePalette = parseJsonList(stylePaletteWidget?.value);
  let active = boxes.length ? 0 : -1;
  const savedState = parseWorkflowStatePayload(
    node._aioIdeogram4PendingWorkflowInfo?.[WORKFLOW_STATE_KEY] ||
      node._aioIdeogram4PendingWorkflowInfo?.ideo ||
      node.properties?.[STATE_PROPERTY],
  );
  if (savedState) {
    applyState(savedState);
    if (active < 0 && boxes.length) active = 0;
  }
  let dragMode = null;
  let dragStart = null;
  let dragBoxStart = null;
  let privacyRestorePending = false;
  let privacyRestoreFailed = false;
  let domWidget = null;
  let currentEditorWidth = MIN_WIDTH;
  let currentEditorHeight = EDITOR_HEIGHT;
  let syncingExecutionWidgets = false;
  let serializeActive = false;
  let workflowSerializationActive = false;

  const wrap = document.createElement("div");
  wrap.className = "aio-ideo-wrap";
  wrap.addEventListener("pointerdown", stopEvent);
  wrap.addEventListener("wheel", stopEvent);
  wrap.addEventListener("pointerenter", () => {
    setPrivacyRevealSource("editor", true);
  });
  wrap.addEventListener("pointerleave", () => {
    setPrivacyRevealSource("editor", false);
  });
  const privacyStatus = document.createElement("div");
  privacyStatus.className = "aio-ideo-privacy-status";
  privacyStatus.style.display = "none";

  const toolbar = document.createElement("div");
  toolbar.className = "aio-ideo-toolbar";
  const libraryBtn = iconButton("Library", "library", "Ideogram Prompt Library");
  const saveLibraryBtn = iconButton("Save", "save", "Save Prompt to Library");
  const copyBtn = iconButton("Copy", "copy", "Copy prompt JSON");
  const pasteBtn = iconButton("Paste", "paste", "Paste Ideogram 4 JSON");
  const clearBtn = iconButton("Clear", "clear", "Clear all regions and palette");
  const addTextBtn = iconButton("Add text", "text", "Add text region");
  const addObjBtn = iconButton("Add object", "obj", "Add object region");
  const compactLabel = document.createElement("label");
  compactLabel.style.display = "flex";
  compactLabel.style.gap = "3px";
  compactLabel.style.alignItems = "center";
  const compact = document.createElement("input");
  compact.type = "checkbox";
  compact.checked = outputFormatWidget?.value !== "pretty";
  compactLabel.append(compact, document.createTextNode("compact"));
  const absoluteLabel = document.createElement("label");
  absoluteLabel.style.display = "flex";
  absoluteLabel.style.gap = "3px";
  absoluteLabel.style.alignItems = "center";
  absoluteLabel.title = "Emit bbox coordinates in resolved pixels instead of the 0-1000 grid";
  const absolute = document.createElement("input");
  absolute.type = "checkbox";
  absolute.checked = coordModeWidget?.value === "absolute";
  absoluteLabel.append(absolute, document.createTextNode("px"));
  const xyLabel = document.createElement("label");
  xyLabel.style.display = "flex";
  xyLabel.style.gap = "3px";
  xyLabel.style.alignItems = "center";
  xyLabel.title = "Emit bbox as [xmin,ymin,xmax,ymax] for Qwen/Krea-style prompts";
  const xyOrder = document.createElement("input");
  xyOrder.type = "checkbox";
  xyOrder.checked = bboxOrderWidget?.value === "xy";
  xyLabel.append(xyOrder, document.createTextNode("xy"));
  const count = document.createElement("span");
  count.className = "aio-ideo-count";
  toolbar.append(libraryBtn, saveLibraryBtn, addObjBtn, addTextBtn, copyBtn, pasteBtn, clearBtn, compactLabel, absoluteLabel, xyLabel, count);

  const main = document.createElement("div");
  main.className = "aio-ideo-main";
  const canvasBox = document.createElement("div");
  canvasBox.className = "aio-ideo-canvasBox";
  const canvas = document.createElement("canvas");
  canvas.className = "aio-ideo-canvas";
  canvas.tabIndex = 0;
  canvasBox.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  const side = document.createElement("div");
  side.className = "aio-ideo-side";
  const list = document.createElement("div");
  list.className = "aio-ideo-list";
  const stylePaletteRow = document.createElement("div");
  stylePaletteRow.className = "aio-ideo-row aio-ideo-swatches";
  const addStyleColor = document.createElement("button");
  addStyleColor.className = "aio-ideo-accent";
  addStyleColor.textContent = "+ Style Color";
  const typeRow = document.createElement("div");
  typeRow.className = "aio-ideo-row";
  const typeSelect = document.createElement("select");
  for (const value of ["obj", "text"]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    typeSelect.appendChild(option);
  }
  typeRow.appendChild(typeSelect);
  const textRow = document.createElement("div");
  textRow.className = "aio-ideo-row";
  const textInput = document.createElement("input");
  textInput.placeholder = "Text element content";
  textRow.appendChild(textInput);
  const descRow = document.createElement("div");
  descRow.className = "aio-ideo-row";
  const descInput = document.createElement("textarea");
  descInput.placeholder = "Region description";
  descRow.appendChild(descInput);
  const bboxRow = document.createElement("div");
  bboxRow.className = "aio-ideo-row";
  const bboxInputs = ["x", "y", "w", "h"].map((name) => {
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.max = "1000";
    input.step = "1";
    input.placeholder = name;
    input.title = `BBox ${name} on the 0-1000 editor grid`;
    return input;
  });
  bboxRow.append(...bboxInputs);
  const paletteRow = document.createElement("div");
  paletteRow.className = "aio-ideo-row aio-ideo-swatches";
  const addColor = document.createElement("button");
  addColor.className = "aio-ideo-accent";
  addColor.textContent = "+ Color";
  const removeBtn = document.createElement("button");
  removeBtn.className = "aio-ideo-danger";
  removeBtn.textContent = "Delete";
  const help = document.createElement("div");
  help.className = "aio-ideo-help";
  help.textContent = "Drag on the canvas to create regions. BBox fields use x/y/w/h on a 0-1000 editor grid; toolbar toggles control JSON scale and order.";
  side.append(list, stylePaletteRow, addStyleColor, typeRow, textRow, descRow, bboxRow, paletteRow, addColor, removeBtn, help);
  main.append(canvasBox, side);
  wrap.append(toolbar, main, privacyStatus);

  function updatePrivacyClasses() {
    updateSensitiveWidgetDomPrivacy();
    const privateMode = privacyEnabled();
    wrap.classList.toggle("is-private", privateMode);
    wrap.classList.toggle("is-privacy-revealed", !privateMode || privacyReveal);
  }

  function nodeDrivenEditorHeight() {
    const nodeHeight = Number(node.size?.[1] || 0);
    const widgetY = Number(domWidget?.last_y || domWidget?.y || 0);
    if (nodeHeight > 0 && widgetY > 0) {
      return Math.max(EDITOR_MIN_HEIGHT, Math.floor(nodeHeight - widgetY - EDITOR_NODE_MARGIN * 2));
    }
    return EDITOR_HEIGHT;
  }

  function nodeDrivenEditorWidth() {
    const nodeWidth = Number(node.size?.[0] || 0);
    if (nodeWidth > 0) {
      return Math.max(MIN_WIDTH, Math.floor(nodeWidth - EDITOR_NODE_MARGIN * 2));
    }
    return MIN_WIDTH;
  }

  function ensureNodeFitsEditor(editorHeight = currentEditorHeight) {
    if (!Array.isArray(node.size)) return;
    const widgetY = Number(domWidget?.last_y || domWidget?.y || 0);
    const requiredHeight =
      widgetY > 0 ? Math.ceil(widgetY + editorWidgetHeight(editorHeight)) : EDITOR_INITIAL_NODE_HEIGHT;
    const requiredWidth = MIN_NODE_WIDTH;
    const nextWidth = Math.max(Number(node.size[0]) || 0, requiredWidth);
    const nextHeight = Math.max(Number(node.size[1]) || 0, requiredHeight);
    if (nextWidth > node.size[0] + 0.5 || nextHeight > node.size[1] + 0.5) {
      node.setSize?.([nextWidth, nextHeight]);
    }
  }

  function syncEditorSize({ fromNodeResize = false } = {}) {
    if (fromNodeResize) {
      currentEditorWidth = nodeDrivenEditorWidth();
      currentEditorHeight = nodeDrivenEditorHeight();
    }
    ensureNodeFitsEditor();
    wrap.style.setProperty("--aio-ideo-editor-width", `${currentEditorWidth}px`);
    wrap.style.setProperty("--aio-ideo-editor-height", `${currentEditorHeight}px`);
    if (domWidget?.element?.style) {
      domWidget.element.style.width = `${currentEditorWidth}px`;
      domWidget.element.style.maxWidth = "none";
    }
    requestAnimationFrame(() => fitCanvas());
    return { width: currentEditorWidth, height: currentEditorHeight };
  }

  function preserveNodeSize(size = node.size) {
    if (!Array.isArray(size)) return;
    const target = [Number(size[0]), Number(size[1])];
    if (!Number.isFinite(target[0]) || !Number.isFinite(target[1])) return;
    const restore = () => {
      if (!Array.isArray(node.size)) return;
      if (node.size[0] < target[0] - 1 || node.size[1] < target[1] - 1) {
        node.setSize?.(target);
        currentEditorWidth = nodeDrivenEditorWidth();
        currentEditorHeight = nodeDrivenEditorHeight();
        syncEditorSize();
      }
    };
    requestAnimationFrame(() => {
      restore();
      requestAnimationFrame(restore);
    });
  }

  node._aioIdeogram4EditorWidth = () => currentEditorWidth;
  node._aioIdeogram4EditorHeight = () => currentEditorHeight;
  node._aioIdeogram4SetDomWidget = (widget) => {
    domWidget = widget;
    syncEditorSize({ fromNodeResize: true });
  };

  function serialize() {
    if (serializeActive || workflowSerializationActive) return;
    serializeActive = true;
    try {
      syncExecutionWidgets();
      if (!privacyRestorePending && !privacyRestoreFailed) {
        writeStateProperty();
      }
      updatePrivacyClasses();
      app.graph?.setDirtyCanvas?.(true, true);
      managedEditHandler?.();
    } finally {
      serializeActive = false;
    }
  }

  function workflowWidgetValue(widget) {
    if (!widget) return "";
    if (widget === elementsWidget) return serializePrivateValue(widget, serializedElementsValue());
    if (widget === stylePaletteWidget) return serializePrivateValue(widget, serializedStylePaletteValue());
    return serializePrivateValue(widget, liveWidgetValue(widget));
  }

  function scrubWorkflowWidgets(output) {
    if (!output || !privacyEnabled() || !Array.isArray(output.widgets_values)) return;
    for (const name of SENSITIVE_WIDGET_NAMES) {
      const widget = widgetByName(node, name);
      const index = node.widgets?.indexOf(widget);
      if (!widget || index == null || index < 0 || index >= output.widgets_values.length) continue;
      output.widgets_values[index] = workflowWidgetValue(widget);
    }
  }

  function serializeForWorkflow(output) {
    if (!output) return;
    if (workflowSerializationActive) return;
    workflowSerializationActive = true;
    try {
      syncExecutionWidgets();
      const payload = workflowStatePayload();
      if (payload) {
        node.properties ||= {};
        node.properties[STATE_PROPERTY] = payload;
        if (output.properties && typeof output.properties === "object") {
          output.properties[STATE_PROPERTY] = payload;
        }
        output[WORKFLOW_STATE_KEY] = payload;
      }
      scrubWorkflowWidgets(output);
    } finally {
      workflowSerializationActive = false;
    }
  }

  function syncControlWidgetsFromState() {
    compact.checked = outputFormatWidget?.value !== "pretty";
    absolute.checked = coordModeWidget?.value === "absolute";
    xyOrder.checked = bboxOrderWidget?.value === "xy";
  }

  function repaintRestoredState() {
    syncControlWidgetsFromState();
    renderList();
    renderForm();
    draw();
    updateCount();
    updatePrivacyClasses();
  }

  function restorePlainState(state, { markDirty = true, restorePrivacyMode = false } = {}) {
    applyState(state, { restorePrivacyMode });
    if (boxes.length && active < 0) active = 0;
    privacyRestorePending = false;
    privacyRestoreFailed = false;
    syncControlWidgetsFromState();
    if (markDirty) refresh();
    else repaintRestoredState();
  }

  node._aioIdeogram4RecoveryReset = () => {
    const widgets = Object.fromEntries(SENSITIVE_WIDGET_NAMES.map((name) => [name, ""]));
    delete node._aioIdeogram4LastPrivatePayload;
    if (node.properties) delete node.properties[STATE_PROPERTY];
    if (node._aioIdeogram4PendingWorkflowInfo) {
      delete node._aioIdeogram4PendingWorkflowInfo[WORKFLOW_STATE_KEY];
      delete node._aioIdeogram4PendingWorkflowInfo.ideo;
    }
    restorePlainState({
      version: 1,
      widgets,
      elements: [],
      style_palette: [],
      bg_brightness: brightnessWidget?.value ?? 25,
      output_format: outputFormatWidget?.value ?? "compact",
      coord_mode: coordModeWidget?.value === "absolute" ? "absolute" : "normalized",
      bbox_order: bboxOrderWidget?.value === "xy" ? "xy" : "yx",
      active: -1,
    });
    setStatus("Private prompt builder state reset.");
  };

  async function decryptPayloadState(rawPayload, label = "prompt builder") {
    assertSupportedPrivacyPayload(rawPayload);
    if (isLegacyPrivacyPayload(rawPayload)) {
      throw new Error("Unsupported legacy AIO privacy payload. Re-enter the private value to save it with the shared privacy keystore.");
    }
    if (!isEncryptedPrivacyPayload(rawPayload)) {
      return parseWorkflowStatePayload(rawPayload);
    }
    privacyRestorePending = true;
    node._aioIdeogram4LastPrivatePayload = parsePrivacyPayload(rawPayload);
    setStatus(`Decrypting private ${label}...`);
    try {
      const state = await decryptState(parsePrivacyPayload(rawPayload));
      setStatus("");
      return state;
    } catch (error) {
      privacyRestorePending = false;
      privacyRestoreFailed = true;
      setStatus(`Private ${label} locked: ${error.message}`);
      console.error("[AIO Ideogram 4 Prompt Builder] privacy decrypt failed", error);
      updatePrivacyClasses();
      return null;
    }
  }

  async function restoreEncryptedWidgets() {
    let restored = false;
    for (const name of SENSITIVE_WIDGET_NAMES) {
      const widget = widgetByName(node, name);
      if (!widget) continue;
      try {
        assertSupportedPrivacyPayload(widget.value);
      } catch (error) {
        privacyRestorePending = false;
        privacyRestoreFailed = true;
        setStatus(`Private prompt builder recovery needed: ${error.message}`);
        console.error("[AIO Ideogram 4 Prompt Builder] privacy widget payload unsupported", error);
        updatePrivacyClasses();
        return false;
      }
      if (!isEncryptedPrivacyPayload(widget.value) && !isLegacyPrivacyPayload(widget.value)) continue;
      privacyRestorePending = true;
      setStatus("Decrypting private prompt builder widgets...");
      try {
        widget.value = await decryptValue(widget.value);
        restored = true;
      } catch (error) {
        privacyRestorePending = false;
        privacyRestoreFailed = true;
        setStatus(`Private prompt builder locked: ${error.message}`);
        console.error("[AIO Ideogram 4 Prompt Builder] privacy widget decrypt failed", error);
        updatePrivacyClasses();
        return false;
      }
    }
    if (!restored) return false;

    boxes = parseElementsPayload(elementsWidget?.value).elements;
    stylePalette = parseJsonList(stylePaletteWidget?.value);
    if (boxes.length && active < 0) active = 0;
    privacyRestorePending = false;
    privacyRestoreFailed = false;
    setStatus("");
    repaintRestoredState();
    return true;
  }

  function restoreFromWorkflow(info) {
    if (!info || typeof info !== "object") return;
    const rawPayload = info[WORKFLOW_STATE_KEY] || info.ideo || node.properties?.[STATE_PROPERTY];
    if (isAnyAioPrivacyPayload(rawPayload)) {
      decryptPayloadState(rawPayload, "prompt builder").then((state) => {
        if (state) restorePlainState(state);
      }).catch((error) => {
        privacyRestorePending = false;
        privacyRestoreFailed = true;
        setStatus(`Private prompt builder locked: ${error.message}`);
        console.error("[AIO Ideogram 4 Prompt Builder] privacy decrypt failed", error);
        updatePrivacyClasses();
      });
      return;
    }

    let state = parseWorkflowStatePayload(rawPayload);
    if (!state && Array.isArray(info.widgets_values)) {
      let restoredBoxes = null;
      let restoredPalette = null;
      for (const value of info.widgets_values) {
        const boxesCandidate = parseElementsPayload(value).elements;
        if (!restoredBoxes && boxesCandidate.some((box) => box && typeof box === "object" && typeof box.x === "number")) {
          restoredBoxes = boxesCandidate;
          continue;
        }
        if (!restoredPalette && boxesCandidate.every((color) => typeof color === "string" && color.startsWith("#"))) {
          restoredPalette = boxesCandidate;
        }
      }
      if (restoredBoxes || restoredPalette) {
        state = {
          version: 1,
          elements: restoredBoxes || boxes,
          style_palette: restoredPalette || stylePalette,
          active,
        };
      }
    }
    if (state) restorePlainState(state);
    else restoreEncryptedWidgets();
  }

  let managedEditHandler = null;
  node._aioIdeogram4EditorApi = {
    serializeForWorkflow,
    restoreFromWorkflow,
    flushManagedEdits() {
      syncExecutionWidgets();
      return currentState();
    },
    applyManagedState(state) {
      restorePlainState(state, { markDirty: false, restorePrivacyMode: true });
    },
    clearManagedState() {
      const widgets = Object.fromEntries(SENSITIVE_WIDGET_NAMES.map((name) => [name, ""]));
      restorePlainState({
        version: 1,
        widgets,
        elements: [],
        style_palette: [],
        bg_brightness: brightnessWidget?.value ?? 25,
        output_format: outputFormatWidget?.value ?? "compact",
        coord_mode: coordModeWidget?.value === "absolute" ? "absolute" : "normalized",
        bbox_order: bboxOrderWidget?.value === "xy" ? "xy" : "yx",
        active: -1,
      }, { markDirty: false });
    },
    setManagedEditHandler(handler) {
      if (handler !== null && typeof handler !== "function") {
        throw new Error("Invalid managed privacy edit handler.");
      }
      managedEditHandler = handler;
    },
  };

  function captionText() {
    syncExecutionWidgets();
    return formatCaption(node, buildCaption(node, boxes, stylePalette));
  }

  function libraryItemId() {
    return String(node.properties?.[LIBRARY_ITEM_PROPERTY] || "").trim();
  }

  function setLibraryItemId(itemId) {
    node.properties ||= {};
    if (itemId) node.properties[LIBRARY_ITEM_PROPERTY] = itemId;
    else delete node.properties[LIBRARY_ITEM_PROPERTY];
    updateLibraryButtons();
  }

  function libraryPayload() {
    return {
      family: "ideogram4",
      version: 1,
      state: currentState(),
      prompt: captionText(),
    };
  }

  function promptLibraryName() {
    const widgets = currentState().widgets || {};
    const candidates = [
      widgets.high_level_description,
      widgets.background,
      captionText().replace(/\s+/g, " ").slice(0, 60),
    ];
    return String(candidates.find((value) => String(value || "").trim()) || "Untitled Ideogram Prompt").trim();
  }

  function updateLibraryButtons() {
    const linked = Boolean(libraryItemId());
    saveLibraryBtn.classList.toggle("aio-ideo-library-linked", linked);
    saveLibraryBtn.title = linked ? "Update Saved Prompt" : "Save Prompt to Library";
    saveLibraryBtn.setAttribute("aria-label", saveLibraryBtn.title);
  }

  async function saveCurrentPromptToLibrary({
    itemId = libraryItemId(),
    metadata = null,
    statusCallback = setStatus,
  } = {}) {
    const linkedId = String(itemId || "").trim();
    const body = {
      name: metadata?.name || promptLibraryName(),
      description: metadata?.description || "",
      tags: Array.isArray(metadata?.tags) ? metadata.tags : [],
      private: metadata?.private ?? privacyEnabled(),
      prompt: libraryPayload(),
    };
    const url = linkedId
      ? `${LIBRARY_ROUTE}/prompts/${encodeURIComponent(linkedId)}`
      : `${LIBRARY_ROUTE}/prompts`;
    const data = await fetchLibraryJson(url, {
      method: linkedId ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const item = data.item || {};
    if (item.id) setLibraryItemId(item.id);
    statusCallback(linkedId ? "Updated saved prompt." : "Saved prompt.");
    app.graph?.setDirtyCanvas?.(true, true);
    return item;
  }

  async function loadLibraryItem(item, { finish = null, statusCallback = setStatus } = {}) {
    if (!item?.id) return;
    statusCallback("Loading saved prompt...");
    const data = await fetchLibraryJson(`${LIBRARY_ROUTE}/prompts/${encodeURIComponent(item.id)}/use`, { method: "POST" });
    const payload = data.prompt || data.item?.payload || data.item?.prompt;
    const state = payload?.state;
    if (!state || typeof state !== "object") throw new Error("Saved prompt did not include builder state.");
    restorePlainState(state, { restorePrivacyMode: true });
    setLibraryItemId(item.id);
    statusCallback(`Loaded ${data.item?.name || item.name || "saved prompt"}.`);
    finish?.();
  }

  function libraryLoadErrorMessage(item, error) {
    if (item?.private || item?.is_private) {
      return `Private saved prompt cannot be decrypted. It can still be deleted from the library. ${error.message}`;
    }
    return error.message;
  }

  function showPromptLibrary({ openSave = false } = {}) {
    const existing = document.querySelector(".aio-ideo-library");
    existing?.remove();

    const overlay = document.createElement("div");
    overlay.className = `aio-ideo-library${privacyEnabled() ? " privacy-mode" : ""}`;
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Ideogram Prompt Library");
    overlay.addEventListener("pointerdown", stopEvent);
    overlay.addEventListener("wheel", stopEvent);
    overlay.addEventListener("pointerenter", () => overlay.classList.add("is-revealed"));
    overlay.addEventListener("pointerleave", () => overlay.classList.remove("is-revealed"));

    const panel = document.createElement("div");
    panel.className = "aio-ideo-library-panel";
    const head = document.createElement("div");
    head.className = "aio-ideo-library-head";
    const title = document.createElement("div");
    title.className = "aio-ideo-library-title";
    title.textContent = "Ideogram Prompt Library";
    const closeBtn = iconButton("Close", "close", "Close Prompt Library");
    head.append(title, closeBtn);

    const controls = document.createElement("div");
    controls.className = "aio-ideo-library-controls";
    const searchWrap = document.createElement("label");
    searchWrap.className = "aio-ideo-library-search";
    searchWrap.innerHTML = ICONS.search;
    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = "Search prompts...";
    searchWrap.append(search);
    const sort = document.createElement("select");
    for (const [value, label] of [
      ["newest", "Newest"],
      ["oldest", "Oldest"],
      ["name", "Name"],
      ["used", "Last used"],
    ]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      sort.append(option);
    }
    const addBtn = textButton("Save Current", "save");
    addBtn.className = "primary";
    controls.append(searchWrap, sort, addBtn);

    const body = document.createElement("div");
    body.className = "aio-ideo-library-body";
    const grid = document.createElement("div");
    grid.className = "aio-ideo-library-grid";
    const details = document.createElement("div");
    details.className = "aio-ideo-library-details";
    body.append(grid, details);
    const status = document.createElement("div");
    status.className = "aio-ideo-library-status";
    panel.append(head, controls, body, status);
    overlay.append(panel);
    document.body.append(overlay);

    const state = { items: [], search: "", sort: "newest", selectedId: "", saveOpen: openSave };
    const finish = () => overlay.remove();
    const setLibraryStatus = (message = "") => {
      status.textContent = message;
    };
    closeBtn.addEventListener("click", finish);
    addBtn.addEventListener("click", () => {
      state.saveOpen = true;
      render();
    });
    search.addEventListener("input", () => {
      state.search = search.value;
      render();
    });
    sort.addEventListener("change", () => {
      state.sort = sort.value;
      render();
    });

    function filteredItems() {
      const query = state.search.trim().toLowerCase();
      const items = query
        ? state.items.filter((item) => {
            const haystack = [
              item.name,
              item.description,
              item.prompt_preview,
              ...(Array.isArray(item.tags) ? item.tags : []),
            ].join(" ").toLowerCase();
            return haystack.includes(query);
          })
        : [...state.items];
      items.sort((a, b) => {
        if (state.sort === "oldest") return String(a.created_at || "").localeCompare(String(b.created_at || ""));
        if (state.sort === "name") return String(a.name || "").localeCompare(String(b.name || ""));
        if (state.sort === "used") return String(b.last_used_at || "").localeCompare(String(a.last_used_at || ""));
        return String(b.updated_at || b.created_at || "").localeCompare(String(a.updated_at || a.created_at || ""));
      });
      return items;
    }

    function selectedItem() {
      const items = filteredItems();
      return items.find((item) => item.id === state.selectedId) || items[0] || null;
    }

    function render() {
      grid.innerHTML = "";
      const items = filteredItems();
      if (!items.length) {
        const empty = document.createElement("div");
        empty.className = "aio-ideo-library-empty";
        empty.textContent = state.search ? "No matching prompts." : "No saved prompts yet.";
        grid.append(empty);
      }
      for (const item of items) {
        const card = document.createElement("div");
        card.tabIndex = 0;
        card.setAttribute("role", "button");
        card.className = "aio-ideo-library-card" + (item.id === selectedItem()?.id ? " is-selected" : "");
        const cardTitle = document.createElement("div");
        cardTitle.className = "aio-ideo-library-card-title";
        cardTitle.textContent = item.name || "Untitled Ideogram Prompt";
        const meta = document.createElement("div");
        meta.className = "aio-ideo-library-card-meta";
        meta.textContent = `${item.private ? "Private" : "Public"} · ${item.summary?.element_count ?? 0} regions · ${item.summary?.aspect_ratio || "ratio"}`;
        const preview = document.createElement("div");
        preview.className = "aio-ideo-library-preview";
        preview.textContent = item.private ? "Private prompt" : item.prompt_preview || "No prompt preview";
        const actions = document.createElement("div");
        actions.className = "aio-ideo-library-card-actions";
        const loadBtn = iconButton("Load", "load", "Load Saved Prompt");
        loadBtn.classList.add("primary");
        loadBtn.addEventListener("click", async (event) => {
          event.stopPropagation();
          try {
            await loadLibraryItem(item, { finish, statusCallback: setLibraryStatus });
          } catch (error) {
            setLibraryStatus(`Private prompt locked: ${error.message}`);
          }
        });
        actions.append(loadBtn);
        card.append(cardTitle, meta, preview, actions);
        card.addEventListener("click", () => {
          state.selectedId = item.id;
          state.saveOpen = false;
          render();
        });
        card.addEventListener("keydown", (event) => {
          if (event.key !== "Enter" && event.key !== " ") return;
          event.preventDefault();
          state.selectedId = item.id;
          state.saveOpen = false;
          render();
        });
        grid.append(card);
      }
      renderDetails();
    }

    function renderDetails() {
      details.innerHTML = "";
      if (state.saveOpen) {
        renderSaveForm();
        return;
      }
      const item = selectedItem();
      if (!item) {
        const empty = document.createElement("div");
        empty.className = "aio-ideo-library-empty";
        empty.textContent = "Select or save a prompt.";
        details.append(empty);
        return;
      }
      const detailTitle = document.createElement("div");
      detailTitle.className = "aio-ideo-library-detail-title";
      detailTitle.textContent = item.name || "Untitled Ideogram Prompt";
      const meta = document.createElement("div");
      meta.className = "aio-ideo-library-detail-meta";
      meta.textContent = `${item.private ? "Private" : "Public"} · ${item.summary?.prompt_char_count ?? 0} chars · updated ${item.updated_at || "unknown"}`;
      const preview = document.createElement("div");
      preview.className = "aio-ideo-library-preview";
      preview.textContent = item.private ? "Private prompt is available after load." : item.prompt_preview || "No prompt preview";
      const actions = document.createElement("div");
      actions.className = "aio-ideo-library-actions";
      const loadBtn = textButton("Load", "load");
      loadBtn.className = "primary";
      const overwriteBtn = textButton("Overwrite", "save");
      overwriteBtn.className = "positive";
      const renameBtn = iconButton("Rename", "edit", "Rename Saved Prompt");
      const duplicateBtn = iconButton("Duplicate", "copy", "Duplicate Saved Prompt");
      const deleteBtn = iconButton("Delete", "delete", "Delete Saved Prompt");
      deleteBtn.classList.add("danger");
      loadBtn.addEventListener("click", async () => {
        try {
          await loadLibraryItem(item, { finish, statusCallback: setLibraryStatus });
        } catch (error) {
          setLibraryStatus(libraryLoadErrorMessage(item, error));
        }
      });
      overwriteBtn.addEventListener("click", async () => {
        try {
          await saveCurrentPromptToLibrary({
            itemId: item.id,
            metadata: item.private
              ? { private: privacyEnabled() }
              : { name: item.name, description: item.description, tags: item.tags, private: privacyEnabled() },
            statusCallback: setLibraryStatus,
          });
          await refreshLibrary();
        } catch (error) {
          setLibraryStatus(error.message);
        }
      });
      renameBtn.addEventListener("click", async () => {
        const name = window.prompt("Rename saved prompt:", item.name || "");
        if (!name) return;
        try {
          await fetchLibraryJson(`${LIBRARY_ROUTE}/prompts/${encodeURIComponent(item.id)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
          });
          setLibraryStatus("Renamed prompt.");
          await refreshLibrary();
        } catch (error) {
          setLibraryStatus(error.message);
        }
      });
      duplicateBtn.addEventListener("click", async () => {
        try {
          await fetchLibraryJson(`${LIBRARY_ROUTE}/prompts/${encodeURIComponent(item.id)}/duplicate`, { method: "POST" });
          setLibraryStatus("Duplicated prompt.");
          await refreshLibrary();
        } catch (error) {
          setLibraryStatus(error.message);
        }
      });
      deleteBtn.addEventListener("click", async () => {
        if (!window.confirm(`Delete "${item.name || "saved prompt"}"?`)) return;
        try {
          await fetchLibraryJson(`${LIBRARY_ROUTE}/prompts/${encodeURIComponent(item.id)}`, { method: "DELETE" });
          if (libraryItemId() === item.id) setLibraryItemId("");
          state.selectedId = "";
          setLibraryStatus("Deleted prompt.");
          await refreshLibrary();
        } catch (error) {
          setLibraryStatus(error.message);
        }
      });
      actions.append(loadBtn, overwriteBtn, renameBtn, duplicateBtn, deleteBtn);
      details.append(detailTitle, meta, preview, actions);
    }

    function renderSaveForm() {
      const form = document.createElement("div");
      form.className = "aio-ideo-library-save-form";
      const name = document.createElement("input");
      name.placeholder = "Prompt name";
      name.value = promptLibraryName();
      const description = document.createElement("textarea");
      description.placeholder = "Description";
      const privateLabel = document.createElement("label");
      privateLabel.className = "aio-ideo-library-private-row";
      const privateInput = document.createElement("input");
      privateInput.type = "checkbox";
      privateInput.checked = privacyEnabled();
      privateLabel.append(privateInput, document.createTextNode("Private"));
      const actions = document.createElement("div");
      actions.className = "aio-ideo-library-actions";
      const saveBtn = textButton("Save", "save");
      saveBtn.className = "primary";
      const cancelBtn = textButton("Cancel");
      actions.append(cancelBtn, saveBtn);
      form.append(name, description, privateLabel, actions);
      cancelBtn.addEventListener("click", () => {
        state.saveOpen = false;
        render();
      });
      saveBtn.addEventListener("click", async () => {
        try {
          const item = await saveCurrentPromptToLibrary({
            itemId: "",
            metadata: {
              name: name.value,
              description: description.value,
              private: privateInput.checked,
            },
            statusCallback: setLibraryStatus,
          });
          state.saveOpen = false;
          state.selectedId = item.id || "";
          await refreshLibrary();
        } catch (error) {
          setLibraryStatus(error.message);
        }
      });
      details.append(form);
    }

    async function refreshLibrary() {
      const data = await fetchLibraryJson(`${LIBRARY_ROUTE}/items`);
      state.items = Array.isArray(data.prompts) ? data.prompts : [];
      if (!state.selectedId && state.items.length) state.selectedId = libraryItemId() || state.items[0].id;
      render();
    }

    refreshLibrary().catch((error) => {
      setLibraryStatus(error.message || "Could not load prompt library.");
      render();
    });
  }

  async function restoreEncryptedState() {
    const rawState =
      node._aioIdeogram4PendingWorkflowInfo?.[WORKFLOW_STATE_KEY] ||
      node._aioIdeogram4PendingWorkflowInfo?.ideo ||
      node.properties?.[STATE_PROPERTY];
    let state = null;
    try {
      state = await decryptPayloadState(rawState, "prompt builder");
    } catch (error) {
      privacyRestorePending = false;
      privacyRestoreFailed = true;
      setStatus(`Private prompt builder locked: ${error.message}`);
      console.error("[AIO Ideogram 4 Prompt Builder] privacy decrypt failed", error);
      updatePrivacyClasses();
      return;
    }
    if (state) {
      restorePlainState(state, { markDirty: false });
      return;
    }
    await restoreEncryptedWidgets();
  }

  function updateCount() {
    const text = captionText();
    count.textContent = `${Math.ceil(text.length / 4)} tok`;
    count.style.color = text.length >= 8192 ? HELTO.danger : text.length >= 6144 ? HELTO.warn : HELTO.textDim;
  }

  function canvasLogicalWidth() {
    return canvas.offsetWidth || parseFloat(canvas.style.width) || 1;
  }

  function canvasLogicalHeight() {
    return canvas.offsetHeight || parseFloat(canvas.style.height) || 1;
  }

  function fitCanvas() {
    const [width, height] = resolveDims(node);
    const availW = Math.max(1, canvasBox.clientWidth - 8);
    const availH = Math.max(1, canvasBox.clientHeight - 8);
    const aspect = width / height;
    let cw = availW;
    let ch = cw / aspect;
    if (ch > availH) {
      ch = availH;
      cw = ch * aspect;
    }
    canvas.style.width = Math.round(cw) + "px";
    canvas.style.height = Math.round(ch) + "px";
    draw();
  }

  function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    };
  }

  function boxAt(point) {
    for (let index = boxes.length - 1; index >= 0; index--) {
      const box = boxes[index];
      if (box.nobbox) continue;
      const x1 = Math.min(box.x, box.x + box.w);
      const x2 = Math.max(box.x, box.x + box.w);
      const y1 = Math.min(box.y, box.y + box.h);
      const y2 = Math.max(box.y, box.y + box.h);
      if (point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2) return index;
    }
    return -1;
  }

  function boxRect(box) {
    const x1 = Math.min(box.x, box.x + box.w);
    const x2 = Math.max(box.x, box.x + box.w);
    const y1 = Math.min(box.y, box.y + box.h);
    const y2 = Math.max(box.y, box.y + box.h);
    return { x1, y1, x2, y2, w: x2 - x1, h: y2 - y1 };
  }

  function setBoxRect(box, rect) {
    const minSize = 0.005;
    let x1 = Math.min(rect.x1, rect.x2);
    let x2 = Math.max(rect.x1, rect.x2);
    let y1 = Math.min(rect.y1, rect.y2);
    let y2 = Math.max(rect.y1, rect.y2);
    if (x2 - x1 < minSize) x2 = x1 + minSize;
    if (y2 - y1 < minSize) y2 = y1 + minSize;
    if (x1 < 0) {
      x2 -= x1;
      x1 = 0;
    }
    if (y1 < 0) {
      y2 -= y1;
      y1 = 0;
    }
    if (x2 > 1) {
      x1 -= x2 - 1;
      x2 = 1;
    }
    if (y2 > 1) {
      y1 -= y2 - 1;
      y2 = 1;
    }
    x1 = Math.max(0, Math.min(1 - minSize, x1));
    y1 = Math.max(0, Math.min(1 - minSize, y1));
    x2 = Math.max(x1 + minSize, Math.min(1, x2));
    y2 = Math.max(y1 + minSize, Math.min(1, y2));
    box.x = x1;
    box.y = y1;
    box.w = x2 - x1;
    box.h = y2 - y1;
    box.nobbox = false;
  }

  function resizeHandleAt(point, box) {
    if (!box || box.nobbox) return "";
    const rect = boxRect(box);
    const logicalW = canvasLogicalWidth();
    const logicalH = canvasLogicalHeight();
    const threshold = Math.max(
      0.008,
      Math.min(0.035, 8 / Math.max(1, Math.min(logicalW, logicalH))),
    );
    const nearLeft = Math.abs(point.x - rect.x1) <= threshold;
    const nearRight = Math.abs(point.x - rect.x2) <= threshold;
    const nearTop = Math.abs(point.y - rect.y1) <= threshold;
    const nearBottom = Math.abs(point.y - rect.y2) <= threshold;
    const insideX = point.x >= rect.x1 - threshold && point.x <= rect.x2 + threshold;
    const insideY = point.y >= rect.y1 - threshold && point.y <= rect.y2 + threshold;
    const vertical = nearTop && insideX ? "n" : nearBottom && insideX ? "s" : "";
    const horizontal = nearLeft && insideY ? "w" : nearRight && insideY ? "e" : "";
    return vertical + horizontal;
  }

  function cursorForMode(mode) {
    if (mode === "n" || mode === "s") return "ns-resize";
    if (mode === "e" || mode === "w") return "ew-resize";
    if (mode === "ne" || mode === "sw") return "nesw-resize";
    if (mode === "nw" || mode === "se") return "nwse-resize";
    if (mode === "move") return "move";
    return "crosshair";
  }

  function updateCursor(event) {
    if (dragMode) {
      canvas.style.cursor = cursorForMode(dragMode);
      return;
    }
    const point = canvasPoint(event);
    const hit = boxAt(point);
    if (hit >= 0) {
      canvas.style.cursor = cursorForMode(resizeHandleAt(point, boxes[hit]) || "move");
    } else {
      canvas.style.cursor = "crosshair";
    }
  }

  function syncGeometryChange() {
    const size = Array.isArray(node.size) ? [...node.size] : null;
    serialize();
    renderForm();
    draw();
    updateCount();
    preserveNodeSize(size);
  }

  function draw() {
    const isPrivateMasked = privacyEnabled() && !privacyReveal;
    const width = canvasLogicalWidth();
    const height = canvasLogicalHeight();
    const dpr = window.devicePixelRatio || 1;
    const backingWidth = Math.max(1, Math.round(width * dpr));
    const backingHeight = Math.max(1, Math.round(height * dpr));
    if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
      canvas.width = backingWidth;
      canvas.height = backingHeight;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = HELTO.bg;
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = HELTO.borderStrong;
    ctx.lineWidth = 1;
    for (let i = 1; i < 3; i++) {
      const gridX = Math.round((width * i) / 3) + 0.5;
      const gridY = Math.round((height * i) / 3) + 0.5;
      ctx.beginPath();
      ctx.moveTo(gridX, 0);
      ctx.lineTo(gridX, height);
      ctx.moveTo(0, gridY);
      ctx.lineTo(width, gridY);
      ctx.stroke();
    }
    boxes.forEach((box, index) => {
      if (box.nobbox) return;
      const color = normalizedColor((box.palette || [])[0]);
      const selected = index === active;
      const outlineColor = selected ? ACTIVE_COLOR : color;
      const x = box.x * width;
      const y = box.y * height;
      const w = box.w * width;
      const h = box.h * height;
      ctx.strokeStyle = outlineColor;
      ctx.lineWidth = selected ? 3 : 2;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = color + "33";
      ctx.fillRect(x, y, w, h);
      ctx.fillStyle = color;
      ctx.fillRect(x, y, 24, 18);
      if (selected) {
        const handle = 7;
        const points = [
          [x, y],
          [x + w, y],
          [x, y + h],
          [x + w, y + h],
        ];
        ctx.fillStyle = ACTIVE_COLOR;
        ctx.strokeStyle = HELTO.bg;
        ctx.lineWidth = 1;
        for (const [hx, hy] of points) {
          ctx.fillRect(hx - handle / 2, hy - handle / 2, handle, handle);
          ctx.strokeRect(hx - handle / 2, hy - handle / 2, handle, handle);
        }
      }
      if (!isPrivateMasked) {
        ctx.fillStyle = HELTO.bg;
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(index + 1).padStart(2, "0"), x + 12, y + 9);
      }
    });
  }

  function renderList() {
    list.innerHTML = "";
    boxes.forEach((box, index) => {
      const row = document.createElement("div");
      row.className = "aio-ideo-item" + (index === active ? " active" : "");
      const swatch = document.createElement("div");
      swatch.className = "aio-ideo-swatch";
      swatch.style.background = normalizedColor((box.palette || [])[0]);
      const label = document.createElement("span");
      label.textContent = `${index + 1}. ${box.type === "text" ? "text" : "obj"} ${box.desc || box.text || ""}`;
      const up = document.createElement("button");
      up.textContent = "↑";
      up.title = "Move region up";
      up.disabled = index === 0;
      row.append(swatch, label, up);
      row.addEventListener("click", () => {
        active = index;
        refresh();
      });
      up.addEventListener("click", (event) => {
        event.stopPropagation();
        if (index <= 0) return;
        const item = boxes.splice(index, 1)[0];
        boxes.splice(index - 1, 0, item);
        active = index - 1;
        serialize();
        refresh();
      });
      list.appendChild(row);
    });
  }

  function renderForm() {
    const box = boxes[active];
    const hasBox = !!box;
    typeSelect.disabled = !hasBox;
    textInput.disabled = !hasBox;
    descInput.disabled = !hasBox;
    addColor.disabled = !hasBox;
    removeBtn.disabled = !hasBox;
    typeSelect.value = box?.type || "obj";
    textInput.value = box?.text || "";
    descInput.value = box?.desc || "";
    textRow.style.display = box?.type === "text" ? "" : "none";
    bboxInputs.forEach((input, index) => {
      input.disabled = !hasBox || box?.nobbox;
      if (!box || box.nobbox) {
        input.value = "";
        return;
      }
      const bbox = normBBox(box);
      input.value = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]][index];
    });
    stylePaletteRow.innerHTML = "";
    for (const color of stylePalette) {
      const input = document.createElement("input");
      input.type = "color";
      input.value = normalizedColor(color);
      input.className = "aio-ideo-swatch";
      input.title = "Style palette color";
      input.addEventListener("input", () => {
        const idx = Array.from(stylePaletteRow.children).indexOf(input);
        stylePalette[idx] = input.value.toUpperCase();
        serialize();
        updateCount();
      });
      stylePaletteRow.appendChild(input);
    }
    paletteRow.innerHTML = "";
    if (box) {
      for (const color of box.palette || []) {
        const input = document.createElement("input");
        input.type = "color";
        input.value = normalizedColor(color);
        input.className = "aio-ideo-swatch";
        input.addEventListener("input", () => {
          const idx = Array.from(paletteRow.children).indexOf(input);
          box.palette[idx] = input.value.toUpperCase();
          serialize();
          draw();
          renderList();
        });
        paletteRow.appendChild(input);
      }
    }
  }

  function refresh() {
    const size = Array.isArray(node.size) ? [...node.size] : null;
    serialize();
    renderList();
    renderForm();
    draw();
    updateCount();
    preserveNodeSize(size);
  }

  function addBox(type = "obj") {
    boxes.push({
      x: 0.18 + (boxes.length % 4) * 0.04,
      y: 0.18 + (boxes.length % 4) * 0.04,
      w: 0.34,
      h: 0.24,
      type,
      text: "",
      desc: "",
      palette: [],
    });
    active = boxes.length - 1;
    refresh();
  }

  canvas.addEventListener("pointerdown", (event) => {
    stopEvent(event);
    const point = canvasPoint(event);
    const hit = boxAt(point);
    if (hit >= 0) {
      active = hit;
      dragMode = resizeHandleAt(point, boxes[hit]) || "move";
      dragStart = point;
      dragBoxStart = { ...boxes[hit] };
      canvas.setPointerCapture?.(event.pointerId);
      refresh();
      return;
    }
    dragMode = "draw";
    dragStart = point;
    boxes.push({
      x: point.x,
      y: point.y,
      w: 0,
      h: 0,
      type: "obj",
      text: "",
      desc: "",
      palette: [],
    });
    active = boxes.length - 1;
    dragBoxStart = { ...boxes[active] };
    canvas.setPointerCapture?.(event.pointerId);
    refresh();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragMode || active < 0) {
      updateCursor(event);
      return;
    }
    stopEvent(event);
    const point = canvasPoint(event);
    const box = boxes[active];
    if (dragMode === "draw") {
      box.w = point.x - dragStart.x;
      box.h = point.y - dragStart.y;
      syncGeometryChange();
      return;
    }
    if (dragMode === "move") {
      const dx = point.x - dragStart.x;
      const dy = point.y - dragStart.y;
      const rect = boxRect(dragBoxStart);
      setBoxRect(box, {
        x1: rect.x1 + dx,
        y1: rect.y1 + dy,
        x2: rect.x2 + dx,
        y2: rect.y2 + dy,
      });
      syncGeometryChange();
      return;
    }
    const rect = boxRect(dragBoxStart);
    const next = { ...rect };
    if (dragMode.includes("n")) next.y1 = point.y;
    if (dragMode.includes("s")) next.y2 = point.y;
    if (dragMode.includes("w")) next.x1 = point.x;
    if (dragMode.includes("e")) next.x2 = point.x;
    setBoxRect(box, next);
    syncGeometryChange();
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!dragMode) return;
    stopEvent(event);
    const wasDrawing = dragMode === "draw";
    dragMode = null;
    dragStart = null;
    dragBoxStart = null;
    const box = boxes[active];
    if (box && wasDrawing && Math.abs(box.w) < 0.01 && Math.abs(box.h) < 0.01) {
      box.w = 0.25;
      box.h = 0.18;
    }
    if (box && !box.nobbox) setBoxRect(box, boxRect(box));
    canvas.releasePointerCapture?.(event.pointerId);
    refresh();
  });
  canvas.addEventListener("pointerleave", (event) => {
    if (!dragMode) canvas.style.cursor = "crosshair";
    else updateCursor(event);
  });

  addObjBtn.addEventListener("click", () => addBox("obj"));
  addTextBtn.addEventListener("click", () => addBox("text"));
  libraryBtn.addEventListener("click", () => showPromptLibrary());
  saveLibraryBtn.addEventListener("click", async () => {
    const linkedId = libraryItemId();
    if (!linkedId) {
      showPromptLibrary({ openSave: true });
      return;
    }
    if (!window.confirm("Update the linked saved prompt? Choose Cancel to save as a new prompt.")) {
      showPromptLibrary({ openSave: true });
      return;
    }
    try {
      await saveCurrentPromptToLibrary({ itemId: linkedId });
    } catch (error) {
      setStatus(`Prompt library save failed: ${error.message}`);
    }
  });
  removeBtn.addEventListener("click", () => {
    if (active < 0) return;
    boxes.splice(active, 1);
    active = Math.min(active, boxes.length - 1);
    refresh();
  });
  clearBtn.addEventListener("click", () => {
    boxes = [];
    stylePalette = [];
    active = -1;
    refresh();
  });
  copyBtn.addEventListener("click", async () => {
    await navigator.clipboard?.writeText(captionText());
  });
  pasteBtn.addEventListener("click", async () => {
    const text = window.prompt("Paste Ideogram 4 JSON:", (await navigator.clipboard?.readText?.()) || "");
    if (!text) return;
    try {
      const caption = JSON.parse(text);
      const [width, height] = resolveDims(node);
      boxes = captionToBoxes(caption, {
        coordMode: coordModeWidget?.value === "absolute" ? "absolute" : "normalized",
        bboxOrder: bboxOrderWidget?.value === "xy" ? "xy" : "yx",
        width,
        height,
      });
      active = boxes.length ? 0 : -1;
      const cd = caption.compositional_deconstruction || {};
      widgetByName(node, "background").value = cd.background || "";
      if (caption.high_level_description != null) widgetByName(node, "high_level_description").value = caption.high_level_description;
      refresh();
    } catch {
      window.alert("That is not valid Ideogram 4 JSON.");
    }
  });
  compact.addEventListener("change", () => {
    if (outputFormatWidget) outputFormatWidget.value = compact.checked ? "compact" : "pretty";
    serialize();
    updateCount();
  });
  absolute.addEventListener("change", () => {
    if (coordModeWidget) coordModeWidget.value = absolute.checked ? "absolute" : "normalized";
    serialize();
    updateCount();
  });
  xyOrder.addEventListener("change", () => {
    if (bboxOrderWidget) bboxOrderWidget.value = xyOrder.checked ? "xy" : "yx";
    serialize();
    updateCount();
  });
  typeSelect.addEventListener("change", () => {
    if (boxes[active]) boxes[active].type = typeSelect.value;
    refresh();
  });
  textInput.addEventListener("input", () => {
    if (boxes[active]) boxes[active].text = textInput.value;
    refresh();
  });
  descInput.addEventListener("input", () => {
    if (boxes[active]) boxes[active].desc = descInput.value;
    refresh();
  });
  addColor.addEventListener("click", () => {
    if (!boxes[active]) return;
    boxes[active].palette ||= [];
    boxes[active].palette.push("#FFFFFF");
    refresh();
  });
  addStyleColor.addEventListener("click", () => {
    stylePalette.push("#FFFFFF");
    refresh();
  });
  bboxInputs.forEach((input, index) => {
    input.addEventListener("input", () => {
      const box = boxes[active];
      if (!box) return;
      const values = bboxInputs.map((item) => Math.max(0, Math.min(1000, Number(item.value || 0))));
      box.x = values[0] / 1000;
      box.y = values[1] / 1000;
      box.w = values[2] / 1000;
      box.h = values[3] / 1000;
      box.nobbox = false;
      serialize();
      draw();
      updateCount();
      if (index < 2) renderList();
    });
  });

  for (const name of [
    "max side",
    "aspect ratio",
    "multiple value",
    PRIVACY_WIDGET_NAME,
    "background",
    "style",
    "photo",
    "art_style",
    "aesthetics",
    "lighting",
    "medium",
    "high_level_description",
    "import_mode",
    "output_format",
    "bg_brightness",
    "import_json",
  ]) {
    const widget = widgetByName(node, name);
    if (!widget || widget._aioIdeoPatched) continue;
    const original = widget.callback;
    widget.callback = function () {
      if (syncingExecutionWidgets) return undefined;
      const result = original?.apply(this, arguments);
      fitCanvas();
      serialize();
      updateCount();
      return result;
    };
    widget._aioIdeoPatched = true;
  }

  const observer = new ResizeObserver(() => fitCanvas());
  observer.observe(canvasBox);
  const originalRemoved = node.onRemoved;
  node.onRemoved = function () {
    observer.disconnect();
    cleanupSensitiveWidgetDomPrivacy();
    return originalRemoved?.apply(this, arguments);
  };

  const originalResize = node.onResize;
  node.onResize = function () {
    const result = originalResize?.apply(this, arguments);
    syncEditorSize({ fromNodeResize: true });
    return result;
  };

  requestAnimationFrame(() => {
    if (node.size?.[0] < MIN_NODE_WIDTH) {
      node.setSize?.([MIN_NODE_WIDTH, Math.max(node.size?.[1] || 0, EDITOR_INITIAL_NODE_HEIGHT)]);
    }
    syncEditorSize({ fromNodeResize: true });
    fitCanvas();
    updatePrivacyClasses();
    updateLibraryButtons();
    refresh();
    restoreEncryptedState();
  });
  return wrap;
}

app.registerExtension({
  name: "AIO.Ideogram4PromptBuilder",
  setup() {
    registerAioPrivacyRecoveryDescriptors();
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;
    if (!nodeType.prototype._aioIdeogram4WorkflowPatched) {
      nodeType.prototype._aioIdeogram4WorkflowPatched = true;

      const originalConfigure = nodeType.prototype.configure;
      nodeType.prototype.configure = function (info) {
        originalConfigure?.apply(this, arguments);
        applyHeltoNodeTheme(this);
        this._aioIdeogram4PendingWorkflowInfo = info;
        this._aioIdeogram4EditorApi?.restoreFromWorkflow?.(info);
      };

      const originalOnSerialize = nodeType.prototype.onSerialize;
      nodeType.prototype.onSerialize = function (output) {
        const result = originalOnSerialize?.apply(this, arguments);
        this._aioIdeogram4EditorApi?.serializeForWorkflow?.(output);
        return result;
      };

      const originalMenu = nodeType.prototype.getExtraMenuOptions;
      nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
        originalMenu?.apply(this, arguments);
        appendPrivacyRecoveryMenuOption(this, options);
      };
    }

    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      applyHeltoNodeTheme(this);
      this.resizable = true;
      const editor = createEditor(this);
      if (this._aioIdeogram4PendingWorkflowInfo) {
        this._aioIdeogram4EditorApi?.restoreFromWorkflow?.(this._aioIdeogram4PendingWorkflowInfo);
      }
      const domWidget = this.addDOMWidget("aio_ideogram4_prompt_builder", "AIOIdeogram4PromptBuilder", editor, {
        serialize: false,
        hideOnZoom: false,
        margin: EDITOR_NODE_MARGIN,
        getMinHeight: () => editorWidgetHeight(EDITOR_MIN_HEIGHT),
        getMaxHeight: () => editorWidgetHeight(this._aioIdeogram4EditorHeight?.() ?? EDITOR_HEIGHT),
        getHeight: () => editorWidgetHeight(this._aioIdeogram4EditorHeight?.() ?? EDITOR_HEIGHT),
      });
      this._aioIdeogram4SetDomWidget?.(
        domWidget || this.widgets?.find((widget) => widget.name === "aio_ideogram4_prompt_builder"),
      );
    };
  },
});
