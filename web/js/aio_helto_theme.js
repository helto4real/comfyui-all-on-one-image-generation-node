// Helto Design System — canonical design tokens, inlined for self-styled
// ComfyUI widgets that inject their own <style> (no shared stylesheet).
// Values are copied verbatim from helto-design-system/assets/tokens.css.
// Catppuccin Mocha surfaces, a GOLD accent for selection/active, a BLUE focus ring.
// Keep this in sync with the design system; do not introduce new colors here.

export const HELTO_TOKENS_CSS = `
:root {
  /* ---- Surfaces (deepest → raised) ------------------------------------ */
  --helto-bg: #181825;            /* mantle — input wells, grids, viewports */
  --helto-surface: #1e1e2e;       /* base — panels */
  --helto-surface-2: #313244;     /* surface0 — fields, list items */
  --helto-surface-3: #45475a;     /* surface1 — button gradient top */
  --helto-surface-hover: #585b70; /* surface2 — hovered controls */

  /* ---- Borders ---------------------------------------------------------- */
  --helto-border: #313244;        /* surface0 */
  --helto-border-strong: #45475a; /* surface1 */
  --helto-border-hover: #6c7086;  /* overlay0 */

  /* ---- Text (three tiers only) ------------------------------------------ */
  --helto-text: #cdd6f4;          /* text */
  --helto-text-dim: #a6adc8;      /* subtext0 */
  --helto-text-faint: #7f849c;    /* overlay1 */

  /* ---- Accent (GOLD → mocha peach) — selection / active / primary ------- */
  --helto-accent: #fab387;        /* peach */
  --helto-accent-strong: #fddcc4; /* peach, lightened */
  --helto-accent-border: #93664a; /* peach, darkened */
  --helto-accent-bg: #46301f;     /* peach well (switch track on) */

  /* ---- Focus (BLUE) — focus rings only ----------------------------------- */
  --helto-focus: #89b4fa;         /* blue */
  --helto-focus-ring: 0 0 0 3px rgba(137, 180, 250, 0.28);

  /* ---- Status ------------------------------------------------------------ */
  --helto-danger: #f38ba8;        /* red */
  --helto-danger-border: #96526a; /* red, darkened */
  --helto-ok: #a6e3a1;            /* green */
  --helto-warn: #f9e2af;          /* yellow (peach is the accent) */
  --helto-info: #74c7ec;          /* sapphire */

  /* ---- Radii -------------------------------------------------------------- */
  --helto-radius-sm: 5px;  /* controls */
  --helto-radius: 6px;     /* panels, inputs */
  --helto-radius-lg: 10px; /* modals */

  /* ---- Elevation ----------------------------------------------------------- */
  --helto-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
  --helto-shadow-pop: 0 12px 32px rgba(0, 0, 0, 0.5);
  --helto-shadow-glow: 0 0 0 1px rgba(250, 179, 135, 0.35),
                       0 0 12px rgba(250, 179, 135, 0.22);

  /* ---- Motion --------------------------------------------------------------- */
  --helto-transition: 0.12s ease;
  --helto-ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);

  /* ---- Type ------------------------------------------------------------------ */
  --helto-font-sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --helto-font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --helto-font-size: 12px;
  --helto-line: 1.4;
}
`;

// Raw token literal values, for canvas drawing (where var(--helto-*) cannot
// be used). Kept identical to the CSS custom properties above.
export const HELTO = {
  bg: "#181825",
  surface: "#1e1e2e",
  surface2: "#313244",
  surface3: "#45475a",
  surfaceHover: "#585b70",
  border: "#313244",
  borderStrong: "#45475a",
  borderHover: "#6c7086",
  text: "#cdd6f4",
  textDim: "#a6adc8",
  textFaint: "#7f849c",
  accent: "#fab387",
  accentStrong: "#fddcc4",
  accentBorder: "#93664a",
  accentBg: "#46301f",
  focus: "#89b4fa",
  danger: "#f38ba8",
  dangerBorder: "#96526a",
  warn: "#f9e2af",
  ok: "#a6e3a1",
  info: "#74c7ec",
};

const HELTO_LITEGRAPH_WIDGET_THEME = {
  WIDGET_BGCOLOR: HELTO.bg,
  WIDGET_OUTLINE_COLOR: HELTO.borderStrong,
  WIDGET_PROMOTED_OUTLINE_COLOR: HELTO.accent,
  WIDGET_ADVANCED_OUTLINE_COLOR: HELTO.focus,
  WIDGET_TEXT_COLOR: HELTO.text,
  WIDGET_SECONDARY_TEXT_COLOR: HELTO.textDim,
  WIDGET_DISABLED_TEXT_COLOR: HELTO.textFaint,
};

export function applyHeltoNodeTheme(node) {
  if (!node || typeof node !== "object") {
    return false;
  }
  node.color = HELTO.surface3;
  node.bgcolor = HELTO.surface;
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
  return true;
}

export function applyHeltoLiteGraphWidgetTheme(liteGraph = globalThis.LiteGraph) {
  if (!liteGraph || typeof liteGraph !== "object") {
    return null;
  }
  const previous = {};
  for (const [key, value] of Object.entries(HELTO_LITEGRAPH_WIDGET_THEME)) {
    if (key in liteGraph) {
      previous[key] = liteGraph[key];
      liteGraph[key] = value;
    }
  }
  return Object.keys(previous).length ? { liteGraph, previous } : null;
}

export function restoreHeltoLiteGraphWidgetTheme(snapshot) {
  const { liteGraph, previous } = snapshot || {};
  if (!liteGraph || !previous) {
    return false;
  }
  for (const [key, value] of Object.entries(previous)) {
    liteGraph[key] = value;
  }
  return true;
}

export function withHeltoLiteGraphWidgetTheme(callback, liteGraph = globalThis.LiteGraph) {
  const snapshot = applyHeltoLiteGraphWidgetTheme(liteGraph);
  try {
    return callback?.();
  } finally {
    restoreHeltoLiteGraphWidgetTheme(snapshot);
  }
}

const TOKENS_STYLE_ID = "aio-helto-tokens";

// Inject the canonical :root token block once so any aio widget styled with
// var(--helto-*) resolves correctly wherever the UI renders.
export function ensureHeltoTokens() {
  if (typeof document === "undefined" || document.getElementById(TOKENS_STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = TOKENS_STYLE_ID;
  style.textContent = HELTO_TOKENS_CSS;
  // Prepend so component stylesheets (and ComfyUI defaults) can override layout
  // while still inheriting the token variables.
  document.head.prepend(style);
}
