// Helto Design System — canonical design tokens, inlined for self-styled
// ComfyUI widgets that inject their own <style> (no shared stylesheet).
// Values are copied verbatim from helto-designsystem/reference/tokens.css.
// Dark-navy surfaces, a GOLD accent for selection/active, a BLUE focus ring.
// Keep this in sync with the design system; do not introduce new colors here.

export const HELTO_TOKENS_CSS = `
:root {
  /* ---- Surfaces: deepest inset -> most raised ---- */
  --helto-bg:            #0d1320;
  --helto-surface:       #151c2a;
  --helto-surface-2:     #1b2333;
  --helto-surface-3:     #232d3f;
  --helto-surface-hover: #2c3850;

  /* ---- Borders ---- */
  --helto-border:        #2a3346;
  --helto-border-strong: #3a465c;
  --helto-border-hover:  #4c5970;

  /* ---- Text (3 tiers) ---- */
  --helto-text:          #e7ebf3;
  --helto-text-dim:      #9aa6bd;
  --helto-text-faint:    #6f7c95;

  /* ---- Accent (GOLD) — selection / active emphasis ---- */
  --helto-accent:        #f1c75c;
  --helto-accent-strong: #ffd873;
  --helto-accent-bg:     rgba(241, 199, 92, 0.16);
  --helto-accent-border: rgba(241, 199, 92, 0.55);

  /* ---- Focus (BLUE) — keyboard/focus only, never selection ---- */
  --helto-focus:         #5e9bff;
  --helto-focus-ring:    0 0 0 2px rgba(94, 155, 255, 0.5);

  /* ---- Danger / destructive ---- */
  --helto-danger:        #ec5a6b;
  --helto-danger-bg:     #3a1a22;
  --helto-danger-border: #8f3a44;

  /* ---- Status accents (pills) ---- */
  --helto-ok:            #baf0c8;
  --helto-warn:          #ffe3a3;
  --helto-info:          #b9dafc;

  /* ---- Radii ---- */
  --helto-radius-sm:     5px;
  --helto-radius:        6px;
  --helto-radius-lg:     10px;

  /* ---- Typography ---- */
  --helto-font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
  --helto-font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                     "Roboto Mono", monospace;
  --helto-font-size: 12px;
  --helto-line: 1.4;

  /* ---- Elevation ---- */
  --helto-shadow:      0 1px 2px rgba(0, 0, 0, 0.35);
  --helto-shadow-pop:  0 14px 36px rgba(0, 0, 0, 0.55);
  --helto-shadow-glow: 0 0 10px rgba(241, 199, 92, 0.35);

  /* ---- Motion ---- */
  --helto-transition: 0.12s ease;
  --helto-ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}
`;

// Raw token literal values, for canvas drawing (where var(--helto-*) cannot
// be used). Kept identical to the CSS custom properties above.
export const HELTO = {
  bg: "#0d1320",
  surface: "#151c2a",
  surface2: "#1b2333",
  surface3: "#232d3f",
  surfaceHover: "#2c3850",
  border: "#2a3346",
  borderStrong: "#3a465c",
  borderHover: "#4c5970",
  text: "#e7ebf3",
  textDim: "#9aa6bd",
  textFaint: "#6f7c95",
  accent: "#f1c75c",
  accentStrong: "#ffd873",
  focus: "#5e9bff",
  danger: "#ec5a6b",
  warn: "#ffe3a3",
  ok: "#baf0c8",
  info: "#b9dafc",
};

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
