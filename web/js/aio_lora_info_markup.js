export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function escapedValueMarkup(value) {
  return `<span>${escapeHtml(value)}</span>`;
}

export function imageInfoFieldMarkup(label, value) {
  if (value == null) {
    return "";
  }
  return `<span>${label ? `<label>${escapeHtml(label)} </label>` : ""}${escapeHtml(value)}</span>`;
}

export function safeUrl(value, { allowRelative = false } = {}) {
  const candidate = String(value ?? "").trim();
  if (!candidate) {
    return "";
  }
  if (allowRelative && candidate.startsWith("/") && !candidate.startsWith("//")) {
    return candidate;
  }
  try {
    const parsed = new URL(candidate);
    return parsed.protocol === "https:" || parsed.protocol === "http:" ? candidate : "";
  } catch {
    return "";
  }
}
