export const PRIVACY_SCHEMA = "helto.aio-image-generate.v2";
export const LEGACY_PRIVACY_SCHEMA = "helto.aio-image-generate";

const ROUTE_PREFIX = "/aio_image_generate/privacy";
const SHARED_PRIVACY_ROUTE = "/helto_privacy/ui/privacy.js";
const PRIVACY_TOKEN_HEADER = "X-Helto-Privacy-Token";
const PRIVACY_TOKEN_STORAGE_KEY = "helto_privacy_token";
const LEGACY_MESSAGE = "Unsupported legacy AIO privacy payload. Re-enter the private value to save it with the shared privacy keystore.";

let privacyModulePromise = null;

export function parsePrivacyPayload(value) {
  if (!value) return null;
  if (typeof value === "object") return value;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

export function isEncryptedPrivacyPayload(value) {
  const parsed = parsePrivacyPayload(value);
  return Boolean(parsed?.encrypted === true && parsed.schema === PRIVACY_SCHEMA && parsed.algorithm === "AES-256-GCM");
}

export function isLegacyPrivacyPayload(value) {
  const parsed = parsePrivacyPayload(value);
  return Boolean(parsed?.encrypted === true && parsed.schema === LEGACY_PRIVACY_SCHEMA && parsed.algorithm === "AES-256-GCM");
}

export function isAnyAioPrivacyPayload(value) {
  return isEncryptedPrivacyPayload(value) || isLegacyPrivacyPayload(value);
}

export function assertSupportedPrivacyPayload(value) {
  if (isLegacyPrivacyPayload(value)) throw new Error(LEGACY_MESSAGE);
}

export function getStoredPrivacyToken() {
  try {
    return globalThis.localStorage?.getItem(PRIVACY_TOKEN_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export async function getSharedPrivacyUi() {
  if (!privacyModulePromise) {
    privacyModulePromise = import(SHARED_PRIVACY_ROUTE).catch(() => null);
  }
  return privacyModulePromise;
}

export function privacyFetchHeaders(headers = {}) {
  const next = { ...headers };
  const token = getStoredPrivacyToken();
  if (token) next[PRIVACY_TOKEN_HEADER] = token;
  return next;
}

async function maybeUnlockForError(error) {
  const privacy = await getSharedPrivacyUi();
  if (!privacy?.isPrivacyLockedError?.(error)) return false;
  const result = await privacy.showPrivacyKeystoreDialog?.("auto");
  privacy.ensureStoredPrivacyTokenCookie?.();
  return Boolean(result);
}

export async function fetchPrivacyJson(endpoint, payload = null, retry = true) {
  const privacy = await getSharedPrivacyUi();
  privacy?.ensureStoredPrivacyTokenCookie?.();
  const options = payload
    ? { method: "POST", headers: privacyFetchHeaders({ "Content-Type": "application/json" }), body: JSON.stringify(payload) }
    : undefined;
  const response = await fetch(`${ROUTE_PREFIX}/${endpoint}`, options);
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text || response.statusText || `HTTP ${response.status}`);
  }
  if (!response.ok || data.ok === false || data.error) {
    const error = new Error(data.error || response.statusText || `HTTP ${response.status}`);
    if (retry && await maybeUnlockForError(error)) return fetchPrivacyJson(endpoint, payload, false);
    throw error;
  }
  return data;
}

export function encryptStateSync(state) {
  if (typeof XMLHttpRequest !== "function") {
    throw new Error("Synchronous privacy encryption is unavailable in this environment.");
  }
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `${ROUTE_PREFIX}/encrypt`, false);
  xhr.setRequestHeader("Content-Type", "application/json");
  const token = getStoredPrivacyToken();
  if (token) xhr.setRequestHeader(PRIVACY_TOKEN_HEADER, token);
  xhr.send(JSON.stringify({ state }));
  let data = {};
  try {
    data = xhr.responseText ? JSON.parse(xhr.responseText) : {};
  } catch {
    throw new Error(xhr.responseText || xhr.statusText || `HTTP ${xhr.status}`);
  }
  if (xhr.status < 200 || xhr.status >= 300 || data.ok === false || data.error) {
    throw new Error(data.error || xhr.statusText || `HTTP ${xhr.status}`);
  }
  return data.envelope;
}

export function encryptValueSync(value) {
  assertSupportedPrivacyPayload(value);
  return JSON.stringify(encryptStateSync({ value }));
}

export async function decryptState(payload) {
  assertSupportedPrivacyPayload(payload);
  const data = await fetchPrivacyJson("decrypt", { payload });
  return data.state || {};
}

export async function decryptValue(value) {
  if (isLegacyPrivacyPayload(value)) throw new Error(LEGACY_MESSAGE);
  if (!isEncryptedPrivacyPayload(value)) return value;
  const state = await decryptState(parsePrivacyPayload(value));
  return state.value ?? "";
}
