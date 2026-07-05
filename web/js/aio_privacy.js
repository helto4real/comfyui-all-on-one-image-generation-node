export const PRIVACY_SCHEMA = "helto.aio-image-generate.v2";
export const LEGACY_PRIVACY_SCHEMA = "helto.aio-image-generate";

const ROUTE_PREFIX = "/aio_image_generate/privacy";
const SHARED_PRIVACY_ROUTE = "/helto_privacy/ui/privacy.js";
const PRIVACY_TOKEN_HEADER = "X-Helto-Privacy-Token";
const PRIVACY_TOKEN_STORAGE_KEY = "helto_privacy_token";
const LEGACY_MESSAGE = "Unsupported legacy AIO privacy payload. Re-enter the private value to save it with the shared privacy keystore.";
const UNSUPPORTED_MESSAGE = "Unsupported AIO privacy payload schema. Use Privacy Recovery to reset or re-enter the private value.";
const PRIVACY_UNLOCK_CODES = ["PRIVACY_LOCKED", "PRIVACY_TOKEN_REQUIRED", "PRIVACY_KEYSTORE_UNINITIALIZED"];

let privacyModulePromise = null;
const failedEnvelopeFingerprints = new Set();
let recoveryDialogTimer = null;

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

export function isEncryptedLookingPrivacyPayload(value) {
  const parsed = parsePrivacyPayload(value);
  return Boolean(
    parsed &&
      typeof parsed === "object" &&
      (
        parsed.encrypted === true ||
        parsed.algorithm === "AES-256-GCM" ||
        ("ciphertext" in parsed && "nonce" in parsed) ||
        "keyId" in parsed
      )
  );
}

export function isUnsupportedEncryptedPrivacyPayload(value) {
  return isEncryptedLookingPrivacyPayload(value) && !isEncryptedPrivacyPayload(value);
}

export function isAnyAioPrivacyPayload(value) {
  return isEncryptedPrivacyPayload(value) || isLegacyPrivacyPayload(value);
}

export function assertSupportedPrivacyPayload(value) {
  if (isLegacyPrivacyPayload(value)) {
    schedulePrivacyRecoveryDialog("legacy");
    throw new Error(LEGACY_MESSAGE);
  }
  if (isUnsupportedEncryptedPrivacyPayload(value)) {
    schedulePrivacyRecoveryDialog("unsupported");
    throw new Error(UNSUPPORTED_MESSAGE);
  }
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

export function isPrivacyUnlockRequiredError(error) {
  const message = String(error?.message ?? error ?? "");
  return PRIVACY_UNLOCK_CODES.some((code) => message.includes(code));
}

async function maybeUnlockForError(error) {
  const privacy = await getSharedPrivacyUi();
  const unlockRequired = Boolean(
    privacy?.isPrivacyUnlockRequiredError?.(error) ||
      privacy?.isPrivacyLockedError?.(error) ||
      isPrivacyUnlockRequiredError(error)
  );
  if (!unlockRequired) return false;
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

export async function encryptState(state) {
  const data = await fetchPrivacyJson("encrypt", { state });
  return data.envelope;
}

export function encryptValueSync(value) {
  assertSupportedPrivacyPayload(value);
  return JSON.stringify(encryptStateSync({ value }));
}

export async function encryptValue(value) {
  assertSupportedPrivacyPayload(value);
  return JSON.stringify(await encryptState({ value }));
}

export async function ensureEncryptedPrivacyValue(options = {}) {
  const privacy = await getSharedPrivacyUi();
  if (privacy?.ensureEncryptedPrivacyValue) {
    return privacy.ensureEncryptedPrivacyValue({
      schema: PRIVACY_SCHEMA,
      encrypt: async (value) => encryptValue(value),
      ...options,
    });
  }
  if (options.privacyMode === false) return String(options.value ?? "");
  return encryptValue(options.value ?? "");
}

export async function decryptState(payload) {
  assertSupportedPrivacyPayload(payload);
  try {
    const data = await fetchPrivacyJson("decrypt", { payload });
    forgetFailedPrivacyEnvelope(payload);
    return data.state || {};
  } catch (error) {
    if (isEncryptedPrivacyPayload(payload) && !isPrivacyUnlockRequiredError(error)) {
      rememberFailedPrivacyEnvelope(payload);
      schedulePrivacyRecoveryDialog("decrypt-failed");
    }
    throw error;
  }
}

export async function decryptValue(value) {
  if (isLegacyPrivacyPayload(value)) {
    schedulePrivacyRecoveryDialog("legacy");
    throw new Error(LEGACY_MESSAGE);
  }
  if (isUnsupportedEncryptedPrivacyPayload(value)) {
    schedulePrivacyRecoveryDialog("unsupported");
    throw new Error(UNSUPPORTED_MESSAGE);
  }
  if (!isEncryptedPrivacyPayload(value)) return value;
  const state = await decryptState(parsePrivacyPayload(value));
  return state.value ?? "";
}

export function privacyEnvelopeFingerprint(value) {
  const payload = parsePrivacyPayload(value);
  if (!payload || typeof payload !== "object") return "";
  return [
    payload.schema ?? "",
    payload.version ?? "",
    payload.algorithm ?? "",
    payload.keyId ?? "",
    payload.nonce ?? "",
    payload.ciphertext ?? "",
  ].map((part) => String(part)).join("|");
}

export function rememberFailedPrivacyEnvelope(value) {
  const fingerprint = privacyEnvelopeFingerprint(value);
  if (fingerprint) failedEnvelopeFingerprints.add(fingerprint);
}

export function forgetFailedPrivacyEnvelope(value) {
  const fingerprint = privacyEnvelopeFingerprint(value);
  if (fingerprint) failedEnvelopeFingerprints.delete(fingerprint);
}

export function isFailedPrivacyEnvelope(value) {
  const fingerprint = privacyEnvelopeFingerprint(value);
  return Boolean(fingerprint && failedEnvelopeFingerprints.has(fingerprint));
}

export function acceptsAioRecoveryEnvelope(value) {
  return isEncryptedPrivacyPayload(value) && !isFailedPrivacyEnvelope(value);
}

export async function showPrivacyRecoveryDialog(options = {}) {
  const privacy = await getSharedPrivacyUi();
  if (!privacy?.showPrivacyRecoveryDialog) return null;
  return privacy.showPrivacyRecoveryDialog(options);
}

export function schedulePrivacyRecoveryDialog(reason = "privacy") {
  if (recoveryDialogTimer) return;
  recoveryDialogTimer = setTimeout(async () => {
    recoveryDialogTimer = null;
    const privacy = await getSharedPrivacyUi();
    if (!privacy?.showPrivacyRecoveryDialog || privacy.isPrivacyRecoveryDialogOpen?.()) return;
    await privacy.showPrivacyRecoveryDialog({ mode: "auto", reason });
  }, 0);
}
