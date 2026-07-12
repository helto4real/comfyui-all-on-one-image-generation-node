// Product-facing facade over the inactive shared private-record handle.

export const AIO_PROMPT_LIBRARY_RESOURCE_ID = "ideogram-prompts";
export const AIO_PROMPT_RECORD_KIND = "ideogram-prompt";

function fail() {
  throw new Error("PRIVACY_AIO_PROMPT_LIBRARY_INVALID");
}

function receipt(value, operation) {
  if (!value || value.kind !== AIO_PROMPT_RECORD_KIND || value.operation !== operation
      || typeof value.recordId !== "string" || !value.recordId.startsWith("hp-rec-")) fail();
  return Object.freeze({ ...value });
}

function revealed(value, recordId) {
  const record = value?.value?.record;
  if (!record || typeof record !== "object" || Array.isArray(record)) fail();
  return Object.freeze({ id: recordId, ...structuredClone(record) });
}

function productMetadata(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    ["name", "description", "tags"]
      .filter((key) => Object.prototype.hasOwnProperty.call(value, key))
      .map((key) => [key, structuredClone(value[key])]),
  );
}

export function createAioManagedPromptLibrary({ recordsHandle } = {}) {
  if (!recordsHandle) fail();
  return Object.freeze({
    async list() {
      const shells = await recordsHandle.list(AIO_PROMPT_RECORD_KIND);
      if (!Array.isArray(shells) || shells.some((item) => (
        !item || Object.keys(item).sort().join(",") !== "id,kind,label,private"
        || item.kind !== AIO_PROMPT_RECORD_KIND
        || item.private !== true
        || item.label !== "Private record"
      ))) fail();
      return Object.freeze(shells.map((item) => Object.freeze({ ...item })));
    },
    async create(payload, metadata = {}) {
      return receipt(
        await recordsHandle.create(AIO_PROMPT_RECORD_KIND, {
          payload,
          metadata: productMetadata(metadata),
        }),
        "create",
      );
    },
    async details(recordId) {
      return revealed(
        await recordsHandle.reveal(AIO_PROMPT_RECORD_KIND, recordId, "details"),
        recordId,
      );
    },
    async use(recordId) {
      return revealed(
        await recordsHandle.reveal(AIO_PROMPT_RECORD_KIND, recordId, "use"),
        recordId,
      );
    },
    async replace(recordId, payload, metadata = {}) {
      return receipt(
        await recordsHandle.mutate(
          AIO_PROMPT_RECORD_KIND,
          recordId,
          "replace",
          { payload, metadata: productMetadata(metadata) },
        ),
        "replace",
      );
    },
    async patch(recordId, { payload, metadata } = {}) {
      const value = { metadata: productMetadata(metadata) };
      if (payload !== undefined) value.payload = payload;
      return receipt(
        await recordsHandle.mutate(
          AIO_PROMPT_RECORD_KIND,
          recordId,
          "patch",
          value,
        ),
        "patch",
      );
    },
    async duplicate(recordId, metadata = {}) {
      return receipt(
        await recordsHandle.mutate(
          AIO_PROMPT_RECORD_KIND,
          recordId,
          "duplicate",
          { metadata: productMetadata(metadata) },
        ),
        "duplicate",
      );
    },
    delete(recordId) {
      return recordsHandle.delete(AIO_PROMPT_RECORD_KIND, recordId);
    },
  });
}
