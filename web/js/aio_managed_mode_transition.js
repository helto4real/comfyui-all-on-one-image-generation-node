// Exact browser-owned workflow transition mechanics shared by AIO adapters.

import { aioManagedNodeType } from "./aio_managed_privacy_graph.js";

const TRANSITION_FROZEN = "__aioManagedPrivacyTransitionFrozen";
const ENVELOPE_KEYS = Object.freeze([
  "algorithm", "ciphertext", "encrypted", "keyId", "nonce", "schema", "version",
]);

function graphValues(value) {
  if (value instanceof Map || value instanceof Set) return [...value.values()];
  if (Array.isArray(value)) return value;
  if (value && typeof value === "object") return Object.values(value);
  return [];
}

function stableLocatorPart(value, fail) {
  const normalized = String(value ?? "");
  if (!/^[A-Za-z0-9._~:-]{1,128}$/u.test(normalized)) fail();
  return normalized;
}

function exactBytes(value, fail) {
  if (value instanceof Uint8Array) return new Uint8Array(value);
  if (value instanceof ArrayBuffer) return new Uint8Array(value.slice(0));
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer.slice(
      value.byteOffset,
      value.byteOffset + value.byteLength,
    ));
  }
  fail();
}

function equalBytes(left, right) {
  return left.byteLength === right.byteLength
    && left.every((value, index) => value === right[index]);
}

function transitionField(context, fail) {
  const field = context?.field;
  if (
    !field
    || typeof field !== "object"
    || !/^[a-z0-9][a-z0-9._-]*$/u.test(String(field.id || ""))
    || !Array.isArray(field.nodeTypes)
    || !field.nodeTypes.length
    || !field.externalTransitionPolicy
  ) fail();
  return field;
}

function maximumBytes(field, key, fail) {
  const value = field.externalTransitionPolicy?.[key];
  if (!Number.isInteger(value) || value < 1024) fail();
  return value;
}

export function parseAioModeTransitionStorage(value, fail) {
  if (typeof value !== "string" || !value.trim()) fail();
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch {
    fail();
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) fail();
  return parsed;
}

export function isAioCurrentModeEnvelope(value, schema) {
  return value
    && typeof value === "object"
    && !Array.isArray(value)
    && Object.keys(value).sort().join("\0") === [...ENVELOPE_KEYS].sort().join("\0")
    && value.version === 1
    && value.schema === schema
    && value.encrypted === true
    && value.algorithm === "AES-256-GCM"
    && typeof value.keyId === "string"
    && value.keyId.length > 0
    && typeof value.nonce === "string"
    && value.nonce.length > 0
    && typeof value.ciphertext === "string"
    && value.ciphertext.length > 0;
}

export function createAioExternalWorkflowTransition({
  app = null,
  owners,
  registerNode,
  readStorage,
  writeStorage,
  readDetachedStorage,
  settleOwner = () => {},
  reloadRuntime,
  reconcileRuntime = () => {},
  fail,
}) {
  if (
    !(owners instanceof Set)
    || ![
      registerNode,
      readStorage,
      writeStorage,
      readDetachedStorage,
      settleOwner,
      reloadRuntime,
      reconcileRuntime,
      fail,
    ].every((item) => typeof item === "function")
  ) fail();

  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const expectedReadbacks = new WeakMap();
  let transitionDepth = 0;
  let internalMutationDepth = 0;

  function frozen() {
    return transitionDepth > 0;
  }

  function withInternalMutation(callback) {
    internalMutationDepth += 1;
    try {
      return callback();
    } finally {
      internalMutationDepth -= 1;
    }
  }

  function requireTransition() {
    if (!frozen()) fail();
  }

  function requireMutable() {
    if (frozen() && internalMutationDepth === 0) fail();
  }

  function synchronizeOwner(node) {
    node[TRANSITION_FROZEN] = frozen();
  }

  function liveGraphEntries() {
    const firstOwner = owners.values().next().value;
    const candidate = app?.rootGraph
      ?? app?.graph?.rootGraph
      ?? app?.graph
      ?? firstOwner?.graph?.rootGraph
      ?? firstOwner?.graph
      ?? null;
    if (!candidate) {
      return owners.size ? [{ graph: null, graphId: "root", nodes: [...owners] }] : [];
    }
    const root = candidate.rootGraph ?? candidate;
    const entries = [{ graph: root, graphId: "root", nodes: root?._nodes ?? root?.nodes ?? [] }];
    const seen = new Set([root]);
    const pending = [
      ...graphValues(root?._subgraphs),
      ...graphValues(root?.subgraphs),
    ];
    for (let index = 0; index < pending.length; index += 1) {
      const graph = pending[index];
      if (!graph || seen.has(graph)) continue;
      seen.add(graph);
      entries.push({
        graph,
        graphId: stableLocatorPart(graph.id, fail),
        nodes: Array.isArray(graph._nodes) ? graph._nodes : graph.nodes ?? [],
      });
      pending.push(
        ...graphValues(graph._subgraphs),
        ...graphValues(graph.subgraphs),
      );
    }
    return entries;
  }

  function liveOwnerRecords(context) {
    const field = transitionField(context, fail);
    const records = [];
    const identities = new Set();
    for (const entry of liveGraphEntries()) {
      if (!Array.isArray(entry.nodes)) fail();
      for (const node of entry.nodes) {
        if (!field.nodeTypes.includes(aioManagedNodeType(node))) continue;
        registerNode(node);
        if (!owners.has(node)) fail();
        synchronizeOwner(node);
        const locator = Object.freeze({
          rootGraphId: "root",
          graphId: stableLocatorPart(entry.graphId, fail),
          nodeId: stableLocatorPart(node.id, fail),
        });
        const identity = JSON.stringify(locator);
        if (identities.has(identity)) fail();
        identities.add(identity);
        records.push(Object.freeze({ node, fieldId: field.id, locator }));
      }
    }
    if (records.length > field.externalTransitionPolicy.maxOwners) fail();
    return records;
  }

  function serializedGraphEntries(serialized) {
    if (!serialized || typeof serialized !== "object" || !Array.isArray(serialized.nodes)) fail();
    const entries = [{ graphId: "root", nodes: serialized.nodes }];
    const definitions = serialized.definitions?.subgraphs;
    if (definitions != null && !Array.isArray(definitions)) fail();
    for (const graph of definitions ?? []) {
      if (!graph || typeof graph !== "object" || !Array.isArray(graph.nodes)) fail();
      entries.push({ graphId: stableLocatorPart(graph.id, fail), nodes: graph.nodes });
    }
    return entries;
  }

  function detachedSerialization() {
    const graph = app?.rootGraph ?? app?.graph ?? liveGraphEntries()[0]?.graph;
    if (typeof graph?.serialize !== "function") fail();
    try {
      return graph.serialize();
    } catch {
      fail();
    }
  }

  function offlineRepresentationCount(records, serialized, context) {
    const field = transitionField(context, fail);
    const liveCounts = new Map();
    for (const { locator } of records) {
      const identity = JSON.stringify(locator);
      liveCounts.set(identity, (liveCounts.get(identity) ?? 0) + 1);
    }
    let count = 0;
    for (const entry of serializedGraphEntries(serialized)) {
      for (const node of entry.nodes) {
        if (!field.nodeTypes.includes(aioManagedNodeType(node))) continue;
        const identity = JSON.stringify({
          rootGraphId: "root",
          graphId: entry.graphId,
          nodeId: stableLocatorPart(node.id, fail),
        });
        const remaining = liveCounts.get(identity) ?? 0;
        if (remaining > 0) liveCounts.set(identity, remaining - 1);
        else count += 1;
      }
    }
    return count;
  }

  function requireOwner(owner, context) {
    const field = transitionField(context, fail);
    if (
      !owner
      || typeof owner !== "object"
      || owner.fieldId !== field.id
      || !owners.has(owner.node)
      || !field.nodeTypes.includes(aioManagedNodeType(owner.node))
    ) fail();
    return { owner, field };
  }

  function expectationMap(node) {
    let values = expectedReadbacks.get(node);
    if (!values) {
      values = new Map();
      expectedReadbacks.set(node, values);
    }
    return values;
  }

  function encodeStorage(value, maximum) {
    if (typeof value !== "string") fail();
    const exact = encoder.encode(value);
    let roundTrip;
    try {
      roundTrip = decoder.decode(exact);
    } catch {
      fail();
    }
    if (roundTrip !== value || exact.byteLength > maximum) fail();
    return exact;
  }

  function decodeStorage(exact, maximum) {
    const bytes = exactBytes(exact, fail);
    if (bytes.byteLength > maximum) fail();
    try {
      return { bytes, value: decoder.decode(bytes) };
    } catch {
      fail();
    }
  }

  function readExact(owner, context) {
    requireTransition();
    const { owner: record, field } = requireOwner(owner, context);
    const expectation = expectationMap(record.node).get(field.id);
    const exact = encodeStorage(
      readStorage(record.node, context),
      maximumBytes(
        field,
        expectation?.maximumKey ?? "maxOriginalBytesPerOwner",
        fail,
      ),
    );
    if (expectation && equalBytes(exact, expectation.exact)) expectation.verified = true;
    return exact;
  }

  function writeExact(owner, exact, context, maximumKey) {
    requireTransition();
    const { owner: record, field } = requireOwner(owner, context);
    const decoded = decodeStorage(
      exact,
      maximumBytes(field, maximumKey, fail),
    );
    withInternalMutation(() => writeStorage(record.node, decoded.value, context));
    expectationMap(record.node).set(field.id, {
      exact: decoded.bytes,
      maximumKey,
      verified: false,
      reloaded: false,
    });
  }

  function requireVerified(owner, context) {
    const { owner: record, field } = requireOwner(owner, context);
    const expectation = expectationMap(record.node).get(field.id);
    if (!expectation?.verified) fail();
    const current = encodeStorage(
      readStorage(record.node, context),
      maximumBytes(field, expectation.maximumKey, fail),
    );
    if (!equalBytes(current, expectation.exact)) fail();
    return { record, field, expectation };
  }

  return Object.freeze({
    isFrozen: frozen,
    isInternalMutation: () => internalMutationDepth > 0,
    requireMutable,
    synchronizeOwner,
    withInternalMutation,
    settleModeTransition(context) {
      transitionField(context, fail);
      transitionDepth += 1;
      for (const node of owners) synchronizeOwner(node);
      let released = false;
      const settled = Promise.resolve().then(() => {
        const before = liveOwnerRecords(context);
        for (const record of before) settleOwner(record.node, context);
        const records = liveOwnerRecords(context);
        return Object.freeze({
          offlineRepresentationCount: offlineRepresentationCount(
            records,
            detachedSerialization(),
            context,
          ),
        });
      });
      return Object.freeze({
        settled,
        async release() {
          if (released) return;
          released = true;
          transitionDepth = Math.max(0, transitionDepth - 1);
          for (const node of owners) synchronizeOwner(node);
        },
      });
    },
    inventoryModeTransitionOwners(context) {
      requireTransition();
      return liveOwnerRecords(context).map((owner) => Object.freeze({
        owner,
        ...owner.locator,
      }));
    },
    readModeTransitionOwnerExact: readExact,
    applyModeTransitionOwnerExact(owner, exact, context) {
      writeExact(owner, exact, context, "maxTargetBytesPerOwner");
    },
    extractDetachedModeTransitionOwnerExact(owner, serialized, context) {
      requireTransition();
      const { owner: record, field } = requireOwner(owner, context);
      const graph = serializedGraphEntries(serialized).find(
        (entry) => entry.graphId === record.locator.graphId,
      );
      const serializedNode = graph?.nodes.find(
        (node) => String(node?.id) === record.locator.nodeId,
      );
      if (!serializedNode || !field.nodeTypes.includes(aioManagedNodeType(serializedNode))) fail();
      return encodeStorage(
        readDetachedStorage(record.node, serializedNode, context),
        maximumBytes(field, "maxTargetBytesPerOwner", fail),
      );
    },
    restoreModeTransitionOwnerExact(owner, exact, context) {
      writeExact(owner, exact, context, "maxOriginalBytesPerOwner");
    },
    reloadModeTransitionRuntime(owner, context) {
      requireTransition();
      const { record, field, expectation } = requireVerified(owner, context);
      reloadRuntime(record.node, readStorage(record.node, context), context);
      expectation.reloaded = true;
      expectationMap(record.node).set(field.id, expectation);
    },
    reconcileModeTransitionRuntime(owner, context) {
      requireTransition();
      const { record, field, expectation } = requireVerified(owner, context);
      if (!expectation.reloaded) fail();
      reconcileRuntime(record.node, context);
      expectationMap(record.node).delete(field.id);
    },
  });
}
