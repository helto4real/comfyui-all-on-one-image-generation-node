import assert from "node:assert/strict";
import test from "node:test";

import {
  AIO_GENERATE_NEGATIVE_FIELD_ID,
  AIO_GENERATE_POSITIVE_FIELD_ID,
  AIO_KREA_INPAINT_FIELD_ID,
  createAioPromptModeBrowserAdapter,
  createAioPromptWorkflowBrowserAdapter,
  installAioPromptPrivacyBootstrap,
} from "../../web/js/aio_managed_prompt_privacy.js";
import {
  AIO_BUILDER_LEGACY_WORKFLOW_STATE_KEY,
  AIO_BUILDER_STATE_FIELD_ID,
  AIO_BUILDER_STATE_PROPERTY,
  AIO_BUILDER_WIDGET_FIELD_IDS,
  AIO_BUILDER_WORKFLOW_STATE_KEY,
  createAioBuilderModeBrowserAdapter,
  createAioBuilderWorkflowBrowserAdapter,
} from "../../web/js/aio_managed_builder_privacy.js";
import {
  AIO_PROMPT_RECORD_KIND,
  createAioManagedPromptLibrary,
} from "../../web/js/aio_managed_prompt_library_privacy.js";


function generateNode(mode = undefined) {
  return {
    comfyClass: "AIOImageGenerate",
    widgets: [
      { name: "privacy_mode", value: mode },
      { name: "positive_prompt", value: "positive" },
      { name: "negative_prompt", value: "negative" },
    ],
  };
}


function externalField(id, nodeType, location) {
  return Object.freeze({
    id,
    nodeTypes: [nodeType],
    location,
    externalTransitionPolicy: Object.freeze({
      maxOwners: 1024,
      maxOriginalBytesPerOwner: 1024 * 1024,
      maxTargetBytesPerOwner: 1024 * 1024,
      maxTotalBytes: 32 * 1024 * 1024,
      leaseSeconds: 300,
    }),
  });
}


function serializedNode(node) {
  return {
    id: node.id,
    type: node.comfyClass,
    widgets_values: node.widgets
      .filter((item) => item.serialize !== false && item.options?.serialize !== false)
      .map((item) => structuredClone(item.value)),
    properties: structuredClone(node.properties || {}),
    [AIO_BUILDER_WORKFLOW_STATE_KEY]: structuredClone(
      node._aioIdeogram4PendingWorkflowInfo?.[AIO_BUILDER_WORKFLOW_STATE_KEY],
    ),
    [AIO_BUILDER_LEGACY_WORKFLOW_STATE_KEY]: structuredClone(
      node._aioIdeogram4PendingWorkflowInfo?.[AIO_BUILDER_LEGACY_WORKFLOW_STATE_KEY],
    ),
  };
}

function textElement(value = "") {
  const classes = new Set();
  return {
    value,
    eventTypes: [],
    classList: {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name),
      contains: (name) => classes.has(name),
    },
    setAttribute(name, item) { this[name] = item; },
    addEventListener(name) { this.eventTypes.push(name); },
  };
}


test("inactive suite bootstrap masks prompt DOM fields before managed activation", () => {
  const extensions = [];
  const app = {
    registerExtension(extension) { extensions.push(extension); },
  };
  installAioPromptPrivacyBootstrap(app);
  assert.equal(extensions.length, 1);

  const node = generateNode(true);
  for (const target of node.widgets.filter((item) => item.name.endsWith("_prompt"))) {
    target.inputEl = textElement(target.value);
  }
  extensions[0].nodeCreated(node);

  for (const target of node.widgets.filter((item) => item.name.endsWith("_prompt"))) {
    assert.equal(target.inputEl.classList.contains("aio-managed-private-field"), true);
    assert.equal(target.inputEl.classList.contains("aio-managed-privacy-unavailable"), true);
    assert.equal(target.inputEl["data-aio-privacy-unavailable"], "true");
    assert.deepEqual(target.inputEl.eventTypes, []);
  }
  assert.equal(node.__aioManagedPrivacyUnavailable, true);
  assert.equal(node.__aioManagedPrivacyMasked, true);

  const managed = createAioPromptWorkflowBrowserAdapter({
    workflowHandle: {
      markEdited() {},
      notifyModeChange: async () => {},
    },
  });
  managed.onPrivacySessionChange({ state: "unlocked" });
  managed.reconcileNode(node);
  for (const target of node.widgets.filter((item) => item.name.endsWith("_prompt"))) {
    assert.equal(target.inputEl.classList.contains("aio-managed-private-field"), true);
    assert.equal(target.inputEl.classList.contains("aio-managed-privacy-unavailable"), false);
    assert.equal(target.inputEl["data-aio-privacy-unavailable"], "false");
  }
  assert.equal(node.__aioManagedPrivacyUnavailable, false);
});


test("inactive suite bootstrap leaves explicitly public prompts visible", () => {
  const extensions = [];
  const app = {
    registerExtension(extension) { extensions.push(extension); },
  };
  installAioPromptPrivacyBootstrap(app);

  const node = generateNode(false);
  for (const target of node.widgets.filter((item) => item.name.endsWith("_prompt"))) {
    target.inputEl = textElement(target.value);
  }
  extensions[0].nodeCreated(node);

  for (const target of node.widgets.filter((item) => item.name.endsWith("_prompt"))) {
    assert.equal(target.inputEl.classList.contains("aio-managed-private-field"), false);
    assert.equal(target.inputEl.classList.contains("aio-managed-privacy-unavailable"), false);
    assert.equal(target.inputEl["data-aio-private"], "false");
    assert.equal(target.inputEl["data-aio-privacy-unavailable"], "false");
  }
  assert.equal(node.__aioManagedPrivacyUnavailable, true);
  assert.equal(node.__aioManagedPrivacyMasked, false);
});


test("inactive suite bootstrap updates presentation when privacy mode changes", () => {
  const extensions = [];
  const app = {
    registerExtension(extension) { extensions.push(extension); },
  };
  installAioPromptPrivacyBootstrap(app);

  const node = generateNode(true);
  for (const target of node.widgets.filter((item) => item.name.endsWith("_prompt"))) {
    target.inputEl = textElement(target.value);
  }
  extensions[0].nodeCreated(node);

  const privacyMode = node.widgets.find((item) => item.name === "privacy_mode");
  privacyMode.value = false;
  privacyMode.callback(false);
  for (const target of node.widgets.filter((item) => item.name.endsWith("_prompt"))) {
    assert.equal(target.inputEl.classList.contains("aio-managed-private-field"), false);
    assert.equal(target.inputEl.classList.contains("aio-managed-privacy-unavailable"), false);
  }
  assert.equal(node.__aioManagedPrivacyMasked, false);

  privacyMode.value = true;
  privacyMode.callback(true);
  for (const target of node.widgets.filter((item) => item.name.endsWith("_prompt"))) {
    assert.equal(target.inputEl.classList.contains("aio-managed-private-field"), true);
    assert.equal(target.inputEl.classList.contains("aio-managed-privacy-unavailable"), true);
  }
  assert.equal(node.__aioManagedPrivacyMasked, true);
});


test("private DOM prompt redraw preserves its value and native selection", () => {
  const extensions = [];
  const app = {
    registerExtension(extension) { extensions.push(extension); },
  };
  installAioPromptPrivacyBootstrap(app);

  const node = generateNode(true);
  const prompt = node.widgets.find((item) => item.name === "positive_prompt");
  prompt.inputEl = textElement(prompt.value);
  prompt.inputEl.selectionStart = 1;
  prompt.inputEl.selectionEnd = 5;
  let promptValue = prompt.value;
  let valueWrites = 0;
  Object.defineProperty(prompt, "value", {
    configurable: true,
    get() { return promptValue; },
    set(value) {
      promptValue = value;
      valueWrites += 1;
      prompt.inputEl.selectionStart = value.length;
      prompt.inputEl.selectionEnd = value.length;
    },
  });
  const drawnValues = [];
  prompt.draw = function drawPrompt() {
    drawnValues.push(this.value);
    return "drawn";
  };
  extensions[0].nodeCreated(node);

  assert.equal(prompt.draw(), "drawn");
  assert.deepEqual(drawnValues, ["positive"]);
  assert.equal(prompt.value, "positive");
  assert.equal(valueWrites, 0);
  assert.equal(prompt.inputEl.selectionStart, 1);
  assert.equal(prompt.inputEl.selectionEnd, 5);
});


test("private legacy canvas prompt redraw masks without retaining the mask", () => {
  const extensions = [];
  const app = {
    registerExtension(extension) { extensions.push(extension); },
  };
  installAioPromptPrivacyBootstrap(app);

  const node = generateNode(true);
  const prompt = node.widgets.find((item) => item.name === "positive_prompt");
  const drawnValues = [];
  prompt.draw = function drawPrompt() {
    drawnValues.push(this.value);
    return "drawn";
  };
  extensions[0].nodeCreated(node);

  assert.equal(prompt.draw(), "drawn");
  assert.deepEqual(drawnValues, ["••••••••"]);
  assert.equal(prompt.value, "positive");
});


test("restored node privacy bootstrap reconciles after its DOM widgets mount", () => {
  const extensions = [];
  const scheduledFrames = [];
  const originalRequestAnimationFrame = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = (callback) => {
    scheduledFrames.push(callback);
    return scheduledFrames.length;
  };

  try {
    const app = {
      registerExtension(extension) { extensions.push(extension); },
    };
    installAioPromptPrivacyBootstrap(app);

    const node = generateNode(true);
    extensions[0].loadedGraphNode(node);
    assert.equal(scheduledFrames.length, 1);

    for (const target of node.widgets.filter((item) => item.name.endsWith("_prompt"))) {
      target.inputEl = textElement(target.value);
    }
    scheduledFrames.shift()();

    for (const target of node.widgets.filter((item) => item.name.endsWith("_prompt"))) {
      assert.equal(target.inputEl.classList.contains("aio-managed-private-field"), true);
      assert.equal(target.inputEl.classList.contains("aio-managed-privacy-unavailable"), true);
      assert.equal(target.inputEl["data-aio-privacy-unavailable"], "true");
    }
    assert.equal(scheduledFrames.length, 0);
  } finally {
    if (originalRequestAnimationFrame === undefined) {
      delete globalThis.requestAnimationFrame;
    } else {
      globalThis.requestAnimationFrame = originalRequestAnimationFrame;
    }
  }
});


function callbackOnAssignmentWidget(name, initialValue) {
  let value = initialValue;
  return {
    name,
    callback: null,
    get value() { return value; },
    set value(next) {
      value = next;
      this.callback?.(next);
    },
  };
}


test("managed prompt library delegates every private operation to the typed record handle", async () => {
  const id = `hp-rec-${"A".repeat(32)}`;
  const calls = [];
  const handle = {
    list: async (...args) => {
      calls.push(["list", ...args]);
      return [{ id, kind: AIO_PROMPT_RECORD_KIND, private: true, label: "Private record" }];
    },
    create: async (...args) => {
      calls.push(["create", ...args]);
      return { recordId: id, kind: AIO_PROMPT_RECORD_KIND, operation: "create" };
    },
    reveal: async (...args) => {
      calls.push(["reveal", ...args]);
      return { value: { record: { name: "authorized", payload: { state: {} } } } };
    },
    mutate: async (...args) => {
      calls.push(["mutate", ...args]);
      return { recordId: id, kind: AIO_PROMPT_RECORD_KIND, operation: args[2] };
    },
    delete: async (...args) => {
      calls.push(["delete", ...args]);
      return { operation: "delete" };
    },
  };
  const library = createAioManagedPromptLibrary({ recordsHandle: handle });
  const payload = { state: {}, prompt: "synthetic" };

  assert.equal((await library.list())[0].label, "Private record");
  assert.equal((await library.create(payload, {
    name: "synthetic",
    private: false,
  })).recordId, id);
  assert.equal((await library.details(id)).id, id);
  assert.equal((await library.use(id)).name, "authorized");
  assert.equal((await library.replace(id, payload)).operation, "replace");
  assert.equal((await library.patch(id, { metadata: { tags: ["one"] } })).operation, "patch");
  assert.equal((await library.duplicate(id)).operation, "duplicate");
  await library.delete(id);

  assert.deepEqual(calls.map((item) => item[0]), [
    "list", "create", "reveal", "reveal", "mutate", "mutate", "mutate", "delete",
  ]);
  assert.deepEqual(calls[1][2].metadata, { name: "synthetic" });
  assert.ok(calls.every((item) => item[0] === "list" || item[1] === AIO_PROMPT_RECORD_KIND));
});


test("managed prompt library rejects consumer-built metadata shells", async () => {
  const library = createAioManagedPromptLibrary({
    recordsHandle: {
      list: async () => [{
        id: `hp-rec-${"A".repeat(32)}`,
        kind: AIO_PROMPT_RECORD_KIND,
        private: true,
        label: "Private record",
        name: "leak",
      }],
    },
  });
  await assert.rejects(library.list(), /PRIVACY_AIO_PROMPT_LIBRARY_INVALID/);
});


test("Generate legacy boolean maps explicitly while missing and Krea inherit", () => {
  const mode = createAioPromptModeBrowserAdapter();
  assert.equal(mode.readDeclaredMode(generateNode()), "inherit");
  assert.equal(mode.readDeclaredMode(generateNode(false)), "public");
  assert.equal(mode.readDeclaredMode(generateNode(true)), "private");
  assert.equal(mode.readDeclaredMode({ comfyClass: "AIOKrea2Settings", widgets: [] }), "inherit");
});


test("workflow adapter separates protected snapshot from revealed prompt memory", () => {
  const node = generateNode(true);
  const edits = [];
  const adapter = createAioPromptWorkflowBrowserAdapter({
    workflowHandle: {
      markEdited: (owner, fieldId) => edits.push([owner, fieldId]),
    },
  });
  const positive = { fieldId: AIO_GENERATE_POSITIVE_FIELD_ID, location: { name: "positive_prompt" } };
  const negative = { fieldId: AIO_GENERATE_NEGATIVE_FIELD_ID, location: { name: "negative_prompt" } };

  adapter.writeProtected(node, "CURRENT_POSITIVE", positive);
  adapter.writeProtected(node, "CURRENT_NEGATIVE", negative);
  adapter.apply(node, { value: "revealed positive" }, positive);
  assert.deepEqual(adapter.normalize(node, positive), { value: "revealed positive" });
  assert.equal(adapter.readProtected(node, positive), "CURRENT_POSITIVE");
  assert.equal(adapter.readProtected(node, negative), "CURRENT_NEGATIVE");
  assert.equal(node.widgets[1].value, "CURRENT_POSITIVE");
  assert.equal(node.widgets[2].value, "CURRENT_NEGATIVE");

  adapter.reconcileNode(node);
  node.widgets[1].value = "edited positive";
  node.widgets[1].callback("edited positive");
  assert.deepEqual(adapter.normalize(node, positive), { value: "edited positive" });
  assert.equal(node.widgets[1].value, "CURRENT_POSITIVE");
  assert.deepEqual(edits, [[node, AIO_GENERATE_POSITIVE_FIELD_ID]]);

  adapter.onPrivacySessionChange({ state: "locked" });
  adapter.reconcileNode(node);
  assert.equal(node.widgets[1].value, "CURRENT_POSITIVE");
  assert.equal(node.widgets[2].value, "CURRENT_NEGATIVE");
  assert.equal(adapter.readProtected(node, positive), "CURRENT_POSITIVE");
});


test("programmatic DOM widget assignments do not become user prompt edits", () => {
  const node = generateNode(true);
  node.widgets[1] = callbackOnAssignmentWidget("positive_prompt", "positive");
  const edits = [];
  const adapter = createAioPromptWorkflowBrowserAdapter({
    workflowHandle: {
      markEdited: (owner, fieldId) => edits.push([owner, fieldId]),
    },
  });
  const positive = {
    fieldId: AIO_GENERATE_POSITIVE_FIELD_ID,
    location: { name: "positive_prompt" },
  };

  adapter.reconcileNode(node);
  adapter.apply(node, { value: "revealed positive" }, positive);
  adapter.writeProtected(node, "CURRENT_POSITIVE", positive);

  assert.deepEqual(edits, []);
  assert.equal(node.widgets[1].value, "CURRENT_POSITIVE");
  node.widgets[1].callback("user edit");
  assert.deepEqual(edits, [[node, AIO_GENERATE_POSITIVE_FIELD_ID]]);
});


test("private Generate prompts auto-mask and managed transport widgets stay hidden", async () => {
  const node = generateNode(false);
  const positiveElement = textElement("SYNTHETIC_PRIVATE_PROMPT");
  const negativeElement = textElement("SYNTHETIC_PRIVATE_NEGATIVE");
  node.widgets[1].inputEl = positiveElement;
  node.widgets[2].inputEl = negativeElement;
  const drawValues = [];
  node.widgets[1].draw = function () { drawValues.push(this.value); };
  node.widgets.push(
    { name: "privacy_mode_reference", value: "" },
    { name: "private_execution", value: "" },
  );
  let notifications = 0;
  const adapter = createAioPromptWorkflowBrowserAdapter({
    workflowHandle: {
      markEdited() {},
      notifyModeChange: async () => { notifications += 1; },
    },
  });
  adapter.onPrivacySessionChange({ state: "unlocked" });
  adapter.reconcileNode(node);

  const privacy = node.widgets.find((item) => item.name === "privacy_mode");
  privacy.value = true;
  privacy.callback(true);
  await node.__aioManagedPrivacyModeSettlement;

  assert.equal(notifications, 1);
  assert.equal(node.__aioManagedPrivacyMasked, true);
  assert.equal(positiveElement.classList.contains("aio-managed-private-field"), true);
  assert.equal(negativeElement.classList.contains("aio-managed-private-field"), true);
  assert.equal(positiveElement["data-aio-private"], "true");
  for (const name of ["privacy_mode_reference", "private_execution"]) {
    const hidden = node.widgets.find((item) => item.name === name);
    assert.equal(hidden.hidden, true);
    assert.equal(hidden.type, "hidden");
    assert.deepEqual(hidden.computeSize(), [0, -4]);
  }

  node.widgets[1].draw();
  assert.deepEqual(drawValues, ["positive"]);
  assert.equal(node.widgets[1].value, "positive");
});


test("private workflow storage remains protected while plaintext stays only in live memory", () => {
  const node = generateNode(true);
  const adapter = createAioPromptWorkflowBrowserAdapter();
  const positive = { fieldId: AIO_GENERATE_POSITIVE_FIELD_ID, location: { name: "positive_prompt" } };
  const secret = "SYNTHETIC_PRIVATE_WORKFLOW_SECRET";
  const envelope = JSON.stringify({
    version: 1,
    schema: "helto.aio-image-generate.v2",
    encrypted: true,
    algorithm: "AES-256-GCM",
    keyId: "synthetic-key",
    nonce: "synthetic-nonce",
    ciphertext: "synthetic-ciphertext",
  });

  adapter.onPrivacySessionChange({ state: "unlocked" });
  adapter.apply(node, { value: secret }, positive);
  adapter.writeProtected(node, envelope, positive);
  const output = serializedNode(node);

  assert.equal(node.widgets.find((item) => item.name === "positive_prompt").value, envelope);
  assert.equal(JSON.stringify(output).includes(secret), false);
  assert.equal(output.widgets_values[1], envelope);
  assert.deepEqual(adapter.normalize(node, positive), { value: secret });
});


test("workflow projections skip non-serialized product widgets", () => {
  const node = generateNode(true);
  node.widgets.splice(1, 0, { name: "product_button", value: "not serialized", serialize: false });
  const adapter = createAioPromptWorkflowBrowserAdapter();
  const output = {
    widgets_values: node.widgets.filter((item) => item.serialize !== false).map((item) => item.value),
  };
  adapter.writeWorkflowProjection(
    node,
    output,
    "CURRENT_POSITIVE",
    { fieldId: AIO_GENERATE_POSITIVE_FIELD_ID, location: { name: "positive_prompt" } },
  );

  assert.equal(output.widgets_values[1], "CURRENT_POSITIVE");
  assert.equal(output.widgets_values.length, 3);
});


test("Krea declaration and facts follow connected Generate nodes", () => {
  const first = generateNode(false);
  first.id = 1;
  first.inputs = [{ name: "model_settings" }];
  const second = generateNode(false);
  second.id = 2;
  second.inputs = [{ name: "model_settings" }];
  const krea = {
    id: 3,
    comfyClass: "AIOKrea2Settings",
    widgets: [{ name: "inpaint_positive_prompt", value: "" }],
    outputs: [{ links: [11, 12] }],
  };
  const graph = {
    _nodes: [first, second, krea],
    links: {
      11: { target_id: 1, target_slot: 0 },
      12: { target_id: 2, target_slot: 0 },
    },
  };
  for (const node of graph._nodes) node.graph = graph;
  const mode = createAioPromptModeBrowserAdapter();

  assert.equal(mode.readDeclaredMode(krea), "public");
  assert.deepEqual(mode.readModeFacts(krea), {
    upstream: [
      { sourceId: "aio-generate-1", mode: "public" },
      { sourceId: "aio-generate-2", mode: "public" },
    ],
  });

  second.widgets[0].value = true;
  assert.equal(mode.readDeclaredMode(krea), "inherit");
  assert.equal(mode.readModeFacts(krea).upstream[1].mode, "private");
});


test("connected Krea inpaint prompt inherits the Generate mask and hides transport widgets", () => {
  const generate = generateNode(true);
  generate.id = 1;
  generate.inputs = [{ name: "model_settings" }];
  const element = textElement("SYNTHETIC_KREA_PRIVATE_PROMPT");
  const krea = {
    id: 2,
    comfyClass: "AIOKrea2Settings",
    widgets: [
      { name: "inpaint_positive_prompt", value: "krea", inputEl: element },
      { name: "privacy_mode_reference", value: "" },
      { name: "private_execution", value: "" },
    ],
    outputs: [{ links: [11] }],
  };
  const graph = {
    _nodes: [generate, krea],
    links: { 11: { target_id: 1, target_slot: 0 } },
  };
  generate.graph = graph;
  krea.graph = graph;
  const adapter = createAioPromptWorkflowBrowserAdapter();
  adapter.onPrivacySessionChange({ state: "unlocked" });
  adapter.reconcileNode(krea);

  assert.equal(krea.__aioManagedPrivacyMasked, true);
  assert.equal(element.classList.contains("aio-managed-private-field"), true);
  assert.equal(krea.widgets[1].hidden, true);
  assert.equal(krea.widgets[2].hidden, true);
});


test("Krea field uses its own widget location", () => {
  const node = {
    comfyClass: "AIOKrea2Settings",
    widgets: [{ name: "inpaint_positive_prompt", value: "local inpaint" }],
  };
  const context = {
    fieldId: AIO_KREA_INPAINT_FIELD_ID,
    location: { name: "inpaint_positive_prompt" },
  };
  const adapter = createAioPromptWorkflowBrowserAdapter();
  assert.deepEqual(adapter.normalize(node, context), { value: "local inpaint" });
  adapter.clear(node, context);
  assert.equal(node.widgets[0].value, "");
});


test("prompt workflow transition freezes edits and proves exact detached readback", async () => {
  const node = generateNode(true);
  node.id = 41;
  const graph = {
    _nodes: [node],
    serialize: () => ({ nodes: [serializedNode(node)] }),
  };
  node.graph = graph;
  const edits = [];
  const adapter = createAioPromptWorkflowBrowserAdapter({
    app: { graph },
    workflowHandle: { markEdited: (...args) => edits.push(args) },
  });
  const field = externalField(
    AIO_GENERATE_POSITIVE_FIELD_ID,
    "AIOImageGenerate",
    { kind: "widget", name: "positive_prompt" },
  );
  const context = { field };
  const privateExact = JSON.stringify({
    version: 1,
    schema: "helto.aio-image-generate.v2",
    encrypted: true,
    algorithm: "AES-256-GCM",
    keyId: "synthetic-key",
    nonce: "synthetic-nonce",
    ciphertext: "synthetic-ciphertext",
  });
  adapter.writeProtected(node, privateExact, { fieldId: field.id });
  adapter.reconcileNode(node);

  const settlement = adapter.settleModeTransition(context);
  assert.deepEqual(await settlement.settled, { offlineRepresentationCount: 0 });
  assert.equal(node.__aioManagedPrivacyTransitionFrozen, true);
  assert.throws(
    () => node.widgets.find((item) => item.name === "positive_prompt").callback("blocked"),
    /PRIVACY_AIO_PROMPT_STATE_INVALID/,
  );
  assert.deepEqual(edits, []);

  const [inventory] = await adapter.inventoryModeTransitionOwners(context);
  assert.deepEqual(
    { rootGraphId: inventory.rootGraphId, graphId: inventory.graphId, nodeId: inventory.nodeId },
    { rootGraphId: "root", graphId: "root", nodeId: "41" },
  );
  assert.equal(
    Buffer.from(await adapter.readModeTransitionOwnerExact(inventory.owner, context)).toString(),
    privateExact,
  );

  const publicExact = new TextEncoder().encode('{"value":"public exact prompt"}');
  await adapter.applyModeTransitionOwnerExact(inventory.owner, publicExact, context);
  assert.deepEqual(
    await adapter.readModeTransitionOwnerExact(inventory.owner, context),
    publicExact,
  );
  assert.deepEqual(
    await adapter.extractDetachedModeTransitionOwnerExact(
      inventory.owner,
      graph.serialize(),
      context,
    ),
    publicExact,
  );
  await adapter.reloadModeTransitionRuntime(inventory.owner, context);
  assert.deepEqual(adapter.normalize(node, { fieldId: field.id }), {
    value: "public exact prompt",
  });
  await adapter.reconcileModeTransitionRuntime(inventory.owner, context);
  await settlement.release();
  assert.equal(node.__aioManagedPrivacyTransitionFrozen, false);
});


function builderState() {
  return {
    version: 1,
    effective_privacy_mode: true,
    widgets: {
      "max side": 1024,
      "aspect ratio": "1:1",
      "multiple value": "none",
      privacy_mode: true,
      style: "none",
      import_mode: "when empty",
      output_format: "compact",
      coord_mode: "normalized",
      bbox_order: "yx",
      bg_brightness: 25,
      ...Object.fromEntries(Object.keys(AIO_BUILDER_WIDGET_FIELD_IDS).map(
        (name) => [name, `plain ${name}`],
      )),
    },
    elements: [],
    style_palette: [],
    bg_brightness: 25,
    output_format: "compact",
    coord_mode: "normalized",
    bbox_order: "yx",
    active: -1,
  };
}


function builderNode() {
  let runtime = structuredClone(builderState());
  let editHandler = null;
  const widgets = [
    { name: "privacy_mode", value: true },
    ...Object.keys(AIO_BUILDER_WIDGET_FIELD_IDS).map(
      (name) => ({ name, value: runtime.widgets[name] }),
    ),
  ];
  return {
    comfyClass: "AIOIdeogram4PromptBuilder",
    widgets,
    properties: {},
    _aioIdeogram4PendingWorkflowInfo: {},
    _aioIdeogram4EditorApi: {
      flushManagedEdits: () => structuredClone(runtime),
      applyManagedState: (state) => { runtime = structuredClone(state); },
      clearManagedState: () => {
        for (const name of Object.keys(AIO_BUILDER_WIDGET_FIELD_IDS)) runtime.widgets[name] = "";
        runtime.elements = [];
        runtime.style_palette = [];
      },
      setManagedEditHandler: (handler) => { editHandler = handler; },
    },
    runtime: () => structuredClone(runtime),
    edit: (name, value) => {
      runtime.widgets[name] = value;
      editHandler?.();
    },
  };
}


test("builder workflow projection skips non-serialized product widgets", () => {
  const node = builderNode();
  node.widgets.splice(1, 0, { name: "product_button", value: "not serialized", serialize: false });
  const adapter = createAioBuilderWorkflowBrowserAdapter();
  const fieldId = AIO_BUILDER_WIDGET_FIELD_IDS.high_level_description;
  const output = {
    widgets_values: node.widgets.filter((item) => item.serialize !== false).map((item) => item.value),
  };
  adapter.writeWorkflowProjection(
    node,
    output,
    "CURRENT_DESCRIPTION",
    { fieldId, location: { name: "high_level_description" } },
  );

  assert.equal(output.widgets_values[1], "CURRENT_DESCRIPTION");
  assert.equal(output.widgets_values.length, node.widgets.length - 1);
});


test("builder mode facts carry every connected Generate private floor", () => {
  const builder = builderNode();
  builder.id = 1;
  builder.outputs = [{ links: [11] }];
  const settings = { id: 2, comfyClass: "AIOIdeogram4Settings", outputs: [{ links: [12] }] };
  const generate = generateNode(true);
  generate.id = 3;
  const graph = {
    _nodes: [builder, settings, generate],
    links: {
      11: { target_id: 2, target_slot: 0 },
      12: { target_id: 3, target_slot: 0 },
    },
  };
  for (const node of graph._nodes) node.graph = graph;
  const adapter = createAioBuilderModeBrowserAdapter();

  assert.equal(adapter.readDeclaredMode(builder), "private");
  assert.deepEqual(adapter.readModeFacts(builder), {
    upstream: [{ sourceId: "aio-generate-3", mode: "private" }],
  });
  generate.widgets[0].value = false;
  assert.equal(adapter.readModeFacts(builder).upstream[0].mode, "public");

  const workflow = createAioBuilderWorkflowBrowserAdapter();
  builder.widgets.find((item) => item.name === "privacy_mode").value = false;
  builder.edit("privacy_mode", false);
  const privateState = workflow.normalize(builder, {
    fieldId: AIO_BUILDER_STATE_FIELD_ID,
    effectiveMode: "private",
  });
  assert.equal(privateState.widgets.privacy_mode, false);
  assert.equal(privateState.effective_privacy_mode, true);
  const publicState = workflow.normalize(builder, {
    fieldId: AIO_BUILDER_STATE_FIELD_ID,
    effectiveMode: "public",
  });
  assert.equal(publicState.widgets.privacy_mode, false);
  assert.equal(publicState.effective_privacy_mode, false);
});


test("builder adapter flushes edits and preserves every locked serialized byte", () => {
  const node = builderNode();
  const edits = [];
  const adapter = createAioBuilderWorkflowBrowserAdapter({
    workflowHandle: { markEdited: (_owner, fieldId) => edits.push(fieldId) },
  });
  const stateContext = { fieldId: AIO_BUILDER_STATE_FIELD_ID };
  const stateCiphertext = "CURRENT_WHOLE_STATE";
  for (const [name, fieldId] of Object.entries(AIO_BUILDER_WIDGET_FIELD_IDS)) {
    adapter.writeProtected(node, `CURRENT_${name}`, { fieldId, location: { name } });
  }
  adapter.writeProtected(node, stateCiphertext, stateContext);
  adapter.apply(node, builderState(), stateContext);
  adapter.reconcileNode(node);

  node.edit("background", "edited runtime background");
  assert.deepEqual(
    new Set(edits),
    new Set([...Object.values(AIO_BUILDER_WIDGET_FIELD_IDS), AIO_BUILDER_STATE_FIELD_ID]),
  );
  assert.deepEqual(
    adapter.normalize(node, {
      fieldId: AIO_BUILDER_WIDGET_FIELD_IDS.background,
      location: { name: "background" },
    }),
    { value: "edited runtime background" },
  );

  adapter.onPrivacySessionChange({ state: "locked" });
  adapter.reconcileNode(node);
  assert.equal(node.runtime().widgets.background, "");
  assert.equal(
    node.widgets.find((item) => item.name === "background").value,
    "CURRENT_background",
  );

  const output = { widgets_values: node.widgets.map((item) => item.value), properties: {} };
  for (const [name, fieldId] of Object.entries(AIO_BUILDER_WIDGET_FIELD_IDS)) {
    adapter.writeWorkflowProjection(
      node,
      output,
      `CURRENT_${name}`,
      { fieldId, location: { name } },
    );
  }
  adapter.writeWorkflowProjection(node, output, stateCiphertext, stateContext);
  assert.equal(output.properties[AIO_BUILDER_STATE_PROPERTY], stateCiphertext);
  assert.equal(output[AIO_BUILDER_WORKFLOW_STATE_KEY], stateCiphertext);
  assert.equal(output[AIO_BUILDER_LEGACY_WORKFLOW_STATE_KEY], stateCiphertext);
  for (const [name] of Object.entries(AIO_BUILDER_WIDGET_FIELD_IDS)) {
    const index = node.widgets.findIndex((item) => item.name === name);
    assert.equal(output.widgets_values[index], `CURRENT_${name}`);
  }
});


test("builder workflow projection rejects an incomplete serialized node", () => {
  const node = builderNode();
  const adapter = createAioBuilderWorkflowBrowserAdapter();
  assert.throws(
    () => adapter.writeWorkflowProjection(
      node,
      { widgets_values: [] },
      "CURRENT_background",
      {
        fieldId: AIO_BUILDER_WIDGET_FIELD_IDS.background,
        location: { name: "background" },
      },
    ),
    /PRIVACY_AIO_BUILDER_STATE_INVALID/,
  );
});


test("builder workflow transition handles widget and mirrored state exact bytes", async () => {
  const node = builderNode();
  node.id = 52;
  const graph = {
    _nodes: [node],
    serialize: () => ({ nodes: [serializedNode(node)] }),
  };
  node.graph = graph;
  const adapter = createAioBuilderWorkflowBrowserAdapter({
    app: { graph },
    workflowHandle: { markEdited() {} },
  });
  const widgetField = externalField(
    AIO_BUILDER_WIDGET_FIELD_IDS.background,
    "AIOIdeogram4PromptBuilder",
    { kind: "widget", name: "background" },
  );
  const widgetContext = { field: widgetField };
  const privateExact = JSON.stringify({
    version: 1,
    schema: "helto.aio-ideogram4-builder.v2",
    encrypted: true,
    algorithm: "AES-256-GCM",
    keyId: "synthetic-key",
    nonce: "synthetic-nonce",
    ciphertext: "synthetic-ciphertext",
  });
  adapter.writeProtected(node, privateExact, { fieldId: widgetField.id });
  adapter.reconcileNode(node);

  const widgetSettlement = adapter.settleModeTransition(widgetContext);
  assert.deepEqual(await widgetSettlement.settled, { offlineRepresentationCount: 0 });
  const [widgetOwner] = await adapter.inventoryModeTransitionOwners(widgetContext);
  const publicWidgetExact = new TextEncoder().encode('{"value":"public background"}');
  await adapter.applyModeTransitionOwnerExact(
    widgetOwner.owner,
    publicWidgetExact,
    widgetContext,
  );
  assert.deepEqual(
    await adapter.readModeTransitionOwnerExact(widgetOwner.owner, widgetContext),
    publicWidgetExact,
  );
  assert.deepEqual(
    await adapter.extractDetachedModeTransitionOwnerExact(
      widgetOwner.owner,
      graph.serialize(),
      widgetContext,
    ),
    publicWidgetExact,
  );
  await adapter.reloadModeTransitionRuntime(widgetOwner.owner, widgetContext);
  await adapter.reconcileModeTransitionRuntime(widgetOwner.owner, widgetContext);
  assert.equal(node.runtime().widgets.background, "public background");
  await widgetSettlement.release();

  const stateField = externalField(
    AIO_BUILDER_STATE_FIELD_ID,
    "AIOIdeogram4PromptBuilder",
    { kind: "property", name: AIO_BUILDER_STATE_PROPERTY },
  );
  const stateContext = { field: stateField };
  adapter.writeProtected(node, privateExact, { fieldId: stateField.id });
  const stateSettlement = adapter.settleModeTransition(stateContext);
  assert.deepEqual(await stateSettlement.settled, { offlineRepresentationCount: 0 });
  const [stateOwner] = await adapter.inventoryModeTransitionOwners(stateContext);
  const publicState = builderState();
  publicState.effective_privacy_mode = false;
  const publicStateExact = new TextEncoder().encode(JSON.stringify(publicState));
  await adapter.applyModeTransitionOwnerExact(
    stateOwner.owner,
    publicStateExact,
    stateContext,
  );
  assert.deepEqual(
    await adapter.readModeTransitionOwnerExact(stateOwner.owner, stateContext),
    publicStateExact,
  );
  assert.deepEqual(
    await adapter.extractDetachedModeTransitionOwnerExact(
      stateOwner.owner,
      graph.serialize(),
      stateContext,
    ),
    publicStateExact,
  );
  await adapter.reloadModeTransitionRuntime(stateOwner.owner, stateContext);
  await adapter.reconcileModeTransitionRuntime(stateOwner.owner, stateContext);
  assert.equal(node.runtime().effective_privacy_mode, false);
  await stateSettlement.release();
});
