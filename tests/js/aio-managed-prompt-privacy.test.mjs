import assert from "node:assert/strict";
import test from "node:test";

import {
  AIO_GENERATE_NEGATIVE_FIELD_ID,
  AIO_GENERATE_POSITIVE_FIELD_ID,
  AIO_KREA_INPAINT_FIELD_ID,
  createAioPromptModeBrowserAdapter,
  createAioPromptWorkflowBrowserAdapter,
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
  adapter.serializeForWorkflow(node, output);
  assert.equal(output.properties[AIO_BUILDER_STATE_PROPERTY], stateCiphertext);
  assert.equal(output[AIO_BUILDER_WORKFLOW_STATE_KEY], stateCiphertext);
  assert.equal(output[AIO_BUILDER_LEGACY_WORKFLOW_STATE_KEY], stateCiphertext);
  for (const [name] of Object.entries(AIO_BUILDER_WIDGET_FIELD_IDS)) {
    const index = node.widgets.findIndex((item) => item.name === name);
    assert.equal(output.widgets_values[index], `CURRENT_${name}`);
  }
});


test("builder serialization blocks when one protected generation is incomplete", () => {
  const node = builderNode();
  const adapter = createAioBuilderWorkflowBrowserAdapter();
  adapter.writeProtected(node, "CURRENT_WHOLE_STATE", { fieldId: AIO_BUILDER_STATE_FIELD_ID });
  assert.throws(
    () => adapter.serializeForWorkflow(
      node,
      { widgets_values: node.widgets.map((item) => item.value), properties: {} },
    ),
    /PRIVACY_AIO_BUILDER_STATE_INVALID/,
  );
});
