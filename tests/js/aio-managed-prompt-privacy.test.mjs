import assert from "node:assert/strict";
import test from "node:test";

import {
  AIO_GENERATE_NEGATIVE_FIELD_ID,
  AIO_GENERATE_POSITIVE_FIELD_ID,
  AIO_KREA_INPAINT_FIELD_ID,
  createAioPromptModeBrowserAdapter,
  createAioPromptWorkflowBrowserAdapter,
} from "../../web/js/aio_managed_prompt_privacy.js";


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
