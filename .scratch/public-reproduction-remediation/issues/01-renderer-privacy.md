# Correct public-reproduction renderer privacy failures

Status: resolved

## Scope

- Remove idle private prompt plaintext from legacy accessibility snapshots while preserving hover reveal, keyboard editing, and mouse selection.
- Reconcile real Nodes 2.0 widget rows after renderer mounts or switches, keeping encrypted storage separate from the live editable presentation.
- Add regression coverage against ComfyUI frontend `1.45.20` source behavior.
- Do not inspect or modify real workflows, browser state, or the live ComfyUI runtime.

## Evidence

Public suite `helto-suite-2026-07-15.3` remained `cutover-pending` after the legacy accessibility tree exposed the synthetic canary and Vue rendered the encrypted envelope as prompt text.

## Comments

- Exact frontend reference: `Comfy-Org/ComfyUI_frontend` tag `v1.45.20`, commit `7bd9f8b0d0b854ccb5a947623468f36a392ea0ff`.
- Nodes 2.0 targeting now follows the real frontend structure: the node root is selected by `data-node-id`, the widget row by an exact `label` match, and the rendered textarea is treated as presentation only while the widget retains protected storage.
- Private idle presentation uses an empty DOM value. Pointer hover and keyboard focus reveal only the in-memory plaintext; pointer leave and blur restore the empty value without disturbing a mouse selection on `pointerup`.
- Legacy DOM widgets rely on their existing widget callback. The adapter ignores protected-value callback echoes and does not attach a second input handler to the widget-owned textarea, preventing the ComfyUI duplicate callback sequence from replacing the live plaintext with an envelope or empty string.

## Verification

- `node --test tests/js/*.mjs`: passed; direct privacy run passed `26/26` subtests.
- `pytest -q` with isolated XDG, keystore, session, and mode-state paths: `354 passed`.
- Disposable ComfyUI `0.27.0` (`e2a6e30d892402ffcf01d6280c8e2744a4448b9d`), frontend `1.45.20`, Python `3.13.14`:
  - legacy idle accessibility snapshot contained no synthetic canary; workflow serialization contained an AES-256-GCM envelope and no plaintext;
  - legacy hover revealed the synthetic prompt, pointer leave hid it, and selection `0..9` survived `pointerup`;
  - Nodes 2.0 idle accessibility exposed neither plaintext nor envelope, hover/leave worked, a DOM edit remained editable in memory, and workflow serialization stayed encrypted with no plaintext;
  - all keystore, session, mode-state, user, input, output, and browser state lived under disposable `/tmp` paths.

The immutable public AIO artifact in suite `helto-suite-2026-07-15.3` is unchanged and remains unsuitable for promotion. Publishing a corrected artifact and replacement suite is a separate release action.
