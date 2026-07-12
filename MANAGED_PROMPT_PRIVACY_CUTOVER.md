# Managed Generate, Krea, and Ideogram-builder privacy

`services/managed_prompt_privacy.py` and
`web/js/aio_managed_prompt_privacy.js` contain the A1 Generate/Krea slice of
the future atomic `helto.aio-image-generation` privacy profile.
`services/managed_builder_privacy.py` and
`web/js/aio_managed_builder_privacy.js` add the inactive Ideogram-builder
slice to that same profile.
`services/managed_prompt_library_privacy.py` and
`web/js/aio_managed_prompt_library_privacy.js` add the inactive Ideogram prompt
library as a typed private-record resource.
`services/managed_run_info_privacy.py` adds the final inactive AIO slice: one
server-mode-resolved, sensitive-by-default run-info projection.

The slice is intentionally not installed from `__init__.py` and the browser
adapter is intentionally not connected from `aio_image_generate.js`. AIO must
not publish a partial profile: the Ideogram builder, prompt library, and run
information slices still have to join the same profile before the coordinated
activation.

During this expand phase:

- Generate declares `positive_prompt` and `negative_prompt` as one workflow
  snapshot and Krea declares `inpaint_positive_prompt` as a second snapshot.
- The prompt library declares `ideogram-prompt` with an empty safe list
  projection and fixed `Private record` label. Shared privacy owns opaque IDs,
  authorization, encryption, locked shells, delete confirmation, generic
  errors, and verified rollback-safe writes. The AIO store adapter owns only
  JSON document persistence plus name, metadata, timestamp, duplicate, and
  Ideogram payload normalization. The browser facade maps only product metadata
  (`name`, `description`, and `tags`); the legacy item-level `private` flag never
  reaches the adapter because the shared record declaration is authoritative.
- Authorized details and use return the complete product record behind the
  declared `record` projection. Use updates `last_used_at` through
  `RecordProjectionResult`, so shared privacy encrypts and verifies the activity
  rewrite before returning it. Create, replace, patch, and duplicate return only
  opaque receipts.
- The managed store uses `ideogram4_prompt_library_v2.json`; the v1 library is
  deliberately retained as a separate exact-reader source until the user has
  opened and re-saved every workflow/library record. Genuine AIO v1 envelopes
  bind the shared AIO reader and JSON-key import and receive a migration receipt
  only after the v2 record reopens as the current schema. The inactive adapter
  enumerates private envelopes from the exact
  `ideogram4_prompt_library.json` v1 document without decrypting or constructing
  UI shells; both historical-v1 and already-current envelopes remain visible to
  the later activation scanner. The genuine-v1 transaction preserves encrypted
  record timestamps while rewriting the protected state.
- Run-info keeps its existing product structure and performance calculation in
  `services/run_info.py`. `build_run_info_candidate` deliberately performs no
  encryption or debug omission; the shared `emit-run-info` operation resolves
  the Generate scope on the server and releases only declared performance
  booleans/counts in private mode. Model paths/names, prompt overrides, LoRA
  names, warnings, debug data, and every future consumer-derived field are
  sensitive by default. Public mode returns the unchanged product schema.
- Missing Generate mode inherits the shared private default. Stored legacy
  `false` remains an explicit public declaration when no private floor applies.
- Krea has no local privacy toggle and inherits a hard private floor from its
  upstream Generate scope.
- Shared execution grants resolve protected snapshots into semantic prompt
  objects. Generate and Krea accept injected `private_execution` references and
  call their existing product pipelines inside the shared dispatch. Semantic
  identities key only the shared session RAM cache.
- All three locations bind the AIO v1 reader and JSON-key import and rewrite to
  `helto.aio-image-generate.v2` through the shared migration transaction.
- The builder declares ten sensitive widgets plus one whole-editor field. Its
  property and both workflow keys are mirrors of the same shared envelope, and
  all eleven logical fields migrate under one grouped receipt. The migration
  transaction requires the shared resolved effective mode and writes that
  derived fact into the current whole-state envelope before read-back, while
  preserving the legacy local boolean as the durable declaration.
- The builder execution projection includes its prompt text, palettes,
  elements, coordinates, dimensions, and output controls. It rejects missing
  fields or a widget value that disagrees with the whole-editor generation.
  The shared effective mode is written into that projection, so a private
  downstream Generate floor cannot be weakened by the legacy local boolean.
- Builder execution results are not stored in the semantic RAM cache because
  preview images and optional bbox seeds are live execution inputs outside the
  protected workflow state. Generate/Krea prompt results retain their existing
  cache policy.

At atomic activation, connect the profile/browser adapters and let the shared
queue barrier inject the references that the nodes already understand. Only
construct each workflow browser adapter with its profile-bound workflow handle
so widget and DOM edits call `markEdited`. Only then remove the old prompt
`serializeValue` encryption/memo code in
`aio_image_generate.js` and the legacy-only direct
`decrypt_text_if_encrypted` branches in `aio_generate.py` and
`krea2_settings.py`. At that same cutover, remove the builder's synchronous
per-widget and whole-state encryption, custom locked preservation/recovery,
and local toggle policy from `aio_ideogram4_prompt_builder.js`, plus the direct
decrypt branch in `ideogram4_prompt_builder.py`. Keep writing the property
`aio_ideogram4_prompt_builder_state` and workflow keys
`aio_ideogram4_prompt_builder` and `ideo` from the one shared envelope until
legacy workflow support is deliberately retired. The current product path remains available until that
single cutover so partially migrated AIO installs cannot save or execute with
  mixed privacy authorities. At that cutover, replace the live
  `ideogram4_prompt_library` routes and frontend `libraryRequest` calls with the
  managed record facade, then delete the consumer token checks, encryption,
  private shell construction, and exception-derived route responses. Keep the
  legacy v1 reader/import path until the later explicit legacy-removal ticket.
  Replace the live `build_run_info(... privacy_mode=...)` call at the same
  activation with `build_managed_run_info_json(pack, ...)`, always supplying the
  full debug candidate. Then remove `settings_info_from_settings` encryption,
  the `debug=None if private` omission rule, and request/settings-derived
  `run_info_privacy_mode`. The inactive projection is not called by the live
  node before that atomic switch.
