# Managed Generate, Krea, and Ideogram-builder privacy

`services/managed_prompt_privacy.py` and
`web/js/aio_managed_prompt_privacy.js` contain the A1 Generate/Krea slice of
the future atomic `helto.aio-image-generation` privacy profile.
`services/managed_builder_privacy.py` and
`web/js/aio_managed_builder_privacy.js` add the inactive Ideogram-builder
slice to that same profile.

The slice is intentionally not installed from `__init__.py` and the browser
adapter is intentionally not connected from `aio_image_generate.js`. AIO must
not publish a partial profile: the Ideogram builder, prompt library, and run
information slices still have to join the same profile before the coordinated
activation.

During this expand phase:

- Generate declares `positive_prompt` and `negative_prompt` as one workflow
  snapshot and Krea declares `inpaint_positive_prompt` as a second snapshot.
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
mixed privacy authorities.
