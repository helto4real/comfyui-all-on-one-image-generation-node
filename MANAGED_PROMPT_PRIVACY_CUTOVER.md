# Managed AIO privacy profile

The AIO privacy cutover is active as one atomic profile:

- profile ID: `helto.aio-image-generation`
- profile fingerprint: `f63424f85dfa083277d43069d1a399f500f77e132f001a9355da20dab0f133a1`
- distribution: `comfyui-all-on-one-image-generation-node`

`__init__.py` registers the shared privacy UI and installs the complete server
profile. `web/js/aio_managed_privacy.js` requires an active attested suite,
loads the digest-routed browser runtime, and connects the exact matching
browser profile. Either side fails closed when the shared package, suite,
adapter set, digest, or fingerprint is missing or stale.

The shared profile exclusively owns:

- Generate and Krea prompt workflow protection and private execution;
- Ideogram builder workflow protection and private execution;
- subject-mode resolution and queue-time reference injection;
- prompt-library authorization, encryption, locked shells, and mutations;
- private run-info projection;
- legacy envelope discovery, recovery, and verified migration.

Product adapters retain only product normalization, graph locations, execution
dispatch, and opaque prompt-library persistence. The local privacy service,
routes, browser codec, recovery module, prompt-library routes, token retries,
synchronous encryption, and hover-based reveal policy were removed.

Historical `helto.aio-image-generate` v1 envelopes remain supported through the
profile's exact shared reader, JSON-key import, and migration transaction. The
separate v1 prompt-library document remains a read-only migration source until
the explicit legacy-removal ticket.

## Distribution gate

The source checkout requires the exact `helto-privacy` candidate commit used
by the test suite. `requirements.txt` and `pyproject.toml` name the same
immutable source identity; neither declaration uses a local path or mutable
branch. Publication remains a separate gated operation. `[tool.comfy]`
publishes the `web/` tree containing the attested
`aio_managed_privacy.js` entrypoint.
