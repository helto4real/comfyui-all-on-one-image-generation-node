// Atomic browser activation for the complete AIO privacy profile.

import { app } from "/scripts/app.js";
import {
  installPrivacyConnectionSerializationGate,
} from "/helto_privacy/ui/privacy_snapshot.js";

import {
  createAioBuilderModeBrowserAdapter,
  createAioBuilderWorkflowBrowserAdapter,
} from "./aio_managed_builder_privacy.js";
import {
  createAioManagedPromptLibrary,
} from "./aio_managed_prompt_library_privacy.js";
import {
  createAioPromptModeBrowserAdapter,
  createAioPromptWorkflowBrowserAdapter,
  installAioPromptPrivacyBootstrap,
} from "./aio_managed_prompt_privacy.js";

export const AIO_PRIVACY_PROFILE_ID = "helto.aio-image-generation";
export const AIO_PRIVACY_PROFILE_FINGERPRINT = "f63424f85dfa083277d43069d1a399f500f77e132f001a9355da20dab0f133a1";
const activationGate = installPrivacyConnectionSerializationGate(app);
installAioPromptPrivacyBootstrap(app);

function requireActiveSuite(status) {
  const digest = String(status?.suiteManifestDigest || "");
  if (status?.suiteStatus !== "active" || !/^[0-9a-f]{64}$/.test(digest)) {
    throw new Error("PRIVACY_SUITE_BLOCKED");
  }
  return digest;
}

async function connect() {
  let runtime;
  let suiteManifestDigest;
  try {
    const response = await fetch("/helto_privacy/status", {
      cache: "no-store",
      credentials: "same-origin",
    });
    if (!response.ok) throw new Error("PRIVACY_SUITE_BLOCKED");
    suiteManifestDigest = requireActiveSuite(await response.json());
    runtime = await import(
      `/helto_privacy/ui/privacy_profile/${suiteManifestDigest}.js`
    );
  } catch (error) {
    activationGate.markUnavailable();
    throw error;
  }
  activationGate.coalesce();
  return runtime.connectPrivacyPack({
    app,
    packId: AIO_PRIVACY_PROFILE_ID,
    profileFingerprint: AIO_PRIVACY_PROFILE_FINGERPRINT,
    suiteManifestDigest,
    adapters: {
      "ideogram-builder-mode-browser": createAioBuilderModeBrowserAdapter(),
      "prompt-mode-browser": createAioPromptModeBrowserAdapter(),
    },
    adapterFactories: {
      "generate-workflow-browser": ({ handle }) => (
        createAioPromptWorkflowBrowserAdapter({ workflowHandle: handle, app })
      ),
      "krea-workflow-browser": ({ handle }) => (
        createAioPromptWorkflowBrowserAdapter({ workflowHandle: handle, app })
      ),
      "ideogram-builder-workflow-browser": ({ handle }) => (
        createAioBuilderWorkflowBrowserAdapter({ workflowHandle: handle, app })
      ),
    },
  });
}

export const aioPrivacy = connect();

export async function requireAioPrivacy() {
  const pack = await aioPrivacy;
  pack.authorization.requireReady();
  return pack;
}

let promptLibraryPromise = null;

export function requireAioPromptLibrary() {
  if (!promptLibraryPromise) {
    promptLibraryPromise = requireAioPrivacy().then((pack) => (
      createAioManagedPromptLibrary({ recordsHandle: pack.records("ideogram-prompts") })
    ));
  }
  return promptLibraryPromise;
}
