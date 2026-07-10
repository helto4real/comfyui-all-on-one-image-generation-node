import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_MODULE = ROOT / "web" / "js" / "aio_settings_migration.js"


def test_removed_settings_widgets_are_migrated_by_position(tmp_path):
    module_path = tmp_path / "aio_settings_migration.mjs"
    module_path.write_text(MIGRATION_MODULE.read_text(encoding="utf-8"), encoding="utf-8")
    script_path = tmp_path / "test.mjs"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import assert from "node:assert/strict";
            import {{ normalizeFluxSettingsWidgetValues, normalizeZImageSettingsWidgetValues }} from {module_path.as_uri()!r};

            const oldCurrentFlux = ["distilled", 1, "auto", "balanced", 0.5, 1.15, 2, "area", 1, "auto", "off", "inductor", "after_loras"];
            assert.deepEqual(
              normalizeFluxSettingsWidgetValues(oldCurrentFlux),
              ["distilled", 1, "auto", "balanced", 2, "area", 1, "auto", "off", "inductor", "after_loras"],
            );
            assert.deepEqual(
              normalizeFluxSettingsWidgetValues(["distilled", 1, "single_reference", 0.75, "bf16", "low_vram", 0.5, 1.15]),
              ["distilled", 1, "bf16", "low_vram"],
            );
            assert.deepEqual(
              normalizeFluxSettingsWidgetValues(["base", 4, "multi_reference", 0.6, "fp8", "balanced", 0.4, 1.2, 2, "area", 8]),
              ["base", 4, "fp8", "balanced", 2, "area", 8],
            );
            assert.deepEqual(
              normalizeFluxSettingsWidgetValues(["base", 4, 0.6, "fp8", "balanced", 0.4, 1.2, 2, "area", 8]),
              ["base", 4, "fp8", "balanced", 2, "area", 8],
            );
            assert.deepEqual(
              normalizeFluxSettingsWidgetValues(["base", 4, "fp8", "balanced", 0.4, 1.2, 2, "area", 8]),
              ["base", 4, "fp8", "balanced", 2, "area", 8],
            );
            const currentFlux = ["base", 4, "bf16", "balanced", 2, "area", 8, "sage", "on", "inductor", "before_loras"];
            assert.strictEqual(normalizeFluxSettingsWidgetValues(currentFlux), currentFlux);

            const zImage = ["default", 8, "strong", true, "auto", "sage", "on", "inductor", "after_loras"];
            assert.deepEqual(
              normalizeZImageSettingsWidgetValues(zImage),
              [8, "auto", "sage", "on", "inductor", "after_loras"],
            );
            assert.deepEqual(
              normalizeZImageSettingsWidgetValues(["quality", 12, "light", false, "bf16"]),
              [12, "bf16"],
            );

            const current = [8, "auto", "sage", "on", "inductor", "after_loras"];
            assert.strictEqual(normalizeZImageSettingsWidgetValues(current), current);
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(["node", str(script_path)], cwd=ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr + result.stdout
