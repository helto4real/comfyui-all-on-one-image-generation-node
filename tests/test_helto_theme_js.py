import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AIO_HELTO_THEME = ROOT / "web" / "js" / "aio_helto_theme.js"


def _run_theme_module_test(tmp_path: Path, body: str) -> None:
    module_path = tmp_path / "aio_helto_theme.mjs"
    module_path.write_text(AIO_HELTO_THEME.read_text(encoding="utf-8"), encoding="utf-8")
    script_path = tmp_path / "test.mjs"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import assert from "node:assert/strict";
            import * as theme from {module_path.as_uri()!r};

            {textwrap.dedent(body)}
            """
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["node", str(script_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_litegraph_widget_theme_applies_and_restores(tmp_path):
    _run_theme_module_test(
        tmp_path,
        """
        const liteGraph = {
          WIDGET_BGCOLOR: "#222",
          WIDGET_OUTLINE_COLOR: "#666",
          WIDGET_PROMOTED_OUTLINE_COLOR: "#BF00FF",
          WIDGET_ADVANCED_OUTLINE_COLOR: "rgba(56, 139, 253, 0.8)",
          WIDGET_TEXT_COLOR: "#DDD",
          WIDGET_SECONDARY_TEXT_COLOR: "#999",
          WIDGET_DISABLED_TEXT_COLOR: "#777",
          unrelated: "keep",
        };
        const previous = { ...liteGraph };
        const result = theme.withHeltoLiteGraphWidgetTheme(() => {
          assert.equal(liteGraph.WIDGET_BGCOLOR, theme.HELTO.bg);
          assert.equal(liteGraph.WIDGET_OUTLINE_COLOR, theme.HELTO.borderStrong);
          assert.equal(liteGraph.WIDGET_PROMOTED_OUTLINE_COLOR, theme.HELTO.accent);
          assert.equal(liteGraph.WIDGET_ADVANCED_OUTLINE_COLOR, theme.HELTO.focus);
          assert.equal(liteGraph.WIDGET_TEXT_COLOR, theme.HELTO.text);
          assert.equal(liteGraph.WIDGET_SECONDARY_TEXT_COLOR, theme.HELTO.textDim);
          assert.equal(liteGraph.WIDGET_DISABLED_TEXT_COLOR, theme.HELTO.textFaint);
          assert.equal(liteGraph.unrelated, "keep");
          return "painted";
        }, liteGraph);

        assert.equal(result, "painted");
        assert.deepEqual(liteGraph, previous);
        """,
    )


def test_litegraph_widget_theme_restores_after_error_and_does_not_add_missing_keys(tmp_path):
    _run_theme_module_test(
        tmp_path,
        """
        const liteGraph = {
          WIDGET_BGCOLOR: "#222",
        };
        const previous = { ...liteGraph };

        assert.throws(() => {
          theme.withHeltoLiteGraphWidgetTheme(() => {
            assert.equal(liteGraph.WIDGET_BGCOLOR, theme.HELTO.bg);
            assert.equal("WIDGET_TEXT_COLOR" in liteGraph, false);
            throw new Error("draw failed");
          }, liteGraph);
        }, /draw failed/);

        assert.deepEqual(liteGraph, previous);

        const snapshot = theme.applyHeltoLiteGraphWidgetTheme(null);
        assert.equal(snapshot, null);
        assert.equal(theme.restoreHeltoLiteGraphWidgetTheme(snapshot), false);
        """,
    )
