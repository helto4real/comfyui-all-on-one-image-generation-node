import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKUP_MODULE = ROOT / "web" / "js" / "aio_lora_info_markup.js"
MAIN_MODULE = ROOT / "web" / "js" / "aio_image_generate.js"


def test_lora_info_markup_escapes_untrusted_html(tmp_path):
    module_path = tmp_path / "aio_lora_info_markup.mjs"
    module_path.write_text(MARKUP_MODULE.read_text(encoding="utf-8"), encoding="utf-8")
    script_path = tmp_path / "test.mjs"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import assert from "node:assert/strict";
            import {{ escapedValueMarkup, imageInfoFieldMarkup }} from {module_path.as_uri()!r};

            const attack = '<img src=x onerror="globalThis.pwned=true">';
            assert.equal(escapedValueMarkup(attack), '<span>&lt;img src=x onerror=&quot;globalThis.pwned=true&quot;&gt;</span>');
            assert.equal(
              imageInfoFieldMarkup("positive", attack),
              '<span><label>positive </label>&lt;img src=x onerror=&quot;globalThis.pwned=true&quot;&gt;</span>',
            );
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(["node", str(script_path)], cwd=ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr + result.stdout


def test_main_lora_dialog_has_no_markup_sniffing_bypass():
    source = MAIN_MODULE.read_text(encoding="utf-8")

    assert 'startsWith("<")' not in source
