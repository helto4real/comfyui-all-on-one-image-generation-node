import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_MODULE = ROOT / "tests" / "js" / "aio-managed-prompt-privacy.test.mjs"


def test_managed_prompt_browser_adapters():
    result = subprocess.run(
        [
            "node",
            "--test",
            "--experimental-default-type=module",
            str(TEST_MODULE),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
