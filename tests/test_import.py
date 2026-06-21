import importlib.util
import sys
from pathlib import Path

import pytest


def test_custom_node_package_imports_without_torch_or_gguf_dependency():
    if "torch" in sys.modules:
        pytest.skip("torch was already imported by the test environment")
    if "gguf" in sys.modules:
        pytest.skip("gguf was already imported by the test environment")

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "aio_image_generate_testpack",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert set(module.NODE_CLASS_MAPPINGS) == {
        "AIOImageGenerate",
        "AIOZImageTurboSettings",
        "AIOFlux2Klein9BSettings",
        "AIOIdeogram4PromptBuilder",
        "AIOIdeogram4Settings",
        "AIOInpaint",
        "AIOLoraConfiguration",
        "AIOLoadPipelineModels",
    }
    assert "torch" not in sys.modules
    assert "gguf" not in sys.modules
