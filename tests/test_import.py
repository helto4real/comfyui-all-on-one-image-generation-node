import json
import subprocess
import sys
import textwrap
from pathlib import Path


EXPECTED_NODES = {
    "AIOImageGenerate",
    "AIOZImageTurboSettings",
    "AIOFlux2Klein9BSettings",
    "AIOIdeogram4PromptBuilder",
    "AIOIdeogram4Settings",
    "AIOKrea2Settings",
    "AIOInpaint",
    "AIOLoraConfiguration",
    "AIOLoadPipelineModels",
    "AIOModelInfo",
    "AIOPIDInfo",
    "AIOInpaintInfo",
}


def test_custom_node_package_imports_without_torch_or_gguf_dependency():
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys

        root = {str(root)!r}
        spec = importlib.util.spec_from_file_location(
            "aio_image_generate_testpack",
            root + "/__init__.py",
            submodule_search_locations=[root],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        print(json.dumps({{
            "nodes": sorted(module.NODE_CLASS_MAPPINGS),
            "torch": "torch" in sys.modules,
            "gguf": "gguf" in sys.modules,
            "server": "server" in sys.modules,
        }}))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(payload["nodes"]) == EXPECTED_NODES
    assert payload == {
        "nodes": sorted(EXPECTED_NODES),
        "torch": False,
        "gguf": False,
        "server": False,
    }


def test_custom_node_package_registers_all_routes_with_prompt_server():
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys
        from types import ModuleType, SimpleNamespace

        registrations = []

        class Routes:
            def _register(self, method, path):
                def decorator(function):
                    registrations.append((method, path))
                    return function
                return decorator

            def get(self, path):
                return self._register("GET", path)

            def post(self, path):
                return self._register("POST", path)

            def put(self, path):
                return self._register("PUT", path)

            def patch(self, path):
                return self._register("PATCH", path)

            def delete(self, path):
                return self._register("DELETE", path)

        class App:
            pre_frozen = False
            frozen = False

            def __init__(self):
                self.middlewares = []

        server = ModuleType("server")
        server.PromptServer = SimpleNamespace(
            instance=SimpleNamespace(routes=Routes(), app=App())
        )
        folder_paths = ModuleType("folder_paths")
        folder_paths.get_folder_paths = lambda _category: []
        folder_paths.get_full_path = lambda _category, _file: None
        folder_paths.get_filename_list = lambda _category: []
        aiohttp = ModuleType("aiohttp")
        aiohttp.web = SimpleNamespace()
        sys.modules["server"] = server
        sys.modules["folder_paths"] = folder_paths
        sys.modules["aiohttp"] = aiohttp

        root = {str(root)!r}
        spec = importlib.util.spec_from_file_location(
            "aio_image_generate_route_testpack",
            root + "/__init__.py",
            submodule_search_locations=[root],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        print(json.dumps(sorted(registrations)))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    routes = {tuple(item) for item in json.loads(result.stdout.strip().splitlines()[-1])}
    assert {
        ("GET", "/aio-image-gen/api/loras"),
        ("GET", "/aio-image-gen/api/loras/info"),
        ("GET", "/aio-image-gen/api/loras/info/refresh"),
        ("POST", "/aio-image-gen/api/loras/info"),
        ("GET", "/aio-image-gen/api/loras/img"),
        ("GET", "/helto_privacy/status"),
        ("GET", "/helto_privacy/profiles/{pack_id}"),
        ("POST", "/helto_privacy/unlock"),
        ("POST", "/helto_privacy/lock"),
        ("POST", "/helto_privacy/keystore/init"),
        ("POST", "/helto_privacy/keystore/change_password"),
        ("GET", "/helto_privacy/ui/privacy.js"),
    }.issubset(routes)
    assert not any(path.startswith("/aio_image_generate/privacy") for _, path in routes)
    assert not any(
        path.startswith("/aio_image_generate/ideogram4_prompt_library")
        for _, path in routes
    )
