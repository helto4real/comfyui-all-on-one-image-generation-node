import sys
from types import ModuleType
from types import SimpleNamespace

from services.progress import FIRST_SAMPLE_WINDOW, PHASE_PERCENT_HINTS, PROGRESS_TOTAL, ProgressReporter


def _install_progress_bar(monkeypatch, calls, *, fail_update=False):
    comfy_module = ModuleType("comfy")
    utils_module = ModuleType("comfy.utils")

    class FakeProgressBar:
        def __init__(self, total, node_id=None):
            calls.append({"event": "create", "total": total, "node_id": node_id})

        def update_absolute(self, value, total=None, preview=None):
            if fail_update:
                raise RuntimeError("progress unavailable")
            calls.append({"event": "update", "value": value, "total": total, "preview": preview})

    utils_module.ProgressBar = FakeProgressBar
    comfy_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils_module)


def _install_prompt_server(monkeypatch, calls, *, fail_text=False):
    server_module = ModuleType("server")

    def fake_send_progress_text(text, node_id, sid=None):
        del sid
        if fail_text:
            raise RuntimeError("server unavailable")
        calls.append((text, node_id))

    server_module.PromptServer = SimpleNamespace(
        instance=SimpleNamespace(send_progress_text=fake_send_progress_text)
    )
    monkeypatch.setitem(sys.modules, "server", server_module)


def test_progress_reporter_sends_phase_text_and_monotonic_progress(monkeypatch):
    progress_calls = []
    text_calls = []
    _install_progress_bar(monkeypatch, progress_calls)
    _install_prompt_server(monkeypatch, text_calls)

    reporter = ProgressReporter(total_steps=20, node_id="17")
    reporter.phase("resolving models")
    reporter.phase("encoding prompts")
    reporter.update_percent(3)
    reporter.done()

    assert text_calls == [
        ("resolving models", "17"),
        ("encoding prompts", "17"),
        ("done", "17"),
    ]
    assert progress_calls[0] == {"event": "create", "total": PROGRESS_TOTAL, "node_id": "17"}
    values = [call["value"] for call in progress_calls if call["event"] == "update"]
    assert values == sorted(values)
    assert values[-1] == PROGRESS_TOTAL


def test_rebalancing_conditioning_phase_advances_progress(monkeypatch):
    progress_calls = []
    _install_progress_bar(monkeypatch, progress_calls)
    _install_prompt_server(monkeypatch, [])

    reporter = ProgressReporter(node_id="12")
    reporter.phase("encoding prompts")
    reporter.phase("rebalancing conditioning")

    assert PHASE_PERCENT_HINTS["encoding prompts"] < PHASE_PERCENT_HINTS["rebalancing conditioning"]
    values = [call["value"] for call in progress_calls if call["event"] == "update"]
    assert values == sorted(values)
    assert values[-1] == reporter._percent_to_value(PHASE_PERCENT_HINTS["rebalancing conditioning"])


def test_progress_reporter_swallows_missing_and_failing_comfy_apis(monkeypatch):
    for name in ("comfy", "comfy.utils", "server"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    ProgressReporter(node_id="3").phase("loading diffusion model")

    progress_calls = []
    _install_progress_bar(monkeypatch, progress_calls, fail_update=True)
    _install_prompt_server(monkeypatch, [], fail_text=True)

    reporter = ProgressReporter(node_id="4")
    reporter.phase("loading diffusion model")
    reporter.done()


def test_sampling_callback_preserves_preview_and_maps_weighted_progress(monkeypatch):
    progress_calls = []
    _install_progress_bar(monkeypatch, progress_calls)

    latent_preview_module = ModuleType("latent_preview")
    latent_preview_module.get_previewer = lambda load_device, latent_format: SimpleNamespace(
        decode_latent_to_preview_image=lambda fmt, x0: (fmt, load_device, latent_format, x0)
    )
    monkeypatch.setitem(sys.modules, "latent_preview", latent_preview_module)

    reporter = ProgressReporter(node_id="9")
    reporter.phase("sampling")
    x0_output = {}
    model = SimpleNamespace(load_device="cuda", model=SimpleNamespace(latent_format="flux"))
    callback = reporter.prepare_sampling_callback(model, 4, x0_output_dict=x0_output)

    callback(1, "x0", "x", 4)

    updates = [call for call in progress_calls if call["event"] == "update"]
    assert x0_output == {"x0": "x0"}
    assert updates[-1]["preview"] == ("JPEG", "cuda", "flux", "x0")
    assert reporter._percent_to_value(FIRST_SAMPLE_WINDOW[0]) <= updates[-1]["value"]
    assert updates[-1]["value"] <= reporter._percent_to_value(FIRST_SAMPLE_WINDOW[1])
