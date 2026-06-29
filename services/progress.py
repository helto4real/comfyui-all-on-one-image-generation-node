"""Progress reporting facade over ComfyUI progress primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PROGRESS_TOTAL = 1000

FIRST_SAMPLE_WINDOW = (35.0, 82.0)
SECOND_SAMPLE_WINDOW = (82.0, 95.0)

PHASE_PERCENT_HINTS = {
    "resolving models": 1.0,
    "loading diffusion model": 5.0,
    "loading unconditional diffusion model": 8.0,
    "loading text encoder": 10.0,
    "applying model sampling": 14.0,
    "applying performance settings": 16.0,
    "applying loras": 18.0,
    "applying cfg override": 22.0,
    "encoding prompts": 26.0,
    "rebalancing conditioning": 28.0,
    "loading vae": 30.0,
    "preparing inpaint source": 31.0,
    "encoding inpaint image": 33.0,
    "encoding reference images": 33.0,
    "encoding inpaint reference": 34.0,
    "conditioning inpaint": 35.0,
    "preparing guider": 35.0,
    "applying memory policy": 35.0,
    "sampling": FIRST_SAMPLE_WINDOW[0],
    "upscaling second pass": SECOND_SAMPLE_WINDOW[0],
    "encoding second pass image": 84.0,
    "sampling second pass": SECOND_SAMPLE_WINDOW[0],
    "decoding": 92.0,
    "decoding second pass": 96.0,
    "matching inpaint color": 94.0,
    "stitching inpaint": 96.0,
    "blending inpaint": 98.0,
    "done": 100.0,
}

SAMPLING_PHASE_WINDOWS = {
    "sampling": FIRST_SAMPLE_WINDOW,
    "sampling second pass": SECOND_SAMPLE_WINDOW,
}


@dataclass
class ProgressReporter:
    total_steps: int = 0
    node_id: str | None = None
    debug: bool = False
    progress_total: int = PROGRESS_TOTAL
    phases: list[str] = field(default_factory=list)
    _progress_bar: Any = field(default=None, init=False, repr=False)
    _last_value: int = field(default=0, init=False, repr=False)
    _active_sampling_window: tuple[float, float] = field(default=FIRST_SAMPLE_WINDOW, init=False, repr=False)

    def start(self, total_steps: int | None = None) -> None:
        if total_steps is not None:
            self.total_steps = total_steps
        if self.progress_total <= 0:
            self.progress_total = PROGRESS_TOTAL
        try:
            import comfy.utils  # type: ignore

            self._progress_bar = comfy.utils.ProgressBar(self.progress_total, node_id=self.node_id)
        except Exception:
            self._progress_bar = None

    def phase(self, message: str) -> None:
        self.phases.append(message)
        if self.debug:
            print(f"[AIO Image Generate] {message}")
        phase_key = self._phase_key(message)
        if phase_key in SAMPLING_PHASE_WINDOWS:
            self._active_sampling_window = SAMPLING_PHASE_WINDOWS[phase_key]
        self._send_progress_text(message)
        percent = PHASE_PERCENT_HINTS.get(phase_key)
        if percent is not None:
            self.update_percent(percent)

    def update_percent(self, percent: float, preview: Any = None) -> None:
        value = self._percent_to_value(percent)
        self.update_absolute(value, self.progress_total, preview)

    def update_absolute(self, value: int, total: int | None = None, preview: Any = None) -> None:
        if self._progress_bar is None:
            self.start()
        if self._progress_bar is None:
            return
        target_total = int(total or self.progress_total or PROGRESS_TOTAL)
        if target_total <= 0:
            target_total = PROGRESS_TOTAL
        normalized_value = int(value)
        if target_total != self.progress_total:
            normalized_value = int(round((normalized_value / target_total) * self.progress_total))
            target_total = self.progress_total
        normalized_value = min(max(normalized_value, self._last_value), target_total)
        self._last_value = normalized_value
        try:
            self._progress_bar.update_absolute(normalized_value, target_total, preview)
        except Exception:
            return

    def prepare_sampling_callback(self, model: Any, steps: int, x0_output_dict: dict[str, Any] | None = None):
        previewer = None
        try:
            import latent_preview  # type: ignore
            previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
        except Exception:
            previewer = None

        start_percent, end_percent = self._active_sampling_window
        effective_steps = max(1, int(steps or 0))

        def callback(step, x0, x, total_steps):
            if x0_output_dict is not None:
                x0_output_dict["x0"] = x0
            preview = None
            if previewer is not None:
                try:
                    preview = previewer.decode_latent_to_preview_image("JPEG", x0)
                except Exception:
                    preview = None
            total = max(1, int(total_steps or effective_steps))
            current = min(max(int(step) + 1, 0), total)
            percent = start_percent + ((end_percent - start_percent) * (current / total))
            self.update_percent(percent, preview)

        return callback

    def done(self) -> None:
        self.phase("done")
        self.update_percent(100.0)

    def _percent_to_value(self, percent: float) -> int:
        clamped = min(100.0, max(0.0, float(percent)))
        return int(round((clamped / 100.0) * self.progress_total))

    def _send_progress_text(self, message: str) -> None:
        if not self.node_id:
            return
        try:
            from server import PromptServer  # type: ignore

            server = getattr(PromptServer, "instance", None)
            if server is not None:
                server.send_progress_text(message, str(self.node_id))
        except Exception:
            return

    @staticmethod
    def _phase_key(message: str) -> str:
        for phase in sorted(PHASE_PERCENT_HINTS, key=len, reverse=True):
            if message == phase or message.startswith(f"{phase} "):
                return phase
        return message
