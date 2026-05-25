"""Progress reporting facade over ComfyUI progress primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgressReporter:
    total_steps: int = 0
    node_id: str | None = None
    debug: bool = False
    phases: list[str] = field(default_factory=list)
    _progress_bar: Any = field(default=None, init=False, repr=False)

    def start(self, total_steps: int | None = None) -> None:
        if total_steps is not None:
            self.total_steps = total_steps
        if self.total_steps <= 0:
            return
        try:
            import comfy.utils  # type: ignore

            self._progress_bar = comfy.utils.ProgressBar(self.total_steps, node_id=self.node_id)
        except Exception:
            self._progress_bar = None

    def phase(self, message: str) -> None:
        self.phases.append(message)
        if self.debug:
            print(f"[AIO Image Generate] {message}")

    def update_absolute(self, value: int, total: int | None = None, preview: Any = None) -> None:
        if self._progress_bar is None:
            return
        self._progress_bar.update_absolute(value, total, preview)

    def prepare_sampling_callback(self, model: Any, steps: int):
        try:
            import latent_preview  # type: ignore

            return latent_preview.prepare_callback(model, steps)
        except Exception:
            self.start(steps)

            def callback(step, x0, x, total_steps):
                self.update_absolute(step + 1, total_steps)

            return callback

    def done(self) -> None:
        self.phase("done")
        if self.total_steps > 0:
            self.update_absolute(self.total_steps, self.total_steps)
