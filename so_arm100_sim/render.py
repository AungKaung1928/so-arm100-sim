"""Offscreen rendering, made optional.

`mujoco.Renderer` needs a working GL context. On the development machine that
is `MUJOCO_GL=glfw` under a display; on a headless CI runner it is `egl` or
`osmesa` if the image has them, and nothing at all otherwise. Every render
path in this package goes through `Renderer` below, which fails with a
sentence that says what to set instead of a stack trace, and `available()`
lets tests skip cleanly.
"""
import os

import mujoco
import numpy as np

_FAILED = None


class Renderer:
    def __init__(self, model, height=96, width=128):
        self.model = model
        self.height, self.width = int(height), int(width)
        try:
            self._r = mujoco.Renderer(model, self.height, self.width)
        except Exception as e:      # noqa: BLE001 -- GL failures are varied
            raise RuntimeError(
                f"could not create an offscreen renderer ({type(e).__name__}: {e}). "
                f"MUJOCO_GL is {os.environ.get('MUJOCO_GL', 'unset')!r}; set it to "
                f"'glfw' with a display, or 'egl' / 'osmesa' headless.") from e

    def render(self, data, camera="front"):
        self._r.update_scene(data, camera=camera)
        return np.array(self._r.render(), dtype=np.uint8)

    def close(self):
        if getattr(self, "_r", None) is not None:
            self._r.close()
            self._r = None


def available():
    """True if an offscreen renderer can be created right now. Cached."""
    global _FAILED
    if _FAILED is not None:
        return not _FAILED
    try:
        from . import scene
        m = scene.build_model(1)
        Renderer(m, 16, 16).close()
        _FAILED = False
    except Exception:       # noqa: BLE001
        _FAILED = True
    return not _FAILED
