"""numpy backend: the reference implementation and the fallback when no GPU
context is available (for example `blender -b`)."""

import numpy as np

from .ops import to_rgba


class CPUBackend:
    name = "CPU"

    def run(self, op, inputs, params, size, kind):
        buf = op.cpu(inputs, params, size)
        return np.ascontiguousarray(buf, dtype=np.float32)

    def to_numpy(self, buf):
        """Return an (h, w, 4) float32 array."""
        return to_rgba(buf)

    def thumbnail(self, buf, n):
        """Box-filtered (n, n, 4) float32 copy of a result."""
        rgba = to_rgba(buf)
        h, w = rgba.shape[:2]
        if h % n == 0 and w % n == 0:
            return rgba.reshape(n, h // n, n, w // n, 4).mean(axis=(1, 3)).astype(np.float32)
        iy = ((np.arange(n) + 0.5) * h / n).astype(int)
        ix = ((np.arange(n) + 0.5) * w / n).astype(int)
        return np.ascontiguousarray(rgba[np.ix_(iy, ix)])

    def free(self, buf):
        pass
