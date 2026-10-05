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

    def free(self, buf):
        pass
