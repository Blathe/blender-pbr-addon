"""GPU backend: each op is a fragment shader rendered into an offscreen
RGBA32F buffer. Grayscale results are stored as (v, v, v, 1)."""

import numpy as np

import gpu
from gpu_extras.batch import batch_for_shader

from .ops import GLSL_COMMON

_TYPES = {"float": "FLOAT", "int": "INT", "color": "VEC4"}


def set_uniform(setter, name, value):
    """Set a uniform, skipping ones the GLSL compiler optimized out because
    the op never reads them (Blender raises "uniform ... not found")."""
    try:
        setter(name, value)
    except Exception as exc:
        if "not found" not in str(exc):
            raise


class GPUBackend:
    name = "GPU"

    def __init__(self):
        self._shaders = {}
        data = gpu.types.Buffer("FLOAT", 4, [0.0, 0.0, 0.0, 1.0])
        self._dummy = gpu.types.GPUTexture((1, 1), format="RGBA32F", data=data)

    def _compile(self, op):
        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, "VEC2", "pos")
        info.fragment_out(0, "VEC4", "fragColor")
        info.push_constant("IVEC2", "ptg_size")
        for slot, name in enumerate(op.inputs):
            info.sampler(slot, "FLOAT_2D", f"in_{name}")
            info.push_constant("INT", f"has_{name}")
        for p in op.params:
            info.push_constant(_TYPES[p.kind], f"p_{p.name}")
        info.vertex_source("void main() { gl_Position = vec4(pos, 0.0, 1.0); }")
        info.fragment_source(GLSL_COMMON + op.glsl)
        shader = gpu.shader.create_from_info(info)
        batch = batch_for_shader(
            shader, "TRIS",
            {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1))},
            indices=((0, 1, 2), (0, 2, 3)),
        )
        return shader, batch

    def run(self, op, inputs, params, size, kind):
        if op.id not in self._shaders:
            self._shaders[op.id] = self._compile(op)
        shader, batch = self._shaders[op.id]
        w, h = size
        off = gpu.types.GPUOffScreen(w, h, format="RGBA32F")
        with off.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 1.0))
            old_blend = gpu.state.blend_get()
            gpu.state.blend_set("NONE")
            shader.bind()
            set_uniform(shader.uniform_int, "ptg_size", (w, h))
            for name in op.inputs:
                src = inputs.get(name)
                set_uniform(shader.uniform_int, f"has_{name}", 1 if src is not None else 0)
                set_uniform(shader.uniform_sampler, f"in_{name}", src.texture_color if src is not None else self._dummy)
            for p in op.params:
                value = params[p.name]
                if p.kind == "float":
                    set_uniform(shader.uniform_float, f"p_{p.name}", float(value))
                elif p.kind == "int":
                    set_uniform(shader.uniform_int, f"p_{p.name}", int(value))
                else:
                    set_uniform(shader.uniform_float, f"p_{p.name}", tuple(float(v) for v in value))
            batch.draw(shader)
            gpu.state.blend_set(old_blend)
        return off

    def to_numpy(self, buf):
        """Return an (h, w, 4) float32 array."""
        tex = buf.texture_color
        data = tex.read()
        w, h = buf.width, buf.height
        try:
            arr = np.frombuffer(data, dtype=np.float32)
        except (TypeError, ValueError, BufferError):
            arr = np.array(data.to_list(), dtype=np.float32)
        return arr.reshape(h, w, 4).copy()

    def free(self, buf):
        buf.free()
