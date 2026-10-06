"""GPU backend: each op is a fragment shader rendered into an offscreen
RGBA32F buffer. Grayscale results are stored as (v, v, v, 1)."""

import traceback

import numpy as np

import gpu
from gpu_extras.batch import batch_for_shader

from .ops import GLSL_COMMON, to_rgba

_TYPES = {"float": "FLOAT", "int": "INT", "color": "VEC4"}

COPY_GLSL = """
void main() { fragColor = texelFetch(src, ivec2(gl_FragCoord.xy), 0); }
"""

THUMB_GLSL = """
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 cell = vec2(src_size) / vec2(dst_size);
  vec4 acc = vec4(0.0);
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      ivec2 q = ivec2((vec2(p) + (vec2(i, j) + 0.5) / 4.0) * cell);
      acc += texelFetch(src, clamp(q, ivec2(0), src_size - 1), 0);
    }
  }
  fragColor = acc / 16.0;
}
"""


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
    slow_readback = False

    def __init__(self):
        self._shaders = {}
        # (w, h) -> reused gpu.types.Buffer for readback
        self._read_buffers = {}
        self._thumb_targets = {}
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

    def _simple_shader(self, key, source, constants):
        """Compile (once) a full-screen shader with one sampler named src."""
        if key not in self._shaders:
            info = gpu.types.GPUShaderCreateInfo()
            info.vertex_in(0, "VEC2", "pos")
            info.fragment_out(0, "VEC4", "fragColor")
            for kind, name in constants:
                info.push_constant(kind, name)
            info.sampler(0, "FLOAT_2D", "src")
            info.vertex_source("void main() { gl_Position = vec4(pos, 0.0, 1.0); }")
            info.fragment_source(source)
            shader = gpu.shader.create_from_info(info)
            self._shaders[key] = (shader, batch_for_shader(
                shader, "TRIS",
                {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1))},
                indices=((0, 1, 2), (0, 2, 3)),
            ))
        return self._shaders[key]

    def upload(self, array):
        """Copy an (h, w, c) float array into a new offscreen buffer."""
        rgba = np.ascontiguousarray(to_rgba(array), dtype=np.float32)
        h, w = rgba.shape[:2]
        data = gpu.types.Buffer("FLOAT", w * h * 4)
        try:
            np.frombuffer(data, dtype=np.float32)[:] = rgba.ravel()
        except (TypeError, ValueError, BufferError):
            data = gpu.types.Buffer("FLOAT", w * h * 4, rgba.ravel().tolist())
        texture = gpu.types.GPUTexture((w, h), format="RGBA32F", data=data)
        shader, batch = self._simple_shader("__copy__", COPY_GLSL, ())
        off = gpu.types.GPUOffScreen(w, h, format="RGBA32F")
        with off.bind():
            old_blend = gpu.state.blend_get()
            gpu.state.blend_set("NONE")
            shader.bind()
            set_uniform(shader.uniform_sampler, "src", texture)
            batch.draw(shader)
            gpu.state.blend_set(old_blend)
        return off

    def _run_host(self, op, inputs, params, size):
        arrays = {name: (self.to_numpy(buf).copy() if buf is not None else None) for name, buf in inputs.items()}
        return self.upload(op.cpu(arrays, params, size))

    def run(self, op, inputs, params, size, kind):
        if op.glsl is None:
            return self._run_host(op, inputs, params, size)
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
        """Return an (h, w, 4) float32 array. The array may share memory with
        a reused readback buffer, so copy it before the next call if kept."""
        w, h = buf.width, buf.height
        data = None
        if not self.slow_readback:
            data = self._read_buffers.get((w, h))
            if data is None:
                data = gpu.types.Buffer("FLOAT", w * h * 4)
                self._read_buffers[(w, h)] = data
            try:
                with buf.bind():
                    fb = gpu.state.active_framebuffer_get()
                    fb.read_color(0, 0, w, h, 4, 0, "FLOAT", data=data)
                return np.frombuffer(data, dtype=np.float32).reshape(h, w, 4)
            except Exception:
                traceback.print_exc()
                self.slow_readback = True
        data = buf.texture_color.read()
        try:
            arr = np.frombuffer(data, dtype=np.float32)
        except (TypeError, ValueError, BufferError):
            arr = np.array(data.to_list(), dtype=np.float32)
        return arr.reshape(h, w, 4).copy()

    def thumbnail(self, buf, n):
        """Box-filtered (n, n, 4) float32 copy of a result."""
        shader, batch = self._simple_shader("__thumb__", THUMB_GLSL, (("IVEC2", "src_size"), ("IVEC2", "dst_size")))
        off = self._thumb_targets.get(n)
        if off is None:
            off = self._thumb_targets[n] = gpu.types.GPUOffScreen(n, n, format="RGBA32F")
        with off.bind():
            gpu.state.active_framebuffer_get().clear(color=(0.0, 0.0, 0.0, 1.0))
            old_blend = gpu.state.blend_get()
            gpu.state.blend_set("NONE")
            shader.bind()
            set_uniform(shader.uniform_int, "src_size", (buf.width, buf.height))
            set_uniform(shader.uniform_int, "dst_size", (n, n))
            set_uniform(shader.uniform_sampler, "src", buf.texture_color)
            batch.draw(shader)
            gpu.state.blend_set(old_blend)
        return self.to_numpy(off).copy()

    def free(self, buf):
        buf.free()
