"""Run every op's GLSL on a real (software) OpenGL context and compare it
with the numpy version. Needs moderngl and a display; locally:
`xvfb-run -a python -m pytest tests/test_gpu_parity.py`. Skipped otherwise."""

import numpy as np
import pytest

from pbr_texture_graph.core.ops import GLSL_COMMON, OPS, to_rgba

moderngl = pytest.importorskip("moderngl")

try:
    CTX = moderngl.create_standalone_context()
except Exception as exc:  # no display or GL driver
    pytest.skip(f"no OpenGL context: {exc}", allow_module_level=True)

GLSL_TYPES = {"float": "float", "int": "int", "color": "vec4"}
SIZE = (64, 48)
VERTEX = """
#version 330
in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }
"""


def run_gpu(op, inputs, params, size=SIZE):
    w, h = size
    lines = ["#version 330", "uniform ivec2 ptg_size;"]
    for name in op.inputs:
        lines += [f"uniform sampler2D in_{name};", f"uniform int has_{name};"]
    for p in op.params:
        lines.append(f"uniform {GLSL_TYPES[p.kind]} p_{p.name};")
    lines.append("out vec4 fragColor;")
    prog = CTX.program(vertex_shader=VERTEX, fragment_shader="\n".join(lines) + GLSL_COMMON + op.glsl)
    quad = CTX.buffer(np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype="f4").tobytes())
    vao = CTX.vertex_array(prog, [(quad, "2f", "pos")])
    textures = []

    def set_uniform(name, value):
        if name in prog:
            prog[name].value = value

    set_uniform("ptg_size", (w, h))
    for slot, name in enumerate(op.inputs):
        src = inputs.get(name)
        set_uniform(f"has_{name}", 1 if src is not None else 0)
        data = to_rgba(src) if src is not None else np.zeros((1, 1, 4), np.float32)
        tex = CTX.texture((data.shape[1], data.shape[0]), 4, np.ascontiguousarray(data, "f4").tobytes(), dtype="f4")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.use(slot)
        set_uniform(f"in_{name}", slot)
        textures.append(tex)
    for p in op.params:
        value = params[p.name]
        set_uniform(f"p_{p.name}", tuple(float(v) for v in value) if p.kind == "color"
                    else float(value) if p.kind == "float" else int(value))
    target = CTX.texture((w, h), 4, dtype="f4")
    fbo = CTX.framebuffer([target])
    fbo.use()
    vao.render()
    out = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(h, w, 4).copy()
    for obj in textures + [target, fbo, vao, quad, prog]:
        obj.release()
    return out


def run_cpu(op, inputs, params, size=SIZE):
    full = {name: inputs.get(name) for name in op.inputs}
    return to_rgba(op.cpu(full, params, size))


def sample_inputs(op, size=SIZE):
    rng = np.random.default_rng(7)
    w, h = size
    smooth = OPS["PERLIN"].cpu({}, dict(OPS["PERLIN"].defaults), size)
    color = np.concatenate([rng.random((h, w, 3)).astype(np.float32) * 0.5 + smooth * 0.5,
                            np.ones((h, w, 1), np.float32)], axis=2)
    inputs = {}
    for i, name in enumerate(op.inputs):
        if name == "flood_fill":
            mask = (OPS["SHAPE"].cpu({}, dict(OPS["SHAPE"].defaults, shape=1, tiling=4), size))
            inputs[name] = OPS["FLOOD_FILL"].cpu({"mask": mask}, dict(OPS["FLOOD_FILL"].defaults), size)
        else:
            inputs[name] = smooth if i % 2 == 0 else color
    return inputs


# Params that exercise non-default branches.
VARIANTS = {
    "TILE_SAMPLER": [{}, dict(scale_random=0.5, position_random=0.3, rotation_random=0.7, value_random=0.5,
                               drop=0.3, offset=0.5)],
    "FLOOD_FILL_GRAY": [dict(mode=0), dict(mode=1, angle=30.0, angle_random=0.5), dict(mode=2)],
    "VORONOI": [dict(mode=m) for m in range(4)],
    "BLEND": [dict(mode=m) for m in range(8)],
}

CASES = [(op_id, params) for op_id, op in sorted(OPS.items()) if op.glsl is not None
         for params in VARIANTS.get(op_id, [{}])]


@pytest.mark.parametrize("op_id,overrides", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
@pytest.mark.parametrize("with_inputs", [True, False])
def test_glsl_matches_numpy(op_id, overrides, with_inputs):
    op = OPS[op_id]
    params = dict(op.defaults)
    params.update(overrides)
    inputs = sample_inputs(op) if with_inputs else {}
    gpu, cpu = run_gpu(op, inputs, params), run_cpu(op, inputs, params)
    diff = np.abs(gpu - cpu)
    # A handful of pixels may sit exactly on a hash or edge boundary where
    # float32 and float64 round differently.
    assert np.mean(diff > 0.02) < 0.01, (op_id, float(diff.max()), float(np.mean(diff > 0.02)))
