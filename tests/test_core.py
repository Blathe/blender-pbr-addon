import numpy as np
import pytest

from pbr_texture_graph.core.cpu import CPUBackend
from pbr_texture_graph.core.graph import Evaluator, GraphError, NodeSpec
from pbr_texture_graph.core.ops import OPS

SIZE = (64, 64)


def run(op, inputs=None, **params):
    p = dict(OPS[op].defaults)
    p.update(params)
    full = {name: None for name in OPS[op].inputs}
    full.update(inputs or {})
    return OPS[op].cpu(full, p, SIZE)


def seam_ok(img):
    """The jump across each wrap edge is no bigger than the largest jump inside."""
    g = img[:, :, 0]
    inner_x = np.abs(np.diff(g, axis=1)).max()
    inner_y = np.abs(np.diff(g, axis=0)).max()
    return (np.abs(g[:, 0] - g[:, -1]).max() <= inner_x * 1.01 + 1e-6
            and np.abs(g[0, :] - g[-1, :]).max() <= inner_y * 1.01 + 1e-6)


@pytest.mark.parametrize("op,params", [
    ("PERLIN", {}),
    ("PERLIN", {"scale": 3, "octaves": 6, "seed": 7}),
    ("SHAPE", {"tiling": 3, "shape": 0}),
    ("SHAPE", {"tiling": 2, "shape": 2}),
])
def test_generators_tile_and_stay_in_range(op, params):
    img = run(op, **params)
    assert img.shape == (64, 64, 1)
    assert img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert img.std() > 0.01
    assert seam_ok(img)


def test_perlin_seed_changes_output():
    assert not np.allclose(run("PERLIN", seed=1), run("PERLIN", seed=2))


def test_normal_of_flat_height_points_up():
    flat = np.full((64, 64, 1), 0.5, dtype=np.float32)
    n = run("NORMAL", {"height": flat})
    assert np.allclose(n[:, :, :3], (0.5, 0.5, 1.0), atol=1e-6)


def test_normal_formats_flip_green():
    h = run("PERLIN")
    gl = run("NORMAL", {"height": h}, format=0)
    dx = run("NORMAL", {"height": h}, format=1)
    assert np.allclose(gl[:, :, 0], dx[:, :, 0])
    assert np.allclose(gl[:, :, 1], 1.0 - dx[:, :, 1], atol=1e-6)


def test_normal_slope_direction():
    # Height rising to the right gives a normal leaning left (negative x).
    u = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False)).astype(np.float32)
    h = np.tile(u, (64, 1))[:, :, None] * 0.5 + 0.5
    n = run("NORMAL", {"height": h})
    assert n[32, 0, 0] < 0.5  # slope is positive at the start of the sine


def test_blend_modes():
    a = np.full((64, 64, 1), 0.25, dtype=np.float32)
    b = np.full((64, 64, 1), 0.5, dtype=np.float32)
    assert np.allclose(run("BLEND", {"foreground": a, "background": b}, mode=2), 0.125)
    assert np.allclose(run("BLEND", {"foreground": a, "background": b}, mode=1), 0.75)
    assert np.allclose(run("BLEND", {"foreground": a, "background": b}, mode=0, opacity=0.5), 0.375)
    mask = np.zeros((64, 64, 1), dtype=np.float32)
    assert np.allclose(run("BLEND", {"foreground": a, "background": b, "mask": mask}), 0.5)


def test_blend_promotes_to_color():
    a = np.full((64, 64, 1), 0.25, dtype=np.float32)
    c = np.zeros((64, 64, 4), dtype=np.float32)
    c[:, :] = (1.0, 0.0, 0.0, 1.0)
    out = run("BLEND", {"foreground": a, "background": c}, mode=2)
    assert out.shape == (64, 64, 4)
    assert np.allclose(out[0, 0], (0.25, 0.0, 0.0, 1.0))


def test_levels():
    ramp = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))[:, :, None]
    out = run("LEVELS", {"input": ramp}, in_low=0.25, in_high=0.75)
    assert out[0, 0, 0] == 0.0 and out[0, -1, 0] == 1.0
    out = run("LEVELS", {"input": ramp}, out_low=0.6, out_high=0.9)
    assert np.isclose(out.min(), 0.6) and np.isclose(out.max(), 0.9)


def test_gradient_map_hits_stops():
    lo, mid, hi = (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)
    for value, expect in ((0.0, lo), (0.5, mid), (1.0, hi)):
        src = np.full((64, 64, 1), value, dtype=np.float32)
        out = run("GRADIENT_MAP", {"input": src}, color_low=lo, color_mid=mid, color_high=hi)
        assert np.allclose(out[0, 0], expect, atol=1e-5)


class CountingBackend(CPUBackend):
    def __init__(self):
        self.runs = []

    def run(self, op, inputs, params, size, kind):
        self.runs.append(op.id)
        return super().run(op, inputs, params, size, kind)


def graph(scale=4):
    return {
        "noise": NodeSpec("noise", "PERLIN", {"scale": scale}),
        "shape": NodeSpec("shape", "SHAPE"),
        "blend": NodeSpec("blend", "BLEND", {"mode": 2}, {"foreground": "noise", "background": "shape"}),
        "out": NodeSpec("out", "OUTPUT", {}, {"input": "blend"}),
    }


def test_evaluator_caches_unchanged_nodes():
    backend = CountingBackend()
    ev = Evaluator(backend)
    ev.evaluate(graph(), ["out"], SIZE)
    assert sorted(backend.runs) == ["BLEND", "OUTPUT", "PERLIN", "SHAPE"]
    backend.runs.clear()
    ev.evaluate(graph(), ["out"], SIZE)
    assert backend.runs == []
    ev.evaluate(graph(scale=5), ["out"], SIZE)
    assert sorted(backend.runs) == ["BLEND", "OUTPUT", "PERLIN"]


def test_evaluator_reruns_on_resolution_change():
    backend = CountingBackend()
    ev = Evaluator(backend)
    ev.evaluate(graph(), ["out"], SIZE)
    backend.runs.clear()
    res = ev.evaluate(graph(), ["out"], (32, 32))
    assert len(backend.runs) == 4
    assert res["out"][0].shape[:2] == (32, 32)


def test_evaluator_output_kind():
    ev = Evaluator(CPUBackend())
    g = graph()
    assert ev.evaluate(g, ["out"], SIZE)["out"][1] == "GRAY"
    g["map"] = NodeSpec("map", "GRADIENT_MAP", {}, {"input": "blend"})
    g["out"].inputs["input"] = "map"
    assert ev.evaluate(g, ["out"], SIZE)["out"][1] == "COLOR"


def test_evaluator_rejects_cycles():
    g = {
        "a": NodeSpec("a", "LEVELS", {}, {"input": "b"}),
        "b": NodeSpec("b", "LEVELS", {}, {"input": "a"}),
    }
    with pytest.raises(GraphError):
        Evaluator(CPUBackend()).evaluate(g, ["a"], SIZE)


def test_every_op_has_matching_glsl_names():
    for op in OPS.values():
        assert "fragColor" in op.glsl
        for p in op.params:
            assert f"p_{p.name}" in op.glsl, (op.id, p.name)
        for name in op.inputs:
            assert f"in_{name}" in op.glsl and f"has_{name}" in op.glsl, (op.id, name)
