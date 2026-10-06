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
        if op.glsl is None:  # host op, runs its numpy version on every backend
            continue
        assert "fragColor" in op.glsl
        for p in op.params:
            assert f"p_{p.name}" in op.glsl, (op.id, p.name)
        for name in op.inputs:
            assert f"in_{name}" in op.glsl and f"has_{name}" in op.glsl, (op.id, name)


@pytest.mark.parametrize("op,params", [
    ("HEIGHT_TO_LIGHT", {}),
    ("EDGE_HIGHLIGHT", {}),
    ("POSTERIZE", {}),
])
def test_stylized_ops_tile_and_stay_in_range(op, params):
    h = run("PERLIN", octaves=3)
    img = run(op, {"height": h} if op != "POSTERIZE" else {"input": h}, **params)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert seam_ok(img)


def test_posterize_produces_exact_bands():
    ramp = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))[:, :, None]
    out = run("POSTERIZE", {"input": ramp}, steps=4)
    assert sorted(np.unique(np.round(out, 5))) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0], abs=1e-5)


def test_height_to_light_keeps_flat_areas():
    flat = np.full((64, 64, 1), 1.0, dtype=np.float32)
    base = np.zeros((64, 64, 4), dtype=np.float32)
    base[:, :] = (0.2, 0.4, 0.6, 1.0)
    out = run("HEIGHT_TO_LIGHT", {"height": flat, "base": base})
    assert np.allclose(out, base, atol=1e-5)


def test_height_to_light_lights_slopes_facing_the_light():
    u = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    h = (np.tile(np.sin(u), (64, 1)) * 0.5 + 0.5).astype(np.float32)[:, :, None]
    # Light from +x (angle 0): slopes facing +x are where height decreases.
    out = run("HEIGHT_TO_LIGHT", {"height": h}, angle=0.0, cavity=0.0)
    facing, away = out[32, 32, 0], out[32, 0, 0]  # sin falls fastest at pi, rises at 0
    assert facing > 0.5 > away


def test_edge_highlight_marks_raised_rims_only():
    shape = run("SHAPE", size=0.6, softness=0.3)
    flat = run("EDGE_HIGHLIGHT", {"height": np.full((64, 64, 1), 0.5, dtype=np.float32)})
    assert np.allclose(flat[:, :, :3], 0.0)
    rims = run("EDGE_HIGHLIGHT", {"height": shape}, width=0.05)
    assert rims[:, :, 0].max() > 0.5
    assert rims[32, 32, 0] < rims[:, :, 0].max()  # the plateau centre is not a rim


def test_cpu_thumbnail_box_filters():
    img = np.zeros((64, 64, 1), dtype=np.float32)
    img[:, :32] = 1.0
    thumb = CPUBackend().thumbnail(img, 16)
    assert thumb.shape == (16, 16, 4)
    assert np.allclose(thumb[:, :8, 0], 1.0) and np.allclose(thumb[:, 8:, 0], 0.0)


@pytest.mark.parametrize("op,params", [
    ("VORONOI", {"mode": 0}),
    ("VORONOI", {"mode": 1, "scale": 3}),
    ("VORONOI", {"mode": 2}),
    ("VORONOI", {"mode": 3}),
    ("TILE", {}),
    ("TILE", {"tiles_x": 3, "tiles_y": 3, "offset": 0.0}),
    ("GRADIENT", {"mode": 1}),
    ("GRADIENT", {"mode": 1, "angle": 90.0, "repeat": 2}),
])
def test_mvp_generators_tile_and_stay_in_range(op, params):
    img = run(op, **params)
    assert img.shape == (64, 64, 1)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert img.std() > 0.01
    assert seam_ok(img)


def test_voronoi_cells_are_flat_regions():
    cells = run("VORONOI", mode=3, scale=4)
    assert 2 <= len(np.unique(cells)) <= 16


def test_tile_generator_has_gaps_and_varied_heights():
    img = run("TILE", tiles_x=2, tiles_y=2, offset=0.0, gap=0.2, bevel=0.0, variation=0.8)
    assert img.min() == 0.0  # the gaps
    tops = {round(float(img[y, x, 0]), 4) for y, x in ((16, 16), (16, 48), (48, 16), (48, 48))}
    assert len(tops) > 1


def test_gradient_linear_runs_left_to_right():
    img = run("GRADIENT", mode=0)
    assert img[0, 0, 0] < img[0, 32, 0] < img[0, 63, 0]


def test_blur_keeps_mean_and_smooths():
    noise = run("PERLIN", octaves=6, scale=16)
    out = run("BLUR", {"input": noise}, radius=0.05)
    assert np.isclose(out.mean(), noise.mean(), atol=1e-3)
    assert out.std() < noise.std()
    assert seam_ok(out)
    assert np.array_equal(run("BLUR", {"input": noise}, radius=0.0), noise)


def test_warp_with_flat_map_is_identity():
    noise = run("PERLIN")
    flat = np.full((64, 64, 1), 0.5, dtype=np.float32)
    assert np.allclose(run("WARP", {"input": noise, "warp": flat}), noise, atol=1e-5)
    warped = run("WARP", {"input": noise, "warp": run("PERLIN", seed=3)}, intensity=1.0)
    assert not np.allclose(warped, noise)
    assert seam_ok(warped)


def test_transform_offset_and_tiling():
    shape = run("SHAPE", tiling=1)
    shifted = run("TRANSFORM", {"input": shape}, offset_x=0.5)
    assert np.allclose(shifted, np.roll(shape, 32, axis=1), atol=1e-5)
    tiled = run("TRANSFORM", {"input": shape}, tiling=2)
    assert np.allclose(tiled, run("SHAPE", tiling=2), atol=0.05)
    turned = run("TRANSFORM", {"input": run("PERLIN")}, rotation=90.0)
    assert seam_ok(turned)


# ---------------------------------------------------------------------------
# Shape and layout nodes

def tile_mask(n=64, tiles=4, shift=0):
    y, x = np.mgrid[0:n, 0:n]
    t = n // tiles
    m = (((x + shift) % t) > 1) & (((x + shift) % t) < t - 2) & ((y % t) > 1) & ((y % t) < t - 2)
    return m[:, :, None].astype(np.float32)


def test_flood_fill_boxes_wrap_across_edges():
    box = run("FLOOD_FILL", {"mask": tile_mask(shift=8)})
    # The tile cut by the left/right edge is one region with one box.
    np.testing.assert_allclose(box[5, 0], box[5, 63])
    assert box[5, 0, 2] == pytest.approx(12 / 64)
    assert (box[0, 0] == 0).all()  # gaps are background


def test_flood_fill_label_matches_bfs():
    from pbr_texture_graph.core.ops_layout import label_regions

    rng = np.random.default_rng(3)
    for _ in range(20):
        mask = rng.random((19, 27)) > rng.uniform(0.3, 0.6)
        run_idx, region, *_ = label_regions(mask)
        labels = np.where(run_idx >= 0, region[np.maximum(run_idx, 0)], -1)
        # Breadth-first reference with wrap-around.
        ref = -np.ones(mask.shape, int)
        count = 0
        for start in zip(*np.nonzero(mask)):
            if ref[start] >= 0:
                continue
            stack = [start]
            ref[start] = count
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = ((cy + dy) % mask.shape[0], (cx + dx) % mask.shape[1])
                    if mask[q] and ref[q] < 0:
                        ref[q] = count
                        stack.append(q)
            count += 1
        pairs = set(zip(labels[mask].tolist(), ref[mask].tolist()))
        assert len(pairs) == count == len(set(labels[mask].tolist()))


def test_flood_fill_gray_is_flat_per_tile_and_resolution_stable():
    small = run("FLOOD_FILL_GRAY", {"flood_fill": run("FLOOD_FILL", {"mask": tile_mask()})})[..., 0]
    m = tile_mask()[..., 0] > 0
    values = {round(float(small[y, x]), 6) for y, x in zip(*np.nonzero(m))}
    assert len(values) == 16
    big_mask = tile_mask(256)
    big_box = OPS["FLOOD_FILL"].cpu({"mask": big_mask}, dict(OPS["FLOOD_FILL"].defaults), (256, 256))
    big = OPS["FLOOD_FILL_GRAY"].cpu({"flood_fill": big_box}, dict(OPS["FLOOD_FILL_GRAY"].defaults), (256, 256))
    # Same value for the same tile at 4x the resolution.
    assert big[40, 40, 0] == pytest.approx(small[10, 10])


def test_flood_fill_gradient_runs_across_each_tile():
    box = run("FLOOD_FILL", {"mask": tile_mask()})
    g = run("FLOOD_FILL_GRAY", {"flood_fill": box}, mode=1)[..., 0]
    row = g[8, 2:14]
    assert (np.diff(row) > 0).all() and row[0] < 0.15 and row[-1] > 0.85


def test_flood_fill_color_is_color():
    out = run("FLOOD_FILL_COLOR", {"flood_fill": run("FLOOD_FILL", {"mask": tile_mask()})})
    assert out.shape[2] == 4 and out[8, 8, :3].std() > 0


def test_tile_sampler_is_seamless_and_random():
    out = run("TILE_SAMPLER", scale_random=0.5, position_random=0.3, rotation_random=1.0, value_random=0.5)
    assert seam_ok(out)
    assert out.max() > 0.9 and 0.05 < out.mean() < 0.9
    with_pattern = run("TILE_SAMPLER", {"pattern": run("SHAPE", shape=1, size=1.0)}, count_x=4, count_y=4)
    assert with_pattern.max() > 0.9


def test_distance_and_bevel():
    from pbr_texture_graph.core.ops_layout import distance_to

    seeds = np.zeros((40, 50), bool)
    seeds[3, 47] = True
    d = distance_to(seeds, 64)
    assert d[3, 2] == pytest.approx(5.0)  # wraps across the edge
    assert d[13, 47] == pytest.approx(10.0)

    mask = tile_mask()
    dist = run("DISTANCE", {"mask": mask}, distance=0.05)[..., 0]
    assert (dist[mask[..., 0] > 0] == 1).all() and dist.min() < 1
    bevel = run("BEVEL", {"mask": mask}, width=0.05)[..., 0]
    assert bevel[8, 8] == pytest.approx(1.0) and 0 < bevel[8, 3] < 1 and bevel[0, 0] == 0
    assert seam_ok(bevel[:, :, None])
