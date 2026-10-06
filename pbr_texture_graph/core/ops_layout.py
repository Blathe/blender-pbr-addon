"""Shape and layout ops: flood fill, tile sampler, distance and bevel.

Same conventions as ops.py. Flood Fill, Distance and Bevel are host ops
(glsl=None): they need whole regions or long-range searches, so the GPU
backend runs their numpy versions.
"""

import numpy as np

from .ops import (COLOR, GLSL_BILINEAR, GRAY, SAME, OpDef, Param, bilinear, gray, hash_u32, register_op,
                  to_gray, uv_grid)


def _empty(size, channels=1):
    return np.zeros((size[1], size[0], channels), dtype=np.float32)


# ---------------------------------------------------------------------------
# Connected regions

def _union_find(n, a, b):
    """Root (smallest index) for each of n nodes joined by edges a[i]-b[i]."""
    parent = np.arange(n, dtype=np.int64)
    while True:
        while True:
            grand = parent[parent]
            if np.array_equal(grand, parent):
                break
            parent = grand
        pa, pb = parent[a], parent[b]
        differ = pa != pb
        if not differ.any():
            return parent
        np.minimum.at(parent, np.maximum(pa[differ], pb[differ]), np.minimum(pa[differ], pb[differ]))


def label_regions(mask):
    """Label 4-connected regions of a boolean (h, w) mask, wrapping at the
    edges like every texture here. Works on horizontal runs, so the
    union-find sees thousands of runs instead of millions of pixels.
    Returns (run index per pixel, region per run, run start x, run length,
    run row); run index is -1 off the mask."""
    h, w = mask.shape
    starts = mask & ~np.roll(mask, 1, axis=1)
    full = mask.all(axis=1)
    starts[full, 0] = True  # a fully covered row is one run starting at 0
    run = (np.cumsum(starts.ravel(), dtype=np.int64) - 1).reshape(h, w)
    # Pixels before a row's first start continue the row's last run.
    first_start = np.where(starts.any(axis=1), starts.argmax(axis=1), w)
    before = np.arange(w)[None, :] < first_start[:, None]
    run = np.where(before, run[:, -1][:, None], run)
    run = np.where(mask, run, -1)
    count = int(starts.sum())

    below = mask & np.roll(mask, -1, axis=0)
    # Every overlap of two runs on neighbouring rows begins where one of
    # them starts, or at x = 0 for runs that wrap.
    edge = below & (starts | np.roll(starts, -1, axis=0))
    edge[:, 0] |= below[:, 0]
    region = _union_find(count, run[edge], np.roll(run, -1, axis=0)[edge])

    ys, xs = np.nonzero(starts)
    lengths = np.bincount(run[run >= 0], minlength=count)
    return run, region, xs.astype(np.int64), lengths, ys.astype(np.int64)


def _region_boxes(region, x0, length, row, size):
    """Wrap-aware bounding box (min x, min y, width, height in pixels) for
    each region, indexed by region root."""
    w, h = size
    x1 = x0 + length  # exclusive, may pass w for runs that wrap
    n = len(region)

    def extent(lo, hi, period):
        mn = np.full(n, np.inf)
        mx = np.full(n, -np.inf)
        np.minimum.at(mn, region, lo)
        np.maximum.at(mx, region, hi)
        # A region spanning more than half the period most likely wraps:
        # move its runs in the first half one period over and measure again.
        shift = ((mx[region] - mn[region]) > period / 2) & (lo < period / 2)
        lo = np.where(shift, lo + period, lo)
        hi = np.where(shift, hi + period, hi)
        mn = np.full(n, np.inf)
        mx = np.full(n, -np.inf)
        np.minimum.at(mn, region, lo)
        np.maximum.at(mx, region, hi)
        return mn, mx

    xmin, xmax = extent(x0.astype(np.float64), x1.astype(np.float64), w)
    ymin, ymax = extent(row.astype(np.float64), row.astype(np.float64) + 1, h)
    return xmin, ymin, xmax - xmin, ymax - ymin


def _flood_fill_cpu(inputs, p, size):
    src = inputs["mask"]
    if src is None:
        return _empty(size, 4)
    mask = to_gray(src) > float(p["threshold"])
    if not mask.any():
        return _empty(size, 4)
    w, h = size
    run, region, x0, length, row = label_regions(mask)
    xmin, ymin, bw, bh = _region_boxes(region, x0, length, row, size)
    root = np.where(run >= 0, region[np.maximum(run, 0)], 0)
    out = np.stack([
        np.mod(xmin[root] / w, 1.0),
        np.mod(ymin[root] / h, 1.0),
        bw[root] / w,
        bh[root] / h,
    ], axis=2).astype(np.float32)
    out[~mask] = 0.0
    return out


register_op(OpDef(
    id="FLOOD_FILL",
    inputs=["mask"],
    output=COLOR,
    params=[Param("threshold", "float", 0.5)],
    cpu=_flood_fill_cpu,
))


# Flood Fill to Gray / Color: turn Flood Fill's per-region boxes into
# values. A region is identified by its box centre rounded to a 1/64 grid,
# which stays the same across resolutions (pixel rounding moves a centre far
# less than that), so random values don't change between draft and full.

FILL_GRAY_MODES = ("RANDOM", "GRADIENT", "SIZE")
CENTER_GRID = 64


def _region_hash(box, seed):
    cx = np.round(np.mod(box[..., 0] + box[..., 2] * 0.5, 1.0) * CENTER_GRID).astype(np.int64) % CENTER_GRID
    cy = np.round(np.mod(box[..., 1] + box[..., 3] * 0.5, 1.0) * CENTER_GRID).astype(np.int64) % CENTER_GRID
    return hash_u32(cx.astype(np.uint32) + hash_u32(cy.astype(np.uint32) + np.uint32(seed & 0xFFFFFFFF)))


def _unit(hashes):
    return hashes.astype(np.float64) / 4294967296.0


def _boxes(inputs):
    box = inputs["flood_fill"]
    if box is None or box.shape[2] != 4:
        return None, None
    return box, (box[..., 2] > 0) & (box[..., 3] > 0)


def _fill_gray_cpu(inputs, p, size):
    box, inside = _boxes(inputs)
    if box is None:
        return _empty(size)
    h1 = _region_hash(box, int(p["seed"]))
    mode = int(p["mode"])
    if mode == 0:
        value = _unit(h1)
    elif mode == 1:
        u, v = uv_grid(size)
        rel_x = np.mod(u - box[..., 0], 1.0) / np.maximum(box[..., 2], 1e-6)
        rel_y = np.mod(v - box[..., 1], 1.0) / np.maximum(box[..., 3], 1e-6)
        jitter = (_unit(hash_u32(h1 ^ np.uint32(0x9E3779B9))) - 0.5) * 2.0 * float(p["angle_random"]) * 180.0
        a = np.radians(float(p["angle"]) + jitter)
        ca, sa = np.cos(a), np.sin(a)
        # Normalised so the gradient runs 0..1 across the box in any direction.
        proj = (rel_x - 0.5) * ca + (rel_y - 0.5) * sa
        span = (np.abs(ca) + np.abs(sa)) * 0.5
        value = np.clip(proj / np.maximum(span, 1e-6) * 0.5 + 0.5, 0.0, 1.0)
    else:
        value = np.clip(np.maximum(box[..., 2], box[..., 3]), 0.0, 1.0)
    return gray(np.where(inside, value, 0.0))


GLSL_REGION = """
uint region_hash(vec4 box) {
  ivec2 c = ivec2(round(fract(box.xy + box.zw * 0.5) * 64.0)) % 64;
  return ptg_hash(uint(c.x) + ptg_hash(uint(c.y) + uint(p_seed)));
}
float unit(uint h) { return float(h) / 4294967296.0; }
"""

register_op(OpDef(
    id="FLOOD_FILL_GRAY",
    inputs=["flood_fill"],
    output=GRAY,
    params=[
        Param("mode", "int", 0),
        Param("angle", "float", 0.0),
        Param("angle_random", "float", 0.0),
        Param("seed", "int", 0),
    ],
    cpu=_fill_gray_cpu,
    glsl=GLSL_REGION + """
void main() {
  if (has_flood_fill == 0) { fragColor = ptg_gray(0.0); return; }
  vec4 box = texelFetch(in_flood_fill, ptg_px(), 0);
  if (box.z <= 0.0 || box.w <= 0.0) { fragColor = ptg_gray(0.0); return; }
  uint h1 = region_hash(box);
  float value;
  if (p_mode == 0) {
    value = unit(h1);
  } else if (p_mode == 1) {
    vec2 rel = mod(ptg_uv() - box.xy, 1.0) / max(box.zw, vec2(1e-6));
    float a = radians(p_angle + (unit(ptg_hash(h1 ^ 0x9E3779B9u)) - 0.5) * 2.0 * p_angle_random * 180.0);
    float proj = (rel.x - 0.5) * cos(a) + (rel.y - 0.5) * sin(a);
    float span = (abs(cos(a)) + abs(sin(a))) * 0.5;
    value = clamp(proj / max(span, 1e-6) * 0.5 + 0.5, 0.0, 1.0);
  } else {
    value = clamp(max(box.z, box.w), 0.0, 1.0);
  }
  fragColor = ptg_gray(value);
}
""",
))


def _fill_color_cpu(inputs, p, size):
    box, inside = _boxes(inputs)
    if box is None:
        return np.concatenate([_empty(size, 3), np.ones((size[1], size[0], 1), np.float32)], axis=2)
    h1 = _region_hash(box, int(p["seed"]))
    h2 = hash_u32(h1)
    h3 = hash_u32(h2)
    rgb = np.stack([_unit(h1), _unit(h2), _unit(h3)], axis=2)
    rgb[~inside] = 0.0
    return np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], axis=2).astype(np.float32)


register_op(OpDef(
    id="FLOOD_FILL_COLOR",
    inputs=["flood_fill"],
    output=COLOR,
    params=[Param("seed", "int", 0)],
    cpu=_fill_color_cpu,
    glsl=GLSL_REGION + """
void main() {
  vec4 box = has_flood_fill == 1 ? texelFetch(in_flood_fill, ptg_px(), 0) : vec4(0.0);
  if (box.z <= 0.0 || box.w <= 0.0) { fragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  uint h1 = region_hash(box);
  uint h2 = ptg_hash(h1);
  fragColor = vec4(unit(h1), unit(h2), unit(ptg_hash(h2)), 1.0);
}
""",
))


# ---------------------------------------------------------------------------
# Tile Sampler: one instance of a pattern per grid cell, with random offset,
# size, rotation and brightness. Instances are combined with max, like
# stacking stones. Without a pattern it stamps soft discs. Seamless because
# the grid divides the texture and instance positions wrap.

SAMPLER_REACH = 2  # neighbouring cells searched on each side


def _sampler_params(p):
    scale = min(max(float(p["scale"]), 0.01), 1.0)
    scale_random = min(max(float(p["scale_random"]), 0.0), 1.0)
    return (max(1, int(p["count_x"])), max(1, int(p["count_y"])), scale, scale_random,
            min(max(float(p["position_random"]), 0.0), 0.5))


def _tile_sampler_cpu(inputs, p, size):
    pattern = inputs["pattern"]
    pat = to_gray(pattern)[:, :, None] if pattern is not None else None
    nx, ny, scale, scale_random, jitter = _sampler_params(p)
    u, v = uv_grid(size)
    seed = np.uint32(int(p["seed"]) & 0xFFFFFFFF)
    row_offset = float(p["offset"])
    out = np.zeros(u.shape)
    base_row = np.floor(v * ny)
    for dy in range(-SAMPLER_REACH, SAMPLER_REACH + 1):
        row = base_row + dy
        shift = row_offset * row
        base_col = np.floor(u * nx - shift)
        for dx in range(-SAMPLER_REACH, SAMPLER_REACH + 1):
            col = base_col + dx
            wc = np.mod(col, nx).astype(np.uint32)
            wr = np.mod(row, ny).astype(np.uint32)
            h1 = hash_u32(wc + hash_u32(wr + seed))
            h2, h3, h4, h5, h6 = (hash_u32(h1 + np.uint32(k)) for k in range(2, 7))
            if float(p["drop"]) > 0:
                keep = _unit(h6) >= float(p["drop"])
            else:
                keep = True
            cx = (col + shift + 0.5 + (_unit(h2) - 0.5) * 2 * jitter) / nx
            cy = (row + 0.5 + (_unit(h3) - 0.5) * 2 * jitter) / ny
            du = u - cx
            dv = v - cy
            du -= np.round(du)
            dv -= np.round(dv)
            s = scale * (1.0 + (_unit(h4) - 0.5) * 2 * scale_random)
            a = np.radians(float(p["rotation"]) + (_unit(h5) - 0.5) * 2 * float(p["rotation_random"]) * 180.0)
            lx = (du * np.cos(a) + dv * np.sin(a)) * nx / s + 0.5
            ly = (-du * np.sin(a) + dv * np.cos(a)) * ny / s + 0.5
            inside = (lx >= 0) & (lx <= 1) & (ly >= 0) & (ly <= 1) & keep
            if pat is not None:
                value = bilinear(pat, np.clip(lx, 0, 1), np.clip(ly, 0, 1))[..., 0]
            else:
                d = np.sqrt((lx - 0.5) ** 2 + (ly - 0.5) ** 2)
                t = np.clip((0.5 - d) / 0.1, 0.0, 1.0)
                value = t * t * (3 - 2 * t)
            value = value * (1.0 - float(p["value_random"]) * _unit(hash_u32(h1 ^ np.uint32(0x51ED270B))))
            out = np.maximum(out, np.where(inside, value, 0.0))
    return gray(out)


register_op(OpDef(
    id="TILE_SAMPLER",
    inputs=["pattern"],
    output=GRAY,
    params=[
        Param("count_x", "int", 8),
        Param("count_y", "int", 8),
        Param("offset", "float", 0.0),
        Param("scale", "float", 0.8),
        Param("scale_random", "float", 0.0),
        Param("position_random", "float", 0.0),
        Param("rotation", "float", 0.0),
        Param("rotation_random", "float", 0.0),
        Param("value_random", "float", 0.0),
        Param("drop", "float", 0.0),
        Param("seed", "int", 0),
    ],
    cpu=_tile_sampler_cpu,
    glsl=GLSL_BILINEAR + """
float unit(uint h) { return float(h) / 4294967296.0; }
void main() {
  int nx = max(1, p_count_x), ny = max(1, p_count_y);
  float scale = clamp(p_scale, 0.01, 1.0);
  float scale_random = clamp(p_scale_random, 0.0, 1.0);
  float jitter = clamp(p_position_random, 0.0, 0.5);
  vec2 uv = ptg_uv();
  float out_v = 0.0;
  float base_row = floor(uv.y * float(ny));
  for (int dy = -2; dy <= 2; dy++) {
    float row = base_row + float(dy);
    float shift = p_offset * row;
    float base_col = floor(uv.x * float(nx) - shift);
    for (int dx = -2; dx <= 2; dx++) {
      float col = base_col + float(dx);
      uint wc = uint(mod(col, float(nx)));
      uint wr = uint(mod(row, float(ny)));
      uint h1 = ptg_hash(wc + ptg_hash(wr + uint(p_seed)));
      if (p_drop > 0.0 && unit(ptg_hash(h1 + 6u)) < p_drop) continue;
      vec2 c = vec2((col + shift + 0.5 + (unit(ptg_hash(h1 + 2u)) - 0.5) * 2.0 * jitter) / float(nx),
                    (row + 0.5 + (unit(ptg_hash(h1 + 3u)) - 0.5) * 2.0 * jitter) / float(ny));
      vec2 d = uv - c;
      d -= round(d);
      float s = scale * (1.0 + (unit(ptg_hash(h1 + 4u)) - 0.5) * 2.0 * scale_random);
      float a = radians(p_rotation + (unit(ptg_hash(h1 + 5u)) - 0.5) * 2.0 * p_rotation_random * 180.0);
      vec2 l = vec2((d.x * cos(a) + d.y * sin(a)) * float(nx), (-d.x * sin(a) + d.y * cos(a)) * float(ny)) / s + 0.5;
      if (any(lessThan(l, vec2(0.0))) || any(greaterThan(l, vec2(1.0)))) continue;
      float value;
      if (has_pattern == 1) {
        value = ptg_luma(ptg_bilinear(in_pattern, l));
      } else {
        float t = clamp((0.5 - length(l - 0.5)) / 0.1, 0.0, 1.0);
        value = t * t * (3.0 - 2.0 * t);
      }
      value *= 1.0 - p_value_random * unit(ptg_hash(h1 ^ 0x51ED270Bu));
      out_v = max(out_v, value);
    }
  }
  fragColor = ptg_gray(out_v);
}
""",
))


# ---------------------------------------------------------------------------
# Distance fields (jump flooding, wrapping), for Distance and Bevel.

FAR = 1e9


def distance_to(seeds, max_px):
    """Approximate Euclidean distance in pixels from every pixel to the
    nearest True pixel of seeds, wrapping at the edges. Distances beyond
    max_px may be reported as FAR."""
    h, w = seeds.shape
    if not seeds.any():
        return np.full((h, w), FAR)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.int32)
    # Nearest seed found so far. "None yet" is a point 4 sizes away, whose
    # squared wrapped distance (at least 9 sizes squared) loses to any real
    # seed and still fits in int32 up to 4096 px.
    none = -4 * max(w, h)
    sx = np.where(seeds, xs, none).astype(np.int32)
    sy = np.where(seeds, ys, none).astype(np.int32)

    def dist2(cx, cy):
        dx = np.abs(xs - cx)
        dx = np.minimum(dx, w - dx)
        dy = np.abs(ys - cy)
        dy = np.minimum(dy, h - dy)
        return dx * dx + dy * dy

    best = dist2(sx, sy)
    step = 1
    while step < max_px:
        step *= 2
    steps = []
    while step >= 1:
        steps.append(step)
        step //= 2
    for step in steps + [1]:  # a final 1-step pass fixes most JFA errors
        for oy in (-step, 0, step):
            for ox in (-step, 0, step):
                if ox == 0 and oy == 0:
                    continue
                cx = np.roll(sx, (oy, ox), axis=(0, 1))
                cy = np.roll(sy, (oy, ox), axis=(0, 1))
                d = dist2(cx, cy)
                closer = d < best
                np.copyto(sx, cx, where=closer)
                np.copyto(sy, cy, where=closer)
                np.copyto(best, d, where=closer)
    out = np.sqrt(best.astype(np.float64))
    out[best >= 9 * max(w, h) ** 2] = FAR
    return out


def _distance_cpu(inputs, p, size):
    src = inputs["mask"]
    if src is None:
        return _empty(size)
    mask = to_gray(src) > 0.5
    reach = max(float(p["distance"]), 1e-4) * size[0]
    d = distance_to(mask, reach)
    return gray(np.clip(1.0 - d / reach, 0.0, 1.0))


register_op(OpDef(
    id="DISTANCE",
    inputs=["mask"],
    output=GRAY,
    params=[Param("distance", "float", 0.05)],
    cpu=_distance_cpu,
))


def _bevel_cpu(inputs, p, size):
    src = inputs["mask"]
    if src is None:
        return _empty(size)
    mask = to_gray(src) > 0.5
    width = max(float(p["width"]), 1e-4) * size[0]
    inside = distance_to(~mask, width)
    t = np.clip(inside / width, 0.0, 1.0)
    if int(p["profile"]) == 1:  # round
        t = np.sqrt(1.0 - (1.0 - t) ** 2)
    elif int(p["profile"]) == 2:  # smooth
        t = t * t * (3.0 - 2.0 * t)
    return gray(np.where(mask, t, 0.0))


BEVEL_PROFILES = ("LINEAR", "ROUND", "SMOOTH")

register_op(OpDef(
    id="BEVEL",
    inputs=["mask"],
    output=GRAY,
    params=[Param("width", "float", 0.03), Param("profile", "int", 1)],
    cpu=_bevel_cpu,
))
