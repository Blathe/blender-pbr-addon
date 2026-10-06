"""Operation definitions shared by the CPU and GPU backends.

Each op has a numpy implementation (reference and fallback) and a GLSL
fragment body. This module must not import bpy so it can be tested with
plain Python.

Conventions:
- Buffers are (height, width, channels) float32 arrays, row 0 at the bottom,
  matching Blender image pixel order. Channels is 1 (grayscale) or 4 (RGBA).
- Every op samples with wrap-around, so all outputs tile seamlessly.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

GRAY = "GRAY"
COLOR = "COLOR"
# Output kind for ops whose result is color if any input is color.
SAME = "SAME"

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class Param:
    name: str
    kind: str  # "float", "int" or "color"
    default: object


@dataclass
class OpDef:
    id: str
    inputs: list
    output: str
    params: list
    cpu: Callable
    # None marks a host op: one that can't be a single fragment pass (for
    # example flood fill, which needs whole connected regions). The GPU
    # backend reads its inputs back, runs the numpy version and uploads.
    glsl: object = None
    defaults: dict = field(default_factory=dict)

    def __post_init__(self):
        self.defaults = {p.name: p.default for p in self.params}


OPS = {}


def register_op(op):
    OPS[op.id] = op
    return op


# ---------------------------------------------------------------------------
# numpy helpers

def uv_grid(size):
    w, h = size
    u = (np.arange(w, dtype=np.float32) + 0.5) / w
    v = (np.arange(h, dtype=np.float32) + 0.5) / h
    return np.meshgrid(u, v)


def to_gray(buf):
    if buf.shape[2] == 1:
        return buf[:, :, 0]
    return buf[:, :, :3] @ LUMA


def to_rgba(buf):
    if buf.shape[2] == 4:
        return buf
    g = buf[:, :, 0]
    return np.stack([g, g, g, np.ones_like(g)], axis=2)


def gray(arr):
    return arr.astype(np.float32)[:, :, None]


def hash_u32(x):
    x = np.asarray(x, dtype=np.uint32).copy()
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def any_color(*bufs):
    return any(b is not None and b.shape[2] == 4 for b in bufs)


# ---------------------------------------------------------------------------
# GLSL shared by every op. Push constants and samplers are declared by the
# GPU backend from the op definition; inputs are named in_<name> with a
# has_<name> flag, params are named p_<name>.

GLSL_COMMON = """
uint ptg_hash(uint x) {
  x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
  return x;
}
// GLSL leaves % undefined for negative operands, so wrap with floor instead.
ivec2 ptg_mod(ivec2 a, ivec2 n) { return a - n * ivec2(floor(vec2(a) / vec2(n))); }
int ptg_mod(int a, int n) { return a - n * int(floor(float(a) / float(n))); }
ivec2 ptg_wrap(ivec2 p) { return ptg_mod(p, ptg_size); }
float ptg_luma(vec4 c) { return dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)); }
ivec2 ptg_px() { return ivec2(gl_FragCoord.xy); }
vec2 ptg_uv() { return (vec2(ptg_px()) + 0.5) / vec2(ptg_size); }
vec4 ptg_gray(float v) { return vec4(v, v, v, 1.0); }
"""


# ---------------------------------------------------------------------------
# Perlin noise (tileable: the lattice period divides the texture exactly)

def _perlin_octave(u, v, period, seed):
    px, py = u * period, v * period
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    fx, fy = px - x0, py - y0

    def grad_dot(ix, iy, dx, dy):
        h = hash_u32((ix % period).astype(np.uint32) + hash_u32((iy % period).astype(np.uint32) + np.uint32(seed)))
        a = h.astype(np.float64) * (2.0 * np.pi / 4294967296.0)
        return np.cos(a) * dx + np.sin(a) * dy

    n00 = grad_dot(x0, y0, fx, fy)
    n10 = grad_dot(x0 + 1, y0, fx - 1, fy)
    n01 = grad_dot(x0, y0 + 1, fx, fy - 1)
    n11 = grad_dot(x0 + 1, y0 + 1, fx - 1, fy - 1)
    sx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    sy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    nx0 = n00 + sx * (n10 - n00)
    nx1 = n01 + sx * (n11 - n01)
    return nx0 + sy * (nx1 - nx0)


def _perlin_cpu(inputs, p, size):
    u, v = uv_grid(size)
    total, amp, acc = 0.0, 1.0, np.zeros_like(u, dtype=np.float64)
    base = max(1, int(p["scale"]))
    for o in range(max(1, int(p["octaves"]))):
        seed = int(hash_u32(np.uint32((int(p["seed"]) + o * 1013) & 0xFFFFFFFF)))
        acc += amp * _perlin_octave(u, v, base << o, seed)
        total += amp
        amp *= p["persistence"]
    return gray(np.clip(0.5 + 0.7 * acc / total, 0.0, 1.0))


register_op(OpDef(
    id="PERLIN",
    inputs=[],
    output=GRAY,
    params=[
        Param("scale", "int", 4),
        Param("octaves", "int", 4),
        Param("persistence", "float", 0.5),
        Param("seed", "int", 0),
    ],
    cpu=_perlin_cpu,
    glsl="""
float grad_dot(int ix, int iy, int period, uint s, vec2 d) {
  uint h = ptg_hash(uint(ix % period) + ptg_hash(uint(iy % period) + s));
  float a = float(h) * (6.283185307179586 / 4294967296.0);
  return cos(a) * d.x + sin(a) * d.y;
}
float perlin_octave(vec2 uv, int period, uint s) {
  vec2 p = uv * float(period);
  ivec2 i0 = ivec2(floor(p));
  vec2 f = p - vec2(i0);
  float n00 = grad_dot(i0.x, i0.y, period, s, f);
  float n10 = grad_dot(i0.x + 1, i0.y, period, s, f - vec2(1.0, 0.0));
  float n01 = grad_dot(i0.x, i0.y + 1, period, s, f - vec2(0.0, 1.0));
  float n11 = grad_dot(i0.x + 1, i0.y + 1, period, s, f - vec2(1.0, 1.0));
  vec2 t = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  return mix(mix(n00, n10, t.x), mix(n01, n11, t.x), t.y);
}
void main() {
  vec2 uv = ptg_uv();
  float total = 0.0, amp = 1.0, acc = 0.0;
  int base = max(1, p_scale);
  for (int o = 0; o < max(1, p_octaves); o++) {
    uint s = ptg_hash(uint(p_seed + o * 1013));
    acc += amp * perlin_octave(uv, base << o, s);
    total += amp;
    amp *= p_persistence;
  }
  fragColor = ptg_gray(clamp(0.5 + 0.7 * acc / total, 0.0, 1.0));
}
""",
))


# ---------------------------------------------------------------------------
# Shape: a tiled circle, square or diamond

SHAPES = ("CIRCLE", "SQUARE", "DIAMOND")


def _shape_cpu(inputs, p, size):
    u, v = uv_grid(size)
    tiles = max(1, int(p["tiling"]))
    cx = np.mod(u * tiles, 1.0) * 2.0 - 1.0
    cy = np.mod(v * tiles, 1.0) * 2.0 - 1.0
    kind = int(p["shape"])
    if kind == 0:
        d = np.sqrt(cx * cx + cy * cy)
    elif kind == 1:
        d = np.maximum(np.abs(cx), np.abs(cy))
    else:
        d = np.abs(cx) + np.abs(cy)
    soft = max(float(p["softness"]), 1e-4)
    edge = float(p["size"])
    t = np.clip((d - (edge - soft)) / soft, 0.0, 1.0)
    return gray(1.0 - t * t * (3.0 - 2.0 * t))


register_op(OpDef(
    id="SHAPE",
    inputs=[],
    output=GRAY,
    params=[
        Param("shape", "int", 0),
        Param("tiling", "int", 1),
        Param("size", "float", 0.8),
        Param("softness", "float", 0.3),
    ],
    cpu=_shape_cpu,
    glsl="""
void main() {
  vec2 c = fract(ptg_uv() * float(max(1, p_tiling))) * 2.0 - 1.0;
  float d;
  if (p_shape == 0) d = length(c);
  else if (p_shape == 1) d = max(abs(c.x), abs(c.y));
  else d = abs(c.x) + abs(c.y);
  float soft = max(p_softness, 1e-4);
  fragColor = ptg_gray(1.0 - smoothstep(p_size - soft, p_size, d));
}
""",
))


# ---------------------------------------------------------------------------
# Blend

BLEND_MODES = ("NORMAL", "ADD", "MULTIPLY", "SCREEN", "OVERLAY", "SUBTRACT", "DARKEN", "LIGHTEN")


def _blend_cpu(inputs, p, size):
    fg, bg, mask = inputs["foreground"], inputs["background"], inputs["mask"]
    w, h = size
    color = any_color(fg, bg)
    conv = to_rgba if color else (lambda b: b)
    zero = np.zeros((h, w, 4 if color else 1), dtype=np.float32)
    if color:
        zero[:, :, 3] = 1.0
    a = conv(fg) if fg is not None else zero
    b = conv(bg) if bg is not None else zero
    mode = int(p["mode"])
    if mode == 0:
        r = a
    elif mode == 1:
        r = a + b
    elif mode == 2:
        r = a * b
    elif mode == 3:
        r = 1.0 - (1.0 - a) * (1.0 - b)
    elif mode == 4:
        r = np.where(b < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    elif mode == 5:
        r = b - a
    elif mode == 6:
        r = np.minimum(a, b)
    else:
        r = np.maximum(a, b)
    k = float(p["opacity"])
    if mask is not None:
        k = k * to_gray(mask)[:, :, None]
    out = np.clip(b + (r - b) * k, 0.0, 1.0).astype(np.float32)
    if color:
        out[:, :, 3] = 1.0
    return out


register_op(OpDef(
    id="BLEND",
    inputs=["foreground", "background", "mask"],
    output=SAME,
    params=[Param("mode", "int", 0), Param("opacity", "float", 1.0)],
    cpu=_blend_cpu,
    glsl="""
void main() {
  ivec2 px = ptg_px();
  vec4 a = has_foreground != 0 ? texelFetch(in_foreground, px, 0) : vec4(0.0, 0.0, 0.0, 1.0);
  vec4 b = has_background != 0 ? texelFetch(in_background, px, 0) : vec4(0.0, 0.0, 0.0, 1.0);
  vec4 r;
  if (p_mode == 0) r = a;
  else if (p_mode == 1) r = a + b;
  else if (p_mode == 2) r = a * b;
  else if (p_mode == 3) r = 1.0 - (1.0 - a) * (1.0 - b);
  else if (p_mode == 4) r = mix(2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b), step(0.5, b));
  else if (p_mode == 5) r = b - a;
  else if (p_mode == 6) r = min(a, b);
  else r = max(a, b);
  float k = p_opacity;
  if (has_mask != 0) k *= ptg_luma(texelFetch(in_mask, px, 0));
  fragColor = vec4(clamp(mix(b, r, k), 0.0, 1.0).rgb, 1.0);
}
""",
))


# ---------------------------------------------------------------------------
# Levels

def _levels_cpu(inputs, p, size):
    src = inputs["input"]
    if src is None:
        return gray(np.zeros((size[1], size[0]), dtype=np.float32))
    span = max(float(p["in_high"]) - float(p["in_low"]), 1e-5)
    rgb = src if src.shape[2] == 1 else src[:, :, :3]
    t = np.clip((rgb - float(p["in_low"])) / span, 0.0, 1.0)
    t = np.power(t, 1.0 / max(float(p["gamma"]), 1e-3))
    t = float(p["out_low"]) + t * (float(p["out_high"]) - float(p["out_low"]))
    if src.shape[2] == 1:
        return t.astype(np.float32)
    out = src.copy()
    out[:, :, :3] = t
    return out


register_op(OpDef(
    id="LEVELS",
    inputs=["input"],
    output=SAME,
    params=[
        Param("in_low", "float", 0.0),
        Param("in_high", "float", 1.0),
        Param("gamma", "float", 1.0),
        Param("out_low", "float", 0.0),
        Param("out_high", "float", 1.0),
    ],
    cpu=_levels_cpu,
    glsl="""
void main() {
  if (has_input == 0) { fragColor = ptg_gray(0.0); return; }
  vec4 c = texelFetch(in_input, ptg_px(), 0);
  float span = max(p_in_high - p_in_low, 1e-5);
  vec3 t = clamp((c.rgb - p_in_low) / span, 0.0, 1.0);
  t = pow(t, vec3(1.0 / max(p_gamma, 1e-3)));
  fragColor = vec4(p_out_low + t * (p_out_high - p_out_low), c.a);
}
""",
))


# ---------------------------------------------------------------------------
# Normal from height (Sobel, wrapped). Strength is resolution independent.

NORMAL_FORMATS = ("OPENGL", "DIRECTX")


def _normal_cpu(inputs, p, size):
    w, h = size
    src = inputs["height"]
    if src is None:
        out = np.zeros((h, w, 4), dtype=np.float32)
        out[:, :] = (0.5, 0.5, 1.0, 1.0)
        return out
    g = to_gray(src)

    def s(dx, dy):
        return np.roll(np.roll(g, -dy, axis=0), -dx, axis=1)

    gx = (s(1, -1) + 2 * s(1, 0) + s(1, 1)) - (s(-1, -1) + 2 * s(-1, 0) + s(-1, 1))
    gy = (s(-1, 1) + 2 * s(0, 1) + s(1, 1)) - (s(-1, -1) + 2 * s(0, -1) + s(1, -1))
    k = float(p["intensity"]) / 32.0
    nx = -gx / 8.0 * w * k
    ny = -gy / 8.0 * h * k
    if int(p["format"]) == 1:
        ny = -ny
    length = np.sqrt(nx * nx + ny * ny + 1.0)
    out = np.empty((h, w, 4), dtype=np.float32)
    out[:, :, 0] = nx / length * 0.5 + 0.5
    out[:, :, 1] = ny / length * 0.5 + 0.5
    out[:, :, 2] = 1.0 / length * 0.5 + 0.5
    out[:, :, 3] = 1.0
    return out


register_op(OpDef(
    id="NORMAL",
    inputs=["height"],
    output=COLOR,
    params=[Param("intensity", "float", 1.0), Param("format", "int", 0)],
    cpu=_normal_cpu,
    glsl="""
float hgt(ivec2 p) { return ptg_luma(texelFetch(in_height, ptg_wrap(p), 0)); }
void main() {
  if (has_height == 0) { fragColor = vec4(0.5, 0.5, 1.0, 1.0); return; }
  ivec2 p = ptg_px();
  float gx = (hgt(p + ivec2(1, -1)) + 2.0 * hgt(p + ivec2(1, 0)) + hgt(p + ivec2(1, 1)))
           - (hgt(p + ivec2(-1, -1)) + 2.0 * hgt(p + ivec2(-1, 0)) + hgt(p + ivec2(-1, 1)));
  float gy = (hgt(p + ivec2(-1, 1)) + 2.0 * hgt(p + ivec2(0, 1)) + hgt(p + ivec2(1, 1)))
           - (hgt(p + ivec2(-1, -1)) + 2.0 * hgt(p + ivec2(0, -1)) + hgt(p + ivec2(1, -1)));
  float k = p_intensity / 32.0;
  vec3 n = vec3(-gx / 8.0 * float(ptg_size.x) * k, -gy / 8.0 * float(ptg_size.y) * k, 1.0);
  if (p_format == 1) n.y = -n.y;
  fragColor = vec4(normalize(n) * 0.5 + 0.5, 1.0);
}
""",
))


# ---------------------------------------------------------------------------
# Gradient map: grayscale to color through three stops (stylized ramps)

def _gradient_map_cpu(inputs, p, size):
    w, h = size
    src = inputs["input"]
    t = to_gray(src) if src is not None else np.zeros((h, w), dtype=np.float32)
    lo, mid, hi = (np.asarray(p[k], dtype=np.float32) for k in ("color_low", "color_mid", "color_high"))
    m = min(max(float(p["mid_position"]), 1e-3), 1.0 - 1e-3)
    a = np.clip(t / m, 0.0, 1.0)[:, :, None]
    b = np.clip((t - m) / (1.0 - m), 0.0, 1.0)[:, :, None]
    out = np.where(t[:, :, None] < m, lo + (mid - lo) * a, mid + (hi - mid) * b)
    out = out.astype(np.float32)
    out[:, :, 3] = 1.0
    return out


register_op(OpDef(
    id="GRADIENT_MAP",
    inputs=["input"],
    output=COLOR,
    params=[
        Param("color_low", "color", (0.05, 0.03, 0.08, 1.0)),
        Param("color_mid", "color", (0.35, 0.2, 0.12, 1.0)),
        Param("color_high", "color", (0.9, 0.7, 0.4, 1.0)),
        Param("mid_position", "float", 0.5),
    ],
    cpu=_gradient_map_cpu,
    glsl="""
void main() {
  float t = has_input != 0 ? ptg_luma(texelFetch(in_input, ptg_px(), 0)) : 0.0;
  float m = clamp(p_mid_position, 1e-3, 1.0 - 1e-3);
  vec4 c = t < m ? mix(p_color_low, p_color_mid, clamp(t / m, 0.0, 1.0))
                 : mix(p_color_mid, p_color_high, clamp((t - m) / (1.0 - m), 0.0, 1.0));
  fragColor = vec4(c.rgb, 1.0);
}
""",
))


# ---------------------------------------------------------------------------
# Output: passes its input through; the evaluator collects these per channel.

def _output_cpu(inputs, p, size):
    src = inputs["input"]
    if src is None:
        return gray(np.zeros((size[1], size[0]), dtype=np.float32))
    return src


register_op(OpDef(
    id="OUTPUT",
    inputs=["input"],
    output=SAME,
    params=[],
    cpu=_output_cpu,
    glsl="""
void main() {
  fragColor = has_input != 0 ? texelFetch(in_input, ptg_px(), 0) : ptg_gray(0.0);
}
""",
))


# ---------------------------------------------------------------------------
# Stylized nodes
# ---------------------------------------------------------------------------

def _sobel(g, size):
    """Wrapped Sobel slope of a (h, w) height array, per uv unit / 8."""
    w, h = size

    def s(dx, dy):
        return np.roll(np.roll(g, -dy, axis=0), -dx, axis=1)

    gx = (s(1, -1) + 2 * s(1, 0) + s(1, 1)) - (s(-1, -1) + 2 * s(-1, 0) + s(-1, 1))
    gy = (s(-1, 1) + 2 * s(0, 1) + s(1, 1)) - (s(-1, -1) + 2 * s(0, -1) + s(1, -1))
    return gx / 8.0 * w, gy / 8.0 * h


# Reads the op's in_height sampler; samplers are not passed as arguments to
# stay portable across Blender's GPU backends.
GLSL_SOBEL = """
float sob_h(ivec2 p) { return ptg_luma(texelFetch(in_height, ptg_wrap(p), 0)); }
vec2 sobel_slope() {
  ivec2 p = ptg_px();
  float gx = (sob_h(p + ivec2(1, -1)) + 2.0 * sob_h(p + ivec2(1, 0)) + sob_h(p + ivec2(1, 1)))
           - (sob_h(p + ivec2(-1, -1)) + 2.0 * sob_h(p + ivec2(-1, 0)) + sob_h(p + ivec2(-1, 1)));
  float gy = (sob_h(p + ivec2(-1, 1)) + 2.0 * sob_h(p + ivec2(0, 1)) + sob_h(p + ivec2(1, 1)))
           - (sob_h(p + ivec2(-1, -1)) + 2.0 * sob_h(p + ivec2(0, -1)) + sob_h(p + ivec2(1, -1)));
  return vec2(gx / 8.0 * float(ptg_size.x), gy / 8.0 * float(ptg_size.y));
}
"""


def _base_or_gray(base, size, value):
    if base is not None:
        return to_rgba(base)
    w, h = size
    out = np.full((h, w, 4), value, dtype=np.float32)
    out[:, :, 3] = 1.0
    return out


# Posterize: quantize values into a few flat bands

def _posterize_cpu(inputs, p, size):
    src = inputs["input"]
    if src is None:
        return gray(np.zeros((size[1], size[0]), dtype=np.float32))
    steps = max(2, int(p["steps"]))
    rgb = src if src.shape[2] == 1 else src[:, :, :3]
    q = np.minimum(np.floor(np.clip(rgb, 0.0, 1.0) * steps), steps - 1) / (steps - 1)
    if src.shape[2] == 1:
        return q.astype(np.float32)
    out = src.copy()
    out[:, :, :3] = q
    return out


register_op(OpDef(
    id="POSTERIZE",
    inputs=["input"],
    output=SAME,
    params=[Param("steps", "int", 4)],
    cpu=_posterize_cpu,
    glsl="""
void main() {
  if (has_input == 0) { fragColor = ptg_gray(0.0); return; }
  vec4 c = texelFetch(in_input, ptg_px(), 0);
  float steps = float(max(2, p_steps));
  vec3 q = min(floor(clamp(c.rgb, 0.0, 1.0) * steps), vec3(steps - 1.0)) / (steps - 1.0);
  fragColor = vec4(q, c.a);
}
""",
))


# Height to Light: paints directional light and cavity darkening into base
# color from a height map. Flat areas keep their color.

def _height_to_light_cpu(inputs, p, size):
    base = _base_or_gray(inputs["base"], size, 0.5)
    src = inputs["height"]
    if src is None:
        return base
    hgt = to_gray(src)
    sx, sy = _sobel(hgt, size)
    k = float(p["depth"]) / 32.0
    nx, ny, nz = -sx * k, -sy * k, np.ones_like(hgt)
    length = np.sqrt(nx * nx + ny * ny + 1.0)
    az, el = np.radians(float(p["angle"])), np.radians(min(max(float(p["elevation"]), 1.0), 90.0))
    lx, ly, lz = np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)
    lam = np.clip((nx * lx + ny * ly + nz * lz) / length, 0.0, None) / lz
    shade = 1.0 + (lam - 1.0) * float(p["light"])
    cavity = 1.0 - float(p["cavity"]) * (1.0 - hgt)
    out = base.copy()
    out[:, :, :3] = np.clip(base[:, :, :3] * (shade * cavity)[:, :, None], 0.0, 1.0)
    return out


register_op(OpDef(
    id="HEIGHT_TO_LIGHT",
    inputs=["height", "base"],
    output=COLOR,
    params=[
        Param("angle", "float", 135.0),
        Param("elevation", "float", 45.0),
        Param("depth", "float", 1.0),
        Param("light", "float", 0.6),
        Param("cavity", "float", 0.4),
    ],
    cpu=_height_to_light_cpu,
    glsl=GLSL_SOBEL + """
void main() {
  vec4 base = has_base != 0 ? texelFetch(in_base, ptg_px(), 0) : vec4(0.5, 0.5, 0.5, 1.0);
  if (has_height == 0) { fragColor = base; return; }
  float h = ptg_luma(texelFetch(in_height, ptg_px(), 0));
  vec2 s = sobel_slope() * (p_depth / 32.0);
  vec3 n = normalize(vec3(-s, 1.0));
  float az = radians(p_angle), el = radians(clamp(p_elevation, 1.0, 90.0));
  vec3 l = vec3(cos(az) * cos(el), sin(az) * cos(el), sin(el));
  float lam = max(dot(n, l), 0.0) / l.z;
  float shade = 1.0 + (lam - 1.0) * p_light;
  float cavity = 1.0 - p_cavity * (1.0 - h);
  fragColor = vec4(clamp(base.rgb * shade * cavity, 0.0, 1.0), 1.0);
}
""",
))


# Edge Highlight: bright painted rims where the height is convex (higher
# than its surroundings), measured over a radius relative to the texture.

EDGE_DIRS = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]


def _edge_offsets(radius_px):
    offs = []
    for r in (radius_px, radius_px * 0.5):
        for cx, cy in EDGE_DIRS:
            offs.append((int(np.floor(cx * r + 0.5)), int(np.floor(cy * r + 0.5))))
    return offs


def _edge_mask(hgt, p, size):
    radius = max(1.0, float(p["width"]) * size[0])
    offs = _edge_offsets(radius)
    avg = sum(np.roll(np.roll(hgt, -dy, axis=0), -dx, axis=1) for dx, dy in offs) / len(offs)
    conv = hgt - avg
    lo = float(p["threshold"])
    t = np.clip((conv - lo) / max(float(p["softness"]), 1e-4), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _edge_highlight_cpu(inputs, p, size):
    base = _base_or_gray(inputs["base"], size, 0.0)
    src = inputs["height"]
    if src is None:
        return base
    mask = _edge_mask(to_gray(src), p, size) * float(p["strength"])
    color = np.asarray(p["color"], dtype=np.float32)[:3]
    out = base.copy()
    out[:, :, :3] = np.clip(base[:, :, :3] + (color - base[:, :, :3]) * mask[:, :, None], 0.0, 1.0)
    return out


register_op(OpDef(
    id="EDGE_HIGHLIGHT",
    inputs=["height", "base"],
    output=COLOR,
    params=[
        Param("color", "color", (1.0, 0.9, 0.7, 1.0)),
        Param("width", "float", 0.01),
        Param("threshold", "float", 0.01),
        Param("softness", "float", 0.05),
        Param("strength", "float", 1.0),
    ],
    cpu=_edge_highlight_cpu,
    glsl="""
void main() {
  vec4 base = has_base != 0 ? texelFetch(in_base, ptg_px(), 0) : vec4(0.0, 0.0, 0.0, 1.0);
  if (has_height == 0) { fragColor = base; return; }
  ivec2 p = ptg_px();
  float h = ptg_luma(texelFetch(in_height, p, 0));
  float radius = max(1.0, p_width * float(ptg_size.x));
  float sum = 0.0;
  for (int ring = 0; ring < 2; ring++) {
    float r = ring == 0 ? radius : radius * 0.5;
    for (int i = 0; i < 8; i++) {
      float a = 6.283185307179586 * float(i) / 8.0;
      ivec2 o = ivec2(floor(cos(a) * r + 0.5), floor(sin(a) * r + 0.5));
      sum += ptg_luma(texelFetch(in_height, ptg_wrap(p + o), 0));
    }
  }
  float conv = h - sum / 16.0;
  float t = clamp((conv - p_threshold) / max(p_softness, 1e-4), 0.0, 1.0);
  float mask = t * t * (3.0 - 2.0 * t) * p_strength;
  fragColor = vec4(clamp(mix(base.rgb, p_color.rgb, mask), 0.0, 1.0), 1.0);
}
""",
))


# ---------------------------------------------------------------------------
# MVP generators and filters
# ---------------------------------------------------------------------------

def hash01(x):
    return hash_u32(x).astype(np.float64) / 4294967296.0


def bilinear(img, u, v):
    """Sample an (h, w, c) image at uv arrays with wrap-around."""
    h, w = img.shape[:2]
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0, y0 = x0 % w, y0 % h
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


GLSL_BILINEAR = """
vec4 ptg_bilinear(sampler2D t, vec2 uv) {
  vec2 x = uv * vec2(ptg_size) - 0.5;
  ivec2 i0 = ivec2(floor(x));
  vec2 f = x - vec2(i0);
  vec4 a = texelFetch(t, ptg_wrap(i0), 0);
  vec4 b = texelFetch(t, ptg_wrap(i0 + ivec2(1, 0)), 0);
  vec4 c = texelFetch(t, ptg_wrap(i0 + ivec2(0, 1)), 0);
  vec4 d = texelFetch(t, ptg_wrap(i0 + ivec2(1, 1)), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
"""


# Voronoi (Worley) noise, tileable: cell coordinates wrap at the scale.

VORONOI_MODES = ("F1", "F2", "EDGES", "CELLS")


def _voronoi_cpu(inputs, p, size):
    u, v = uv_grid(size)
    n = max(1, int(p["scale"]))
    seed = np.uint32(int(p["seed"]) & 0xFFFFFFFF)
    rnd = float(p["randomness"])
    px, py = u * n, v * n
    cx, cy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    f1 = np.full(u.shape, 9.0)
    f2 = np.full(u.shape, 9.0)
    cell = np.zeros(u.shape)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            nx, ny = cx + ox, cy + oy
            h1 = hash_u32((nx % n).astype(np.uint32) + hash_u32((ny % n).astype(np.uint32) + seed))
            h2 = hash_u32(h1)
            fx = nx + 0.5 + (h1.astype(np.float64) / 4294967296.0 - 0.5) * rnd
            fy = ny + 0.5 + (h2.astype(np.float64) / 4294967296.0 - 0.5) * rnd
            d = np.sqrt((fx - px) ** 2 + (fy - py) ** 2)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            cell = np.where(closer, hash_u32(h2).astype(np.float64) / 4294967296.0, cell)
            f1 = np.where(closer, d, f1)
    mode = int(p["mode"])
    if mode == 0:
        out = f1
    elif mode == 1:
        out = f2 / 1.5
    elif mode == 2:
        out = (f2 - f1) * 2.0
    else:
        out = cell
    return gray(np.clip(out, 0.0, 1.0))


register_op(OpDef(
    id="VORONOI",
    inputs=[],
    output=GRAY,
    params=[
        Param("scale", "int", 6),
        Param("mode", "int", 0),
        Param("randomness", "float", 1.0),
        Param("seed", "int", 0),
    ],
    cpu=_voronoi_cpu,
    glsl="""
void main() {
  int n = max(1, p_scale);
  vec2 pt = ptg_uv() * float(n);
  ivec2 c = ivec2(floor(pt));
  float f1 = 9.0, f2 = 9.0, cell = 0.0;
  for (int oy = -1; oy <= 1; oy++) {
    for (int ox = -1; ox <= 1; ox++) {
      ivec2 nc = c + ivec2(ox, oy);
      ivec2 wc = ptg_mod(nc, ivec2(n));
      uint h1 = ptg_hash(uint(wc.x) + ptg_hash(uint(wc.y) + uint(p_seed)));
      uint h2 = ptg_hash(h1);
      vec2 fp = vec2(nc) + 0.5 + (vec2(float(h1), float(h2)) / 4294967296.0 - 0.5) * p_randomness;
      float d = length(fp - pt);
      if (d < f1) {
        f2 = f1;
        f1 = d;
        cell = float(ptg_hash(h2)) / 4294967296.0;
      } else {
        f2 = min(f2, d);
      }
    }
  }
  float o;
  if (p_mode == 0) o = f1;
  else if (p_mode == 1) o = f2 / 1.5;
  else if (p_mode == 2) o = (f2 - f1) * 2.0;
  else o = cell;
  fragColor = ptg_gray(clamp(o, 0.0, 1.0));
}
""",
))


# Tile Generator: bricks or tiles with gaps, bevels and per-tile height.
# Seamless vertically when offset * tiles_y is a whole number.

def _tile_cpu(inputs, p, size):
    u, v = uv_grid(size)
    tx, ty = max(1, int(p["tiles_x"])), max(1, int(p["tiles_y"]))
    row = np.floor(v * ty)
    su = u * tx + float(p["offset"]) * row
    col = np.mod(np.floor(su), tx)
    lu, lv = su - np.floor(su), v * ty - row
    ex = np.minimum(lu, 1 - lu) / tx
    ey = np.minimum(lv, 1 - lv) / ty
    e = np.minimum(ex, ey)
    cs = min(1.0 / tx, 1.0 / ty)
    g = float(p["gap"]) * cs * 0.5
    b = max(float(p["bevel"]) * cs, 1e-5)
    t = np.clip((e - g) / b, 0.0, 1.0)
    mask = t * t * (3 - 2 * t)
    seed = np.uint32(int(p["seed"]) & 0xFFFFFFFF)
    r = hash01(col.astype(np.uint32) + hash_u32(np.mod(row, ty).astype(np.uint32) + seed))
    return gray(mask * (1.0 - float(p["variation"]) * r))


register_op(OpDef(
    id="TILE",
    inputs=[],
    output=GRAY,
    params=[
        Param("tiles_x", "int", 4),
        Param("tiles_y", "int", 8),
        Param("offset", "float", 0.5),
        Param("gap", "float", 0.08),
        Param("bevel", "float", 0.15),
        Param("variation", "float", 0.3),
        Param("seed", "int", 0),
    ],
    cpu=_tile_cpu,
    glsl="""
void main() {
  vec2 uv = ptg_uv();
  int tx = max(1, p_tiles_x), ty = max(1, p_tiles_y);
  float row = floor(uv.y * float(ty));
  float su = uv.x * float(tx) + p_offset * row;
  float col = mod(floor(su), float(tx));
  float lu = su - floor(su), lv = uv.y * float(ty) - row;
  float e = min(min(lu, 1.0 - lu) / float(tx), min(lv, 1.0 - lv) / float(ty));
  float cs = min(1.0 / float(tx), 1.0 / float(ty));
  float g = p_gap * cs * 0.5;
  float b = max(p_bevel * cs, 1e-5);
  float t = clamp((e - g) / b, 0.0, 1.0);
  float mask = t * t * (3.0 - 2.0 * t);
  uint h = ptg_hash(uint(col) + ptg_hash(uint(mod(row, float(ty))) + uint(p_seed)));
  float r = float(h) / 4294967296.0;
  fragColor = ptg_gray(mask * (1.0 - p_variation * r));
}
""",
))


# Gradient: linear (sawtooth), mirrored (seamless along its axis) or radial.

GRADIENT_MODES = ("LINEAR", "MIRRORED", "RADIAL")


def _gradient_cpu(inputs, p, size):
    u, v = uv_grid(size)
    a = np.radians(float(p["angle"]))
    x = (u * np.cos(a) + v * np.sin(a)) * max(1, int(p["repeat"]))
    mode = int(p["mode"])
    if mode == 0:
        out = x - np.floor(x)
    elif mode == 1:
        out = 1.0 - np.abs(2.0 * (x - np.floor(x)) - 1.0)
    else:
        out = np.clip(1.0 - np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) * 2.0, 0.0, 1.0)
    return gray(out)


register_op(OpDef(
    id="GRADIENT",
    inputs=[],
    output=GRAY,
    params=[Param("mode", "int", 0), Param("angle", "float", 0.0), Param("repeat", "int", 1)],
    cpu=_gradient_cpu,
    glsl="""
void main() {
  vec2 uv = ptg_uv();
  float a = radians(p_angle);
  float x = dot(uv, vec2(cos(a), sin(a))) * float(max(1, p_repeat));
  float o;
  if (p_mode == 0) o = fract(x);
  else if (p_mode == 1) o = 1.0 - abs(2.0 * fract(x) - 1.0);
  else o = clamp(1.0 - length(uv - 0.5) * 2.0, 0.0, 1.0);
  fragColor = ptg_gray(o);
}
""",
))


# Blur: Gaussian over a 9 x 9 grid of bilinear taps spanning the radius.

BLUR_TAPS = 4  # taps on each side of the centre


def _blur_weights():
    k = np.arange(-BLUR_TAPS, BLUR_TAPS + 1) / BLUR_TAPS
    w = np.exp(-2.0 * k * k)
    return k, w / w.sum()


def _blur_cpu(inputs, p, size):
    src = inputs["input"]
    if src is None:
        return gray(np.zeros((size[1], size[0]), dtype=np.float32))
    radius = float(p["radius"])
    if radius <= 0.0:
        return src
    u, v = uv_grid(size)
    k, w = _blur_weights()
    tmp = sum(wi * bilinear(src, u + ki * radius, v) for ki, wi in zip(k, w))
    return sum(wi * bilinear(tmp, u, v + ki * radius * size[0] / size[1]) for ki, wi in zip(k, w)).astype(np.float32)


register_op(OpDef(
    id="BLUR",
    inputs=["input"],
    output=SAME,
    params=[Param("radius", "float", 0.01)],
    cpu=_blur_cpu,
    glsl=GLSL_BILINEAR + """
void main() {
  if (has_input == 0) { fragColor = ptg_gray(0.0); return; }
  vec2 uv = ptg_uv();
  if (p_radius <= 0.0) { fragColor = texelFetch(in_input, ptg_px(), 0); return; }
  vec2 stp = vec2(p_radius, p_radius * float(ptg_size.x) / float(ptg_size.y)) / 4.0;
  float wsum = 0.0;
  for (int i = -4; i <= 4; i++) wsum += exp(-2.0 * float(i * i) / 16.0);
  vec4 acc = vec4(0.0);
  for (int j = -4; j <= 4; j++) {
    float wj = exp(-2.0 * float(j * j) / 16.0) / wsum;
    for (int i = -4; i <= 4; i++) {
      float wi = exp(-2.0 * float(i * i) / 16.0) / wsum;
      acc += wi * wj * ptg_bilinear(in_input, uv + vec2(float(i), float(j)) * stp);
    }
  }
  fragColor = acc;
}
""",
))


# Warp: push the input along the slope of a warp map.

def _warp_cpu(inputs, p, size):
    src, warp = inputs["input"], inputs["warp"]
    if src is None:
        return gray(np.zeros((size[1], size[0]), dtype=np.float32))
    if warp is None:
        return src
    w, h = size
    g = to_gray(warp)
    dx = (np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)) * 0.5 * w
    dy = (np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0)) * 0.5 * h
    k = float(p["intensity"]) * 0.01
    u, v = uv_grid(size)
    return bilinear(src, u + dx * k, v + dy * k)


register_op(OpDef(
    id="WARP",
    inputs=["input", "warp"],
    output=SAME,
    params=[Param("intensity", "float", 0.2)],
    cpu=_warp_cpu,
    glsl=GLSL_BILINEAR + """
float wv(ivec2 p) { return ptg_luma(texelFetch(in_warp, ptg_wrap(p), 0)); }
void main() {
  if (has_input == 0) { fragColor = ptg_gray(0.0); return; }
  if (has_warp == 0) { fragColor = texelFetch(in_input, ptg_px(), 0); return; }
  ivec2 p = ptg_px();
  vec2 d = vec2((wv(p + ivec2(1, 0)) - wv(p - ivec2(1, 0))) * 0.5 * float(ptg_size.x),
                (wv(p + ivec2(0, 1)) - wv(p - ivec2(0, 1))) * 0.5 * float(ptg_size.y));
  fragColor = ptg_bilinear(in_input, ptg_uv() + d * p_intensity * 0.01);
}
""",
))


# Transform: offset, rotate and tile the input. Stays seamless for whole
# tilings and rotations in steps of 90 degrees.

def _transform_cpu(inputs, p, size):
    src = inputs["input"]
    if src is None:
        return gray(np.zeros((size[1], size[0]), dtype=np.float32))
    u, v = uv_grid(size)
    a = np.radians(float(p["rotation"]))
    qx, qy = u - 0.5 - float(p["offset_x"]), v - 0.5 - float(p["offset_y"])
    rx = qx * np.cos(a) + qy * np.sin(a)
    ry = -qx * np.sin(a) + qy * np.cos(a)
    t = max(1, int(p["tiling"]))
    return bilinear(src, (rx + 0.5) * t, (ry + 0.5) * t)


register_op(OpDef(
    id="TRANSFORM",
    inputs=["input"],
    output=SAME,
    params=[
        Param("offset_x", "float", 0.0),
        Param("offset_y", "float", 0.0),
        Param("rotation", "float", 0.0),
        Param("tiling", "int", 1),
    ],
    cpu=_transform_cpu,
    glsl=GLSL_BILINEAR + """
void main() {
  if (has_input == 0) { fragColor = ptg_gray(0.0); return; }
  float a = radians(p_rotation);
  vec2 q = ptg_uv() - 0.5 - vec2(p_offset_x, p_offset_y);
  vec2 r = vec2(q.x * cos(a) + q.y * sin(a), -q.x * sin(a) + q.y * cos(a));
  fragColor = ptg_bilinear(in_input, (r + 0.5) * float(max(1, p_tiling)));
}
""",
))


# More nodes live in their own modules; importing them registers their ops.
from . import ops_layout  # noqa: E402,F401
