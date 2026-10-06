"""Texture export: channel packing for game engines and a PNG writer.
Independent of bpy (EXR is written through Blender in export_blender.py)."""

import struct
import zlib

import numpy as np

# Preset id -> (label, normal convention, packing)
#   packing "SEPARATE": one file per channel
#   packing "ORM": AO, Roughness, Metallic in R, G, B (glTF, Godot, Unreal)
#   packing "METALLIC_SMOOTHNESS": metallic in RGB, smoothness in A (Unity)
PRESETS = {
    "SEPARATE": ("Separate Maps", "OPENGL", "SEPARATE"),
    "GLTF": ("glTF / Godot", "OPENGL", "ORM"),
    "UNREAL": ("Unreal Engine", "DIRECTX", "ORM"),
    "UNITY": ("Unity", "OPENGL", "METALLIC_SMOOTHNESS"),
}

COLOR_CHANNELS = {"BASE_COLOR", "EMISSION"}
SUFFIXES = {
    "BASE_COLOR": "BaseColor",
    "ROUGHNESS": "Roughness",
    "METALLIC": "Metallic",
    "NORMAL": "Normal",
    "HEIGHT": "Height",
    "AO": "AO",
    "EMISSION": "Emission",
}


def _gray(rgba):
    return rgba[:, :, 0]


def pack(maps, preset, size):
    """Turn {channel: (h, w, 4) linear float array} into the files a preset
    needs. Returns {suffix: (array, is_color)} where array is (h, w),
    (h, w, 3) or (h, w, 4) and is_color marks sRGB-encoded color data.
    Graph normals are OpenGL style; DirectX presets flip green."""
    _, normal_format, packing = PRESETS[preset]
    w, h = size
    files = {}

    def fill(value):
        return np.full((h, w), value, dtype=np.float32)

    for channel in ("BASE_COLOR", "EMISSION"):
        if channel in maps:
            files[SUFFIXES[channel]] = (maps[channel][:, :, :3], True)
    if "NORMAL" in maps:
        n = maps["NORMAL"][:, :, :3].copy()
        if normal_format == "DIRECTX":
            n[:, :, 1] = 1.0 - n[:, :, 1]
        files[SUFFIXES["NORMAL"]] = (n, False)
    if "HEIGHT" in maps:
        files[SUFFIXES["HEIGHT"]] = (_gray(maps["HEIGHT"]), False)

    rough = _gray(maps["ROUGHNESS"]) if "ROUGHNESS" in maps else None
    metal = _gray(maps["METALLIC"]) if "METALLIC" in maps else None
    ao = _gray(maps["AO"]) if "AO" in maps else None

    if packing == "ORM" and (rough is not None or metal is not None or ao is not None):
        orm = np.stack([
            ao if ao is not None else fill(1.0),
            rough if rough is not None else fill(0.5),
            metal if metal is not None else fill(0.0),
        ], axis=2)
        files["ORM"] = (orm, False)
    elif packing == "METALLIC_SMOOTHNESS" and (rough is not None or metal is not None):
        m = metal if metal is not None else fill(0.0)
        smooth = 1.0 - rough if rough is not None else fill(0.5)
        files["MetallicSmoothness"] = (np.stack([m, m, m, smooth], axis=2), False)
        if ao is not None:
            files[SUFFIXES["AO"]] = (ao, False)
    else:
        for channel, data in (("ROUGHNESS", rough), ("METALLIC", metal), ("AO", ao)):
            if data is not None:
                files[SUFFIXES[channel]] = (data, False)
    return files


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def encode_png(array, bits=8, srgb=False):
    """PNG bytes for a float array in 0..1 laid out bottom row first (Blender
    order). Shape (h, w) is grayscale, (h, w, 3) RGB, (h, w, 4) RGBA."""
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 2:
        data = data[:, :, None]
    h, w, channels = data.shape
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    if srgb:
        data = data.copy()
        rgb = min(channels, 3)
        data[:, :, :rgb] = linear_to_srgb(data[:, :, :rgb])
    data = np.clip(data[::-1], 0.0, 1.0)  # PNG rows go top to bottom
    if bits == 16:
        pixels = np.round(data * 65535.0).astype(">u2")
    else:
        pixels = np.round(data * 255.0).astype(np.uint8)
    rows = pixels.reshape(h, -1).view(np.uint8)
    raw = np.zeros((h, rows.shape[1] + 1), dtype=np.uint8)  # filter byte 0 per row
    raw[:, 1:] = rows

    def chunk(kind, payload):
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", w, h, bits, color_type, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    if srgb:
        out += chunk(b"sRGB", b"\x00")
    out += chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b"")
    return out


def filename(template, name, suffix, extension):
    base = template.replace("{name}", name).replace("{map}", suffix)
    safe = "".join(c if c.isalnum() or c in "-_. " else "_" for c in base).strip()
    return f"{safe or suffix}.{extension}"
