import struct
import zlib

import numpy as np
import pytest

from pbr_texture_graph.core import export

H, W = 4, 6


def gray(value):
    a = np.zeros((H, W, 4), dtype=np.float32)
    a[:, :, :3] = value
    a[:, :, 3] = 1.0
    return a


def decode_png(data):
    """Minimal decoder for what encode_png writes (filter 0, no interlace)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks, idat = 8, {}, b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF
        if kind == b"IDAT":
            idat += body
        else:
            chunks[kind] = body
        pos += 12 + length
    w, h, bits, color_type = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    dtype = np.dtype(">u2") if bits == 16 else np.uint8
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, -1)
    assert (raw[:, 0] == 0).all()
    pixels = raw[:, 1:].copy().view(dtype).reshape(h, w, channels)
    return pixels, bits, chunks


def test_orm_packs_ao_rough_metal_with_defaults():
    files = export.pack({"ROUGHNESS": gray(0.3)}, "GLTF", (W, H))
    orm, is_color = files["ORM"]
    assert not is_color
    np.testing.assert_allclose(orm[0, 0], [1.0, 0.3, 0.0])
    assert "Roughness" not in files


def test_orm_uses_all_three_channels():
    maps = {"AO": gray(0.2), "ROUGHNESS": gray(0.4), "METALLIC": gray(0.9)}
    orm = export.pack(maps, "UNREAL", (W, H))["ORM"][0]
    np.testing.assert_allclose(orm[1, 1], [0.2, 0.4, 0.9])


def test_directx_preset_flips_green_only():
    n = gray(0.0)
    n[:, :, :3] = [0.6, 0.8, 0.9]
    out = export.pack({"NORMAL": n}, "UNREAL", (W, H))["Normal"][0]
    np.testing.assert_allclose(out[0, 0], [0.6, 0.2, 0.9], atol=1e-6)
    same = export.pack({"NORMAL": n}, "GLTF", (W, H))["Normal"][0]
    np.testing.assert_allclose(same[0, 0], [0.6, 0.8, 0.9])
    assert n[0, 0, 1] == pytest.approx(0.8)  # input untouched


def test_unity_metallic_smoothness_and_separate_ao():
    maps = {"METALLIC": gray(0.7), "ROUGHNESS": gray(0.25), "AO": gray(0.5)}
    files = export.pack(maps, "UNITY", (W, H))
    np.testing.assert_allclose(files["MetallicSmoothness"][0][0, 0], [0.7, 0.7, 0.7, 0.75])
    np.testing.assert_allclose(files["AO"][0][0, 0], 0.5)


def test_separate_preset_writes_each_map():
    maps = {k: gray(0.5) for k in ("BASE_COLOR", "ROUGHNESS", "METALLIC", "HEIGHT", "AO", "EMISSION")}
    files = export.pack(maps, "SEPARATE", (W, H))
    assert set(files) == {"BaseColor", "Roughness", "Metallic", "Height", "AO", "Emission"}
    assert files["BaseColor"][1] and files["Emission"][1]
    assert not files["Roughness"][1]
    assert files["BaseColor"][0].shape == (H, W, 3)
    assert files["Height"][0].shape == (H, W)


def test_no_packed_file_without_inputs():
    assert export.pack({"BASE_COLOR": gray(0.5)}, "GLTF", (W, H)).keys() == {"BaseColor"}


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("channels", [None, 3, 4])
def test_png_round_trip(bits, channels):
    rng = np.random.default_rng(1)
    shape = (H, W) if channels is None else (H, W, channels)
    data = rng.random(shape).astype(np.float32)
    pixels, got_bits, chunks = decode_png(export.encode_png(data, bits=bits))
    assert got_bits == bits
    assert b"sRGB" not in chunks
    top = 65535 if bits == 16 else 255
    expected = np.round(data[::-1] * top).reshape(pixels.shape)  # Blender rows go bottom up
    np.testing.assert_array_equal(pixels, expected)


def test_png_srgb_encodes_color_but_not_alpha():
    data = np.full((H, W, 4), 0.5, dtype=np.float32)
    pixels, _, chunks = decode_png(export.encode_png(data, srgb=True))
    assert b"sRGB" in chunks
    assert pixels[0, 0, 0] == round(export.linear_to_srgb(np.float32(0.5)) * 255)  # 188
    assert pixels[0, 0, 3] == 128


def test_png_clamps_out_of_range():
    data = np.array([[-1.0, 2.0]], dtype=np.float32)
    pixels, _, _ = decode_png(export.encode_png(data))
    assert pixels.ravel().tolist() == [0, 255]


def test_filename_template():
    assert export.filename("{name}_{map}", "Stone Tiles", "ORM", "png") == "Stone Tiles_ORM.png"
    assert export.filename("T_{name}/{map}", "a:b", "Normal", "exr") == "T_a_b_Normal.exr"
    assert export.filename("", "x", "AO", "png") == "AO.png"
