"""Writes a graph's Output channels to texture files for game engines."""

import os

import bpy
import numpy as np

from . import evaluate
from .core import export


def _as_rgba(array):
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 3:
        alpha = np.ones(array.shape[:2] + (1,), dtype=np.float32)
        array = np.concatenate([array, alpha], axis=2)
    return np.ascontiguousarray(array, dtype=np.float32)


def write_exr(path, array, is_color):
    h, w = array.shape[:2]
    img = bpy.data.images.new("PTG Export", w, h, alpha=True, float_buffer=True)
    try:
        img.colorspace_settings.name = "Linear Rec.709" if is_color else "Non-Color"
        img.pixels.foreach_set(_as_rgba(array).ravel())
        img.filepath_raw = path
        img.file_format = "OPEN_EXR"
        img.save()
    finally:
        bpy.data.images.remove(img)


def render_channels(tree, side):
    """Evaluate every Output node at side x side. Returns {channel: rgba}.
    Uses its own evaluator, freed afterwards, so the editing caches stay."""
    _, outputs = evaluate.extract_graph(tree)
    chosen = {}
    for node in outputs:
        chosen.setdefault(node.channel, node.name)  # first Output per channel wins
    if not chosen:
        return {}
    key = (tree.name, "export")
    try:
        ev, results, _ = evaluate._run(tree, list(chosen.values()), (side, side), "export")
        return {channel: ev.backend.to_numpy(results[name][0]).copy() for channel, name in chosen.items()}
    finally:
        entry = evaluate._evaluators.pop(key, None)
        if entry is not None:
            entry[0].clear()


def export_tree(tree, directory, preset, fmt, side, template):
    """Write the preset's files and return their paths."""
    maps = render_channels(tree, side)
    if not maps:
        return []
    directory = bpy.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    extension = "exr" if fmt == "EXR" else "png"
    paths = []
    for suffix, (array, is_color) in export.pack(maps, preset, (side, side)).items():
        path = os.path.join(directory, export.filename(template, tree.name, suffix, extension))
        if fmt == "EXR":
            write_exr(path, array, is_color)
        else:
            # Height keeps 16 bits even in 8-bit mode to avoid terracing.
            bits = 16 if fmt == "PNG16" or suffix == export.SUFFIXES["HEIGHT"] else 8
            with open(path, "wb") as f:
                f.write(export.encode_png(array, bits=bits, srgb=is_color))
        paths.append(path)
    return paths
