"""Per-node preview thumbnails, stored as icons in a previews collection."""

import bpy
import bpy.utils.previews
import numpy as np

SIZE = 128

_previews = None


def _key(tree, node_name):
    return f"{tree.name}\x1f{node_name}"


def set_thumbnail(tree, node_name, rgba, kind):
    """rgba: (SIZE, SIZE, 4) float array. Color results are linear, so they
    are converted to display (sRGB) values; data like height stays as is."""
    if _previews is None:
        return
    pixels = np.clip(rgba, 0.0, 1.0)
    if kind == "COLOR":
        pixels = pixels.copy()
        pixels[:, :, :3] = pixels[:, :, :3] ** (1.0 / 2.2)
    key = _key(tree, node_name)
    preview = _previews.get(key) or _previews.new(key)
    n = pixels.shape[0]
    if tuple(preview.image_size) != (n, n):
        preview.image_size = (n, n)
    preview.image_pixels_float.foreach_set(np.ascontiguousarray(pixels, dtype=np.float32).ravel())


def get(tree, node_name):
    if _previews is None:
        return None
    return _previews.get(_key(tree, node_name))


def icon_id(tree, node_name):
    if _previews is None:
        return 0
    preview = _previews.get(_key(tree, node_name))
    return preview.icon_id if preview is not None else 0


def clear():
    if _previews is not None:
        _previews.clear()


def register():
    global _previews
    _previews = bpy.utils.previews.new()


def unregister():
    global _previews
    if _previews is not None:
        bpy.utils.previews.remove(_previews)
    _previews = None
