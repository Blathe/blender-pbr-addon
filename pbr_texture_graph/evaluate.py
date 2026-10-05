"""Bridges Blender node trees to the core evaluator and writes the results
into Blender images."""

import time
import traceback

import bpy

from .core.cpu import CPUBackend
from .core.graph import Evaluator, GraphError, NodeSpec
from .nodes import CHANNELS, TREE_ID, PTGNode, PTGNodeOutput

CHANNEL_LABELS = {c[0]: c[1] for c in CHANNELS}
LINEAR_CHANNELS = {"BASE_COLOR", "EMISSION"}
PREVIEW_IMAGE = "Texture Graph Preview"

# tree name -> (Evaluator, status note, requested backend)
_evaluators = {}
_pending = set()
# tree name -> status dict shown in the sidebar. Kept out of RNA on purpose:
# writing tree properties from an evaluation can trigger another tree update.
_status = {}
# True while an evaluation runs; updates it causes are ignored.
_busy = False


def status(tree):
    return _status.setdefault(tree.name, {
        "backend": "", "error": "", "nodes": 0, "total_ms": 0.0, "graph_ms": 0.0,
        "readback_ms": 0.0, "write_ms": 0.0, "images": 0, "recent": [],
    })


def _make_evaluator(tree):
    if tree.backend == "GPU":
        try:
            from .core.gpu_backend import GPUBackend
            return Evaluator(GPUBackend()), ""
        except Exception as exc:  # no GPU context, e.g. background mode
            return Evaluator(CPUBackend()), f"GPU unavailable, using CPU: {exc}"
    return Evaluator(CPUBackend()), ""


def get_evaluator(tree):
    entry = _evaluators.get(tree.name)
    if entry is None or entry[2] != tree.backend:
        if entry is not None:
            entry[0].clear()
        entry = (*_make_evaluator(tree), tree.backend)
        _evaluators[tree.name] = entry
    return entry[0], entry[1]


def clear_all():
    for ev, _, _ in _evaluators.values():
        try:
            ev.clear()
        except Exception:
            pass
    _evaluators.clear()
    _pending.clear()
    _status.clear()


# ---------------------------------------------------------------------------
# Scheduling: property edits arrive in bursts (slider drags), so evaluation
# runs from a short timer instead of inside the update callback.

def schedule(tree):
    if _busy or tree is None or getattr(tree, "bl_idname", "") != TREE_ID or not tree.auto_update:
        return
    _pending.add(tree.name)
    if not bpy.app.timers.is_registered(_run_pending):
        bpy.app.timers.register(_run_pending, first_interval=0.02)


def _run_pending():
    names = list(_pending)
    _pending.clear()
    for name in names:
        tree = bpy.data.node_groups.get(name)
        if tree is not None and tree.bl_idname == TREE_ID:
            evaluate_tree(tree)
    return None


# ---------------------------------------------------------------------------

def _source_node(link):
    """Follow reroutes upstream; return the source node or None."""
    seen = 0
    while link is not None and not link.is_muted and seen < 256:
        node = link.from_node
        if node.bl_idname != "NodeReroute":
            return node
        sock = node.inputs[0]
        link = sock.links[0] if sock.is_linked else None
        seen += 1
    return None


def extract_graph(tree):
    """Return ({name: NodeSpec}, [output nodes])."""
    from .core.ops import OPS

    specs = {}
    outputs = []
    for node in tree.nodes:
        if not isinstance(node, PTGNode):
            continue
        op = OPS[node.op_id]
        inputs = {}
        for name, sock in zip(op.inputs, node.inputs):
            if sock.is_linked:
                src = _source_node(sock.links[0])
                if isinstance(src, PTGNode):
                    inputs[name] = src.name
        specs[node.name] = NodeSpec(node.name, op.id, node.params(), inputs)
        if isinstance(node, PTGNodeOutput):
            outputs.append(node)
    return specs, outputs


def image_name(tree, channel):
    return f"{tree.name} {CHANNEL_LABELS[channel]}"


def write_image(name, rgba, colorspace):
    h, w = rgba.shape[:2]
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    elif tuple(img.size) != (w, h):
        img.scale(w, h)
    if img.colorspace_settings.name != colorspace:
        img.colorspace_settings.name = colorspace
    img.pixels.foreach_set(rgba.ravel())
    img.update()
    return img


def _redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "IMAGE_EDITOR", "NODE_EDITOR"}:
                area.tag_redraw()


def _run(tree, targets, size):
    ev, note = get_evaluator(tree)
    specs, _ = extract_graph(tree)
    try:
        results = ev.evaluate(specs, targets, size)
        return ev, results, note
    except GraphError:
        raise
    except Exception as exc:
        if ev.backend.name != "GPU":
            raise
        # Keep working on the CPU and surface why the GPU path failed.
        traceback.print_exc()
        ev.clear()
        cpu = Evaluator(CPUBackend())
        note = f"GPU failed, using CPU: {exc}"
        _evaluators[tree.name] = (cpu, note, tree.backend)
        return cpu, cpu.evaluate(specs, targets, size), note


def evaluate_tree(tree, force=False):
    """Evaluate every Output node and write one image per channel. Only
    outputs whose result changed are copied into their image unless force.
    Returns {channel: image}."""
    global _busy
    st = status(tree)
    start = time.perf_counter()
    size = (int(tree.resolution), int(tree.resolution))
    images = {}
    _busy = True
    try:
        _, outputs = extract_graph(tree)
        ev, results, note = _run(tree, [o.name for o in outputs], size)
        graph_done = time.perf_counter()
        readback = write = 0.0
        written = 0
        for node in outputs:
            buf, _ = results[node.name]
            name = image_name(tree, node.channel)
            img = bpy.data.images.get(name)
            if force or node.name in ev.ran or img is None or tuple(img.size) != size:
                t0 = time.perf_counter()
                pixels = ev.backend.to_numpy(buf)
                t1 = time.perf_counter()
                colorspace = "Linear Rec.709" if node.channel in LINEAR_CHANNELS else "Non-Color"
                img = write_image(name, pixels, colorspace)
                readback += t1 - t0
                write += time.perf_counter() - t1
                written += 1
            images[node.channel] = img
        st.update(backend=ev.backend.name, nodes=len(ev.ran), error=note, images=written,
                  graph_ms=(graph_done - start) * 1000.0, readback_ms=readback * 1000.0, write_ms=write * 1000.0)
        if getattr(ev.backend, "slow_readback", False):
            st["error"] = (note + " " if note else "") + "Slow GPU readback path in use."
    except Exception as exc:
        traceback.print_exc()
        st["error"] = str(exc)
    finally:
        _busy = False
    now = time.perf_counter()
    st["total_ms"] = (now - start) * 1000.0
    st["recent"] = [t for t in st["recent"] if now - t < 1.0] + [now]
    _redraw()
    return images


def preview_node(tree, node):
    """Evaluate one node and write it to the preview image."""
    global _busy
    _busy = True
    try:
        return _preview_node(tree, node)
    finally:
        _busy = False


def _preview_node(tree, node):
    size = (int(tree.resolution), int(tree.resolution))
    ev, results, note = _run(tree, [node.name], size)
    buf, kind = results[node.name]
    colorspace = "Linear Rec.709" if kind == "COLOR" and not isinstance(node, PTGNodeOutput) else "Non-Color"
    if isinstance(node, PTGNodeOutput):
        colorspace = "Linear Rec.709" if node.channel in LINEAR_CHANNELS else "Non-Color"
    img = write_image(PREVIEW_IMAGE, ev.backend.to_numpy(buf), colorspace)
    status(tree)["error"] = note
    _redraw()
    return img


@bpy.app.handlers.persistent
def _on_load(*_args):
    clear_all()
    for tree in bpy.data.node_groups:
        if tree.bl_idname == TREE_ID:
            schedule(tree)


def register():
    bpy.app.handlers.load_post.append(_on_load)
    bpy.app.handlers.undo_post.append(_on_load)
    bpy.app.handlers.redo_post.append(_on_load)


def unregister():
    for handlers in (bpy.app.handlers.load_post, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if _on_load in handlers:
            handlers.remove(_on_load)
    if bpy.app.timers.is_registered(_run_pending):
        bpy.app.timers.unregister(_run_pending)
    clear_all()
