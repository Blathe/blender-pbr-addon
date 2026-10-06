"""Bridges Blender node trees to the core evaluator and writes the results
into Blender images."""

import time
import traceback

import bpy

from .core.cpu import CPUBackend
from .core.graph import Evaluator, GraphError, NodeSpec
from . import thumbnails
from .nodes import CHANNELS, TREE_ID, PTGNode, PTGNodeOutput

CHANNEL_LABELS = {c[0]: c[1] for c in CHANNELS}
LINEAR_CHANNELS = {"BASE_COLOR", "EMISSION"}
PREVIEW_IMAGE = "Texture Graph Preview"

# (tree name, "full" or "draft") -> (Evaluator, status note, requested backend).
# Draft evaluations run at a lower resolution while the user is editing, so
# each has its own cache.
_evaluators = {}
_pending = set()
# tree name -> time of the last edit, until a full-resolution pass has run.
_last_edit = {}
# (tree name, channel) -> (content key, was draft) of what each image holds.
_written = {}
# (tree name, node name) -> content key of the node's current thumbnail.
_thumbs = {}
# Seconds without edits before the full-resolution pass runs.
SETTLE = 0.35
# tree name -> status dict shown in the sidebar. Kept out of RNA on purpose:
# writing tree properties from an evaluation can trigger another tree update.
_status = {}
# True while an evaluation runs; updates it causes are ignored.
_busy = False


def status(tree):
    return _status.setdefault(tree.name, {
        "backend": "", "error": "", "nodes": 0, "total_ms": 0.0, "graph_ms": 0.0,
        "readback_ms": 0.0, "write_ms": 0.0, "images": 0, "recent": [], "draft": False, "size": 0,
        "thumbs": 0, "thumb_ms": 0.0,
    })


def _make_evaluator(tree):
    if tree.backend == "GPU":
        try:
            from .core.gpu_backend import GPUBackend
            return Evaluator(GPUBackend()), ""
        except Exception as exc:  # no GPU context, e.g. background mode
            return Evaluator(CPUBackend()), f"GPU unavailable, using CPU: {exc}"
    return Evaluator(CPUBackend()), ""


def get_evaluator(tree, quality="full"):
    entry = _evaluators.get((tree.name, quality))
    if entry is None or entry[2] != tree.backend:
        if entry is not None:
            entry[0].clear()
        entry = (*_make_evaluator(tree), tree.backend)
        _evaluators[(tree.name, quality)] = entry
    return entry[0], entry[1]


def draft_size(tree):
    """Resolution used while editing, or None to always use full resolution."""
    if tree.draft_resolution == "OFF":
        return None
    draft = int(tree.draft_resolution)
    return draft if draft < int(tree.resolution) else None


def clear_all():
    for ev, _, _ in _evaluators.values():
        try:
            ev.clear()
        except Exception:
            pass
    _evaluators.clear()
    _pending.clear()
    _last_edit.clear()
    _written.clear()
    _thumbs.clear()
    _status.clear()
    thumbnails.clear()


# ---------------------------------------------------------------------------
# Scheduling: property edits arrive in bursts (slider drags), so evaluation
# runs from a short timer instead of inside the update callback.

def schedule(tree):
    if _busy or tree is None or getattr(tree, "bl_idname", "") != TREE_ID or not tree.auto_update:
        return
    _pending.add(tree.name)
    _last_edit[tree.name] = time.perf_counter()
    if not bpy.app.timers.is_registered(_run_pending):
        bpy.app.timers.register(_run_pending, first_interval=0.02)


def _run_pending():
    """Quick pass right after an edit: draft resolution when enabled."""
    names = list(_pending)
    _pending.clear()
    for name in names:
        tree = bpy.data.node_groups.get(name)
        if tree is None or tree.bl_idname != TREE_ID:
            _last_edit.pop(name, None)
            continue
        draft = draft_size(tree) is not None
        evaluate_tree(tree, draft=draft)
        if not draft:
            _last_edit.pop(name, None)
    if _last_edit and not bpy.app.timers.is_registered(_settle):
        bpy.app.timers.register(_settle, first_interval=SETTLE)
    return None


def _settle():
    """Full-resolution pass once a tree has had no edits for SETTLE seconds."""
    now = time.perf_counter()
    for name, edited in list(_last_edit.items()):
        if now - edited < SETTLE or name in _pending:
            continue
        del _last_edit[name]
        tree = bpy.data.node_groups.get(name)
        if tree is not None and tree.bl_idname == TREE_ID:
            evaluate_tree(tree)
    return SETTLE / 2 if _last_edit else None


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


def _run(tree, targets, size, quality="full"):
    ev, note = get_evaluator(tree, quality)
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
        _evaluators[(tree.name, quality)] = (cpu, note, tree.backend)
        return cpu, cpu.evaluate(specs, targets, size), note


def _update_thumbnails(tree, ev, results):
    """Refresh thumbnails of nodes whose content changed. Returns the count."""
    count = 0
    for node_name, (buf, kind) in results.items():
        content = ev.content_keys.get(node_name)
        if _thumbs.get((tree.name, node_name)) == content:
            continue
        thumbnails.set_thumbnail(tree, node_name, ev.backend.thumbnail(buf, thumbnails.SIZE), kind)
        _thumbs[(tree.name, node_name)] = content
        count += 1
    return count


def evaluate_tree(tree, force=False, draft=False):
    """Evaluate every Output node and write one image per channel.

    Only outputs whose content changed are copied back from the GPU, which is
    the expensive step. With draft=True the graph runs at the draft
    resolution and only changed outputs drop to it; the full-resolution pass
    afterwards restores them. Returns {channel: image}."""
    global _busy
    st = status(tree)
    start = time.perf_counter()
    full = int(tree.resolution)
    side = (draft_size(tree) or full) if draft else full
    draft = side != full
    size = (side, side)
    images = {}
    _busy = True
    try:
        specs, outputs = extract_graph(tree)
        targets = list(specs) if tree.show_thumbnails else [o.name for o in outputs]
        ev, results, note = _run(tree, targets, size, "draft" if draft else "full")
        graph_done = time.perf_counter()
        thumbs = _update_thumbnails(tree, ev, results) if tree.show_thumbnails else 0
        thumbs_done = time.perf_counter()
        readback = write = 0.0
        written = 0
        for node in outputs:
            buf, _ = results[node.name]
            name = image_name(tree, node.channel)
            img = bpy.data.images.get(name)
            content = ev.content_keys[node.name]
            previous = _written.get((tree.name, node.channel))
            if img is None or previous is None:
                stale = True
            elif draft:
                stale = previous[0] != content
            else:
                stale = previous != (content, False) or tuple(img.size) != size
            if force or stale:
                t0 = time.perf_counter()
                pixels = ev.backend.to_numpy(buf)
                t1 = time.perf_counter()
                colorspace = "Linear Rec.709" if node.channel in LINEAR_CHANNELS else "Non-Color"
                img = write_image(name, pixels, colorspace)
                readback += t1 - t0
                write += time.perf_counter() - t1
                written += 1
                _written[(tree.name, node.channel)] = (content, draft)
            images[node.channel] = img
        st.update(backend=ev.backend.name, nodes=len(ev.ran), error=note, images=written, draft=draft, size=side,
                  graph_ms=(graph_done - start) * 1000.0, readback_ms=readback * 1000.0, write_ms=write * 1000.0,
                  thumbs=thumbs, thumb_ms=(thumbs_done - graph_done) * 1000.0)
        if getattr(ev.backend, "slow_readback", False):
            st["error"] = (note + " " if note else "") + "Using the fallback GPU readback (details in the system console)."
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
    for timer in (_run_pending, _settle):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    clear_all()
