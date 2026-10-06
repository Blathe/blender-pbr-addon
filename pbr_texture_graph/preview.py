"""A preview mesh in the scene that shows a graph's material."""

import math

import bmesh
import bpy

# Viewport subdivision on top of the base mesh so displacement has detail.
SUBDIV_LEVELS = 2


def _plane(bm):
    bmesh.ops.create_grid(bm, x_segments=64, y_segments=64, size=1.0)
    uv = bm.loops.layers.uv.verify()
    for face in bm.faces:
        for loop in face.loops:
            co = loop.vert.co
            loop[uv].uv = ((co.x + 1) / 2, (co.y + 1) / 2)


def _cube(bm):
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=15, use_grid_fill=True)
    bm.normal_update()
    uv = bm.loops.layers.uv.verify()
    for face in bm.faces:
        n = face.normal
        side = 1 if max(n, key=abs) > 0 else -1
        for loop in face.loops:
            x, y, z = loop.vert.co
            # One upright, unmirrored tile per face, as seen from outside.
            if abs(n.x) > 0.9:
                u, v = y * side, z
            elif abs(n.y) > 0.9:
                u, v = -x * side, z
            else:
                u, v = x, y * side
            loop[uv].uv = ((u + 1) / 2, (v + 1) / 2)


def _sphere(bm):
    uv = bm.loops.layers.uv.verify()  # calc_uvs needs the layer up front
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0, calc_uvs=True)
    for face in bm.faces:
        face.smooth = True
        for loop in face.loops:
            u, v = loop[uv].uv
            loop[uv].uv = (u * 2, v)  # circumference is twice the pole-to-pole arc


def _cylinder(bm):
    bmesh.ops.create_cone(bm, cap_ends=True, segments=64, radius1=1.0, radius2=1.0, depth=2.0)
    # Rings down the side so displacement has rows to work with.
    vertical = [e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) > 1.0]
    bmesh.ops.subdivide_edges(bm, edges=vertical, cuts=31)
    bm.normal_update()
    uv = bm.loops.layers.uv.verify()
    around = round(math.pi)  # tiles around, so a tile is roughly square
    for face in bm.faces:
        if abs(face.normal.z) > 0.9:
            for loop in face.loops:
                co = loop.vert.co
                loop[uv].uv = ((co.x + 1) / 2, (co.y + 1) / 2)
            continue
        face.smooth = True
        center = math.atan2(face.calc_center_median().y, face.calc_center_median().x)
        for loop in face.loops:
            co = loop.vert.co
            angle = math.atan2(co.y, co.x)
            # Keep each face on the same side of the seam.
            angle += 2 * math.pi * round((center - angle) / (2 * math.pi))
            loop[uv].uv = ((angle / (2 * math.pi) + 0.5) * around, (co.z + 1) / 2)


SHAPES = {"SPHERE": _sphere, "CUBE": _cube, "CYLINDER": _cylinder, "PLANE": _plane}


def build_mesh(name, shape):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    SHAPES[shape](bm)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def preview_object(tree, scene):
    """Create the preview object for tree, or refresh its mesh. Returns it."""
    obj = tree.preview_object
    if obj is None:
        obj = bpy.data.objects.new(f"{tree.name} Preview", bpy.data.meshes.new("PTG Preview"))
        obj.location = scene.cursor.location
        tree.preview_object = obj
    if obj.name not in scene.objects:
        scene.collection.objects.link(obj)
    rebuild_shape(tree)
    mod = obj.modifiers.get("PTG Detail") or obj.modifiers.new("PTG Detail", "SUBSURF")
    mod.subdivision_type = "SIMPLE"
    mod.levels = mod.render_levels = SUBDIV_LEVELS
    return obj


def rebuild_shape(tree):
    """Swap the preview object's mesh for tree.preview_shape."""
    obj = tree.preview_object
    if obj is None:
        return
    mesh = build_mesh(f"{tree.name} Preview", tree.preview_shape)
    if tree.material is not None:
        mesh.materials.append(tree.material)
    old = obj.data
    obj.data = mesh
    if old is not None and old.users == 0:
        bpy.data.meshes.remove(old)
