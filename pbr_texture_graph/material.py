"""Builds a Principled BSDF material wired to a graph's output images."""

import bpy


def _link(nt, out_socket, in_socket):
    nt.links.new(out_socket, in_socket)


def build_material(tree, images):
    """Create or rebuild tree.material from {channel: image}."""
    mat = tree.material
    if mat is None:
        mat = bpy.data.materials.new(tree.name)
        tree.material = mat
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (500, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (150, 0)
    _link(nt, bsdf.outputs["BSDF"], out.inputs["Surface"])

    y = 400
    for channel, image in images.items():
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.label = image.name
        tex.location = (-450, y)
        y -= 300
        color = tex.outputs["Color"]

        if channel == "BASE_COLOR":
            _link(nt, color, bsdf.inputs["Base Color"])
        elif channel == "ROUGHNESS":
            _link(nt, color, bsdf.inputs["Roughness"])
        elif channel == "METALLIC":
            _link(nt, color, bsdf.inputs["Metallic"])
        elif channel == "NORMAL":
            nmap = nt.nodes.new("ShaderNodeNormalMap")
            nmap.location = (-150, tex.location.y)
            _link(nt, color, nmap.inputs["Color"])
            _link(nt, nmap.outputs["Normal"], bsdf.inputs["Normal"])
        elif channel == "HEIGHT":
            disp = nt.nodes.new("ShaderNodeDisplacement")
            disp.location = (150, -450)
            disp.inputs["Scale"].default_value = 0.05
            _link(nt, color, disp.inputs["Height"])
            _link(nt, disp.outputs["Displacement"], out.inputs["Displacement"])
        elif channel == "EMISSION":
            _link(nt, color, bsdf.inputs["Emission Color"])
            bsdf.inputs["Emission Strength"].default_value = 1.0
        # AO has no Principled input; the image node is left for the user to wire.
    return mat


def assign_material(obj, mat):
    if obj is None or not hasattr(obj.data, "materials"):
        return False
    if len(obj.material_slots) == 0:
        obj.data.materials.append(mat)
    else:
        obj.active_material = mat
    return True
