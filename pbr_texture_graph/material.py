"""Builds a Principled BSDF material wired to a graph's output images."""

import bpy

MAPPING = "PTG Tiling"
DISPLACEMENT = "PTG Displacement"


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

    coords = nt.nodes.new("ShaderNodeTexCoord")
    coords.location = (-1050, 0)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.name = MAPPING
    mapping.location = (-850, 0)
    _link(nt, coords.outputs["UV"], mapping.inputs["Vector"])

    y = 400
    for channel, image in images.items():
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        _link(nt, mapping.outputs["Vector"], tex.inputs["Vector"])
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
            disp.name = DISPLACEMENT
            disp.location = (150, -450)
            _link(nt, color, disp.inputs["Height"])
            _link(nt, disp.outputs["Displacement"], out.inputs["Displacement"])
        elif channel == "EMISSION":
            _link(nt, color, bsdf.inputs["Emission Color"])
            bsdf.inputs["Emission Strength"].default_value = 1.0
        # AO has no Principled input; the image node is left for the user to wire.
    # Real displacement in EEVEE and Cycles, not just bump.
    mat.displacement_method = "BOTH"
    apply_preview_settings(tree)
    return mat


def apply_preview_settings(tree):
    """Push the tree's tiling and displacement into its material."""
    mat = tree.material
    if mat is None or mat.node_tree is None:
        return
    mapping = mat.node_tree.nodes.get(MAPPING)
    if mapping is not None:
        mapping.inputs["Scale"].default_value = (tree.preview_tiling, tree.preview_tiling, 1.0)
    disp = mat.node_tree.nodes.get(DISPLACEMENT)
    if disp is not None:
        disp.inputs["Scale"].default_value = tree.preview_displacement


def assign_material(obj, mat):
    if obj is None or not hasattr(obj.data, "materials"):
        return False
    if len(obj.material_slots) == 0:
        obj.data.materials.append(mat)
    else:
        obj.active_material = mat
    return True
