"""Builds a Principled BSDF material wired to a graph's output images."""

import bpy

MAPPING = "PTG Tiling"
DISPLACEMENT = "PTG Displacement"
BUMP = "PTG Bump"
# Triplanar textures repeat once per this many object units, so a unit
# sphere shows about the same tile size as one cube face.
TRIPLANAR_SPAN = 2.0


def _link(nt, out_socket, in_socket):
    nt.links.new(out_socket, in_socket)


def build_material(tree, images):
    """Create or rebuild tree.material (UV mapped) from {channel: image}."""
    if tree.material is None:
        tree.material = bpy.data.materials.new(tree.name)
    _fill(tree.material, tree, images, triplanar=False)
    return tree.material


def build_triplanar_material(tree, images):
    """Create or rebuild tree.preview_material, which projects the maps from
    three sides instead of using UVs. Closed shapes like spheres can't be
    UV mapped without seams or pinching; projection has neither."""
    if tree.preview_material is None:
        tree.preview_material = bpy.data.materials.new(f"{tree.name} Triplanar")
    _fill(tree.preview_material, tree, images, triplanar=True)
    return tree.preview_material


def _fill(mat, tree, images, triplanar):
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
    _link(nt, coords.outputs["Object" if triplanar else "UV"], mapping.inputs["Vector"])

    y = 400
    for channel, image in images.items():
        if triplanar and channel == "NORMAL" and "HEIGHT" in images:
            continue  # tangent-space normals don't survive projection; bump from height instead
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        if triplanar:
            tex.projection = "BOX"
            tex.projection_blend = 0.3
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
            if triplanar:
                bump = nt.nodes.new("ShaderNodeBump")
                bump.name = BUMP
                bump.location = (-150, tex.location.y)
                _link(nt, color, bump.inputs["Height"])
                _link(nt, bump.outputs["Normal"], bsdf.inputs["Normal"])
        elif channel == "EMISSION":
            _link(nt, color, bsdf.inputs["Emission Color"])
            bsdf.inputs["Emission Strength"].default_value = 1.0
        # AO has no Principled input; the image node is left for the user to wire.
    # Real displacement in EEVEE and Cycles. Triplanar shading comes from
    # its own bump node, so it uses displacement alone.
    mat.displacement_method = "DISPLACEMENT" if triplanar else "BOTH"
    apply_preview_settings(tree)


def apply_preview_settings(tree):
    """Push the tree's tiling and displacement into its materials."""
    for mat, span in ((tree.material, 1.0), (tree.preview_material, TRIPLANAR_SPAN)):
        if mat is None or mat.node_tree is None:
            continue
        nodes = mat.node_tree.nodes
        mapping = nodes.get(MAPPING)
        if mapping is not None:
            scale = tree.preview_tiling / span
            mapping.inputs["Scale"].default_value = (scale, scale, 1.0 if span == 1.0 else scale)
        disp = nodes.get(DISPLACEMENT)
        if disp is not None:
            disp.inputs["Scale"].default_value = tree.preview_displacement
        bump = nodes.get(BUMP)
        if bump is not None:
            # Match the bump to the displacement so lighting agrees with the shape.
            bump.inputs["Distance"].default_value = max(tree.preview_displacement, 0.01)


def assign_material(obj, mat):
    if obj is None or not hasattr(obj.data, "materials"):
        return False
    if len(obj.material_slots) == 0:
        obj.data.materials.append(mat)
    else:
        obj.active_material = mat
    return True
