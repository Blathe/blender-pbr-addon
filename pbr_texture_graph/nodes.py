"""The Texture Graph node tree, its sockets and nodes."""

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty,
                       StringProperty)
from bpy.types import Node, NodeSocket, NodeTree

from .core.ops import OPS

TREE_ID = "PTGTextureGraph"


def _changed(self, context):
    from . import evaluate
    evaluate.schedule(self.id_data)


# ---------------------------------------------------------------------------
# Tree

def _preview_shape_changed(self, context):
    from . import preview
    preview.rebuild_shape(self)


def _preview_settings_changed(self, context):
    from .material import apply_preview_settings
    apply_preview_settings(self)


EXPORT_PRESETS = [
    ("SEPARATE", "Separate Maps", "One file per map, OpenGL normals"),
    ("GLTF", "glTF / Godot", "Occlusion, roughness, metallic packed in one ORM file; OpenGL normals"),
    ("UNREAL", "Unreal Engine", "ORM packed file; DirectX normals"),
    ("UNITY", "Unity", "Metallic with smoothness in alpha; OpenGL normals"),
]
RESOLUTIONS = [(str(s), f"{s} px", f"{s} x {s} pixels") for s in (256, 512, 1024, 2048, 4096)]


class PTGTextureGraph(NodeTree):
    """Node-based PBR texture generator"""
    bl_idname = TREE_ID
    bl_label = "Texture Graph"
    bl_icon = "TEXTURE"

    resolution: EnumProperty(name="Resolution", items=RESOLUTIONS, default="1024", update=_changed)
    backend: EnumProperty(
        name="Compute",
        items=[
            ("GPU", "GPU", "Evaluate nodes as GPU shaders"),
            ("CPU", "CPU", "Evaluate nodes with numpy (slower, works without a GPU)"),
        ],
        default="GPU",
        update=_changed,
    )
    draft_resolution: EnumProperty(
        name="While Editing",
        items=[("OFF", "Full", "Always update at full resolution")]
        + [(str(s), f"{s} px", f"Update at {s} x {s} while editing, then refine") for s in (128, 256, 512)],
        default="256",
        description="Resolution used while you adjust values; full resolution follows when you stop",
    )
    show_thumbnails: BoolProperty(name="Thumbnails", default=True, update=_changed,
                                  description="Show a preview of each node's result")
    auto_update: BoolProperty(name="Auto Update", default=True, description="Re-evaluate whenever the graph changes")
    material: PointerProperty(name="Material", type=bpy.types.Material)

    preview_object: PointerProperty(name="Preview Object", type=bpy.types.Object)
    preview_shape: EnumProperty(
        name="Shape",
        items=[
            ("TORUS", "Torus", "Wraps in both directions, so tiling shows no seams anywhere"),
            ("SPHERE", "Sphere", "No pinching, but tiles meet at angles along some edges"),
            ("CUBE", "Cube", "One tile per face; edges meeting the top and bottom show seams"),
            ("CYLINDER", "Cylinder", "Seamless around the side; caps are separate"),
            ("PLANE", "Plane", "A single flat tile"),
        ],
        default="TORUS",
        update=_preview_shape_changed,
    )
    preview_tiling: FloatProperty(name="Tiling", default=1.0, min=0.01, soft_max=16.0, update=_preview_settings_changed,
                                  description="How many times the texture repeats across each UV tile")
    preview_displacement: FloatProperty(name="Displacement", default=0.05, min=0.0, soft_max=0.5,
                                        update=_preview_settings_changed,
                                        description="Height output displacement strength in the material")

    # Export settings, remembered per graph. No update callbacks: changing
    # them must not re-evaluate the graph.
    export_directory: StringProperty(name="Folder", subtype="DIR_PATH", default="//textures/",
                                     options={"PATH_SUPPORTS_BLEND_RELATIVE"},
                                     description="Where texture files are written (// is the .blend file's folder)")
    export_preset: EnumProperty(name="Preset", items=EXPORT_PRESETS, default="GLTF")
    export_format: EnumProperty(
        name="Format",
        items=[
            ("PNG8", "PNG 8-bit", "Smallest files; height maps still use 16 bits"),
            ("PNG16", "PNG 16-bit", "Higher precision PNG"),
            ("EXR", "OpenEXR", "32-bit float, linear"),
        ],
        default="PNG8",
    )
    export_resolution: EnumProperty(
        name="Size",
        items=[("GRAPH", "Graph Resolution", "Use the graph's resolution")] + RESOLUTIONS,
        default="GRAPH",
    )
    export_template: StringProperty(name="File Name", default="{name}_{map}",
                                    description="{name} is the graph name, {map} the map type")

    def update(self):
        from . import evaluate
        evaluate.schedule(self)


# ---------------------------------------------------------------------------
# Sockets

class PTGSocketGray(NodeSocket):
    """Grayscale image"""
    bl_idname = "PTGSocketGray"
    bl_label = "Grayscale"

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.63, 0.63, 0.63, 1.0)


class PTGSocketColor(NodeSocket):
    """Color image"""
    bl_idname = "PTGSocketColor"
    bl_label = "Color"

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.78, 0.78, 0.16, 1.0)


# ---------------------------------------------------------------------------
# Nodes

class PTGNode:
    op_id = ""
    # Enum properties are passed to kernels as their index in this order.
    enums = {}
    # (socket type, label) per op input, in op input order.
    in_sockets = ()
    out_socket = ("PTGSocketGray", "Result")

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == TREE_ID

    def init(self, context):
        for kind, label in self.in_sockets:
            self.inputs.new(kind, label)
        if self.out_socket:
            self.outputs.new(*self.out_socket)

    def params(self):
        values = {}
        for p in OPS[self.op_id].params:
            value = getattr(self, p.name)
            if p.name in self.enums:
                value = self.enums[p.name].index(value)
            elif p.kind == "color":
                value = tuple(value)
            values[p.name] = value
        return values

    def draw_buttons(self, context, layout):
        self.draw_thumbnail(layout)
        for p in OPS[self.op_id].params:
            layout.prop(self, p.name)

    def draw_thumbnail(self, layout):
        if not self.id_data.show_thumbnails:
            return
        from . import thumbnails
        icon = thumbnails.icon_id(self.id_data, self.name)
        if icon:
            layout.template_icon(icon_value=icon, scale=6.0)


class PTGNodePerlin(PTGNode, Node):
    """Tileable Perlin noise"""
    bl_idname = "PTGNodePerlin"
    bl_label = "Perlin Noise"
    op_id = "PERLIN"
    out_socket = ("PTGSocketGray", "Noise")

    scale: IntProperty(name="Scale", default=4, min=1, max=64, update=_changed)
    octaves: IntProperty(name="Octaves", default=4, min=1, max=8, update=_changed)
    persistence: FloatProperty(name="Roughness", default=0.5, min=0.0, max=1.0, update=_changed)
    seed: IntProperty(name="Seed", default=0, min=0, update=_changed)


class PTGNodeShape(PTGNode, Node):
    """Tiled basic shape"""
    bl_idname = "PTGNodeShape"
    bl_label = "Shape"
    op_id = "SHAPE"
    enums = {"shape": ["CIRCLE", "SQUARE", "DIAMOND"]}
    out_socket = ("PTGSocketGray", "Shape")

    shape: EnumProperty(
        name="Shape",
        items=[("CIRCLE", "Circle", ""), ("SQUARE", "Square", ""), ("DIAMOND", "Diamond", "")],
        update=_changed,
    )
    tiling: IntProperty(name="Tiling", default=1, min=1, max=64, update=_changed)
    size: FloatProperty(name="Size", default=0.8, min=0.0, max=1.5, update=_changed)
    softness: FloatProperty(name="Softness", default=0.3, min=0.0, max=1.0, update=_changed)


class PTGNodeBlend(PTGNode, Node):
    """Blend two images, optionally through a mask"""
    bl_idname = "PTGNodeBlend"
    bl_label = "Blend"
    op_id = "BLEND"
    modes = ["NORMAL", "ADD", "MULTIPLY", "SCREEN", "OVERLAY", "SUBTRACT", "DARKEN", "LIGHTEN"]
    enums = {"mode": modes}
    in_sockets = (("PTGSocketColor", "Foreground"), ("PTGSocketColor", "Background"), ("PTGSocketGray", "Mask"))
    out_socket = ("PTGSocketColor", "Result")

    mode: EnumProperty(name="Mode", items=[(m, m.title(), "") for m in modes], update=_changed)
    opacity: FloatProperty(name="Opacity", default=1.0, min=0.0, max=1.0, update=_changed)


class PTGNodeLevels(PTGNode, Node):
    """Remap the value range and midtones"""
    bl_idname = "PTGNodeLevels"
    bl_label = "Levels"
    op_id = "LEVELS"
    in_sockets = (("PTGSocketColor", "Input"),)
    out_socket = ("PTGSocketColor", "Result")

    in_low: FloatProperty(name="In Low", default=0.0, min=0.0, max=1.0, update=_changed)
    in_high: FloatProperty(name="In High", default=1.0, min=0.0, max=1.0, update=_changed)
    gamma: FloatProperty(name="Gamma", default=1.0, min=0.05, max=10.0, update=_changed)
    out_low: FloatProperty(name="Out Low", default=0.0, min=0.0, max=1.0, update=_changed)
    out_high: FloatProperty(name="Out High", default=1.0, min=0.0, max=1.0, update=_changed)


class PTGNodeNormal(PTGNode, Node):
    """Normal map from a height map"""
    bl_idname = "PTGNodeNormal"
    bl_label = "Normal"
    op_id = "NORMAL"
    enums = {"format": ["OPENGL", "DIRECTX"]}
    in_sockets = (("PTGSocketGray", "Height"),)
    out_socket = ("PTGSocketColor", "Normal")

    intensity: FloatProperty(name="Intensity", default=1.0, min=0.0, max=20.0, update=_changed)
    format: EnumProperty(
        name="Format",
        items=[("OPENGL", "OpenGL", "Y+ (Blender, Godot, Unity)"), ("DIRECTX", "DirectX", "Y- (Unreal)")],
        update=_changed,
    )


class PTGNodeGradientMap(PTGNode, Node):
    """Map grayscale to color through three color stops"""
    bl_idname = "PTGNodeGradientMap"
    bl_label = "Gradient Map"
    op_id = "GRADIENT_MAP"
    in_sockets = (("PTGSocketGray", "Input"),)
    out_socket = ("PTGSocketColor", "Color")

    color_low: FloatVectorProperty(name="Low", subtype="COLOR", size=4, min=0.0, max=1.0,
                                   default=(0.05, 0.03, 0.08, 1.0), update=_changed)
    color_mid: FloatVectorProperty(name="Mid", subtype="COLOR", size=4, min=0.0, max=1.0,
                                   default=(0.35, 0.2, 0.12, 1.0), update=_changed)
    color_high: FloatVectorProperty(name="High", subtype="COLOR", size=4, min=0.0, max=1.0,
                                    default=(0.9, 0.7, 0.4, 1.0), update=_changed)
    mid_position: FloatProperty(name="Mid Position", default=0.5, min=0.0, max=1.0, update=_changed)


class PTGNodeVoronoi(PTGNode, Node):
    """Tileable Voronoi (cellular) noise"""
    bl_idname = "PTGNodeVoronoi"
    bl_label = "Voronoi"
    op_id = "VORONOI"
    modes = ["F1", "F2", "EDGES", "CELLS"]
    enums = {"mode": modes}
    out_socket = ("PTGSocketGray", "Noise")

    scale: IntProperty(name="Scale", default=6, min=1, max=64, update=_changed)
    mode: EnumProperty(
        name="Mode",
        items=[
            ("F1", "Distance", "Distance to the nearest point"),
            ("F2", "Second Distance", "Distance to the second nearest point"),
            ("EDGES", "Edges", "Lines between cells"),
            ("CELLS", "Cells", "A random flat value per cell"),
        ],
        update=_changed,
    )
    randomness: FloatProperty(name="Randomness", default=1.0, min=0.0, max=1.0, update=_changed)
    seed: IntProperty(name="Seed", default=0, min=0, update=_changed)


class PTGNodeTile(PTGNode, Node):
    """Bricks or tiles with gaps, bevels and per-tile height"""
    bl_idname = "PTGNodeTile"
    bl_label = "Tile Generator"
    op_id = "TILE"
    out_socket = ("PTGSocketGray", "Height")

    tiles_x: IntProperty(name="Tiles X", default=4, min=1, max=64, update=_changed)
    tiles_y: IntProperty(name="Tiles Y", default=8, min=1, max=64, update=_changed)
    offset: FloatProperty(name="Row Offset", default=0.5, min=0.0, max=1.0, update=_changed,
                          description="Shift each row; stays seamless when offset x Tiles Y is a whole number")
    gap: FloatProperty(name="Gap", default=0.08, min=0.0, max=0.9, update=_changed)
    bevel: FloatProperty(name="Bevel", default=0.15, min=0.0, max=0.5, update=_changed)
    variation: FloatProperty(name="Height Variation", default=0.3, min=0.0, max=1.0, update=_changed)
    seed: IntProperty(name="Seed", default=0, min=0, update=_changed)


class PTGNodeGradient(PTGNode, Node):
    """Linear, mirrored or radial gradient"""
    bl_idname = "PTGNodeGradient"
    bl_label = "Gradient"
    op_id = "GRADIENT"
    modes = ["LINEAR", "MIRRORED", "RADIAL"]
    enums = {"mode": modes}
    out_socket = ("PTGSocketGray", "Gradient")

    mode: EnumProperty(
        name="Mode",
        items=[
            ("LINEAR", "Linear", "Ramp that restarts at each repeat"),
            ("MIRRORED", "Mirrored", "Ramp up and down; seamless along its axis"),
            ("RADIAL", "Radial", "Bright centre fading outward"),
        ],
        update=_changed,
    )
    angle: FloatProperty(name="Angle", default=0.0, min=0.0, max=360.0, update=_changed)
    repeat: IntProperty(name="Repeat", default=1, min=1, max=64, update=_changed)


class PTGNodeBlur(PTGNode, Node):
    """Gaussian blur"""
    bl_idname = "PTGNodeBlur"
    bl_label = "Blur"
    op_id = "BLUR"
    in_sockets = (("PTGSocketColor", "Input"),)
    out_socket = ("PTGSocketColor", "Result")

    radius: FloatProperty(name="Radius", default=0.01, min=0.0, max=0.25, precision=3, update=_changed,
                          description="Blur radius as a fraction of the texture")


class PTGNodeWarp(PTGNode, Node):
    """Push the input along the slopes of a warp map"""
    bl_idname = "PTGNodeWarp"
    bl_label = "Warp"
    op_id = "WARP"
    in_sockets = (("PTGSocketColor", "Input"), ("PTGSocketGray", "Warp"))
    out_socket = ("PTGSocketColor", "Result")

    intensity: FloatProperty(name="Intensity", default=0.2, min=0.0, max=5.0, update=_changed)


class PTGNodeTransform(PTGNode, Node):
    """Offset, rotate and tile the input"""
    bl_idname = "PTGNodeTransform"
    bl_label = "Transform"
    op_id = "TRANSFORM"
    in_sockets = (("PTGSocketColor", "Input"),)
    out_socket = ("PTGSocketColor", "Result")

    offset_x: FloatProperty(name="Offset X", default=0.0, min=-1.0, max=1.0, update=_changed)
    offset_y: FloatProperty(name="Offset Y", default=0.0, min=-1.0, max=1.0, update=_changed)
    rotation: FloatProperty(name="Rotation", default=0.0, min=-360.0, max=360.0, update=_changed,
                            description="Seamless in steps of 90 degrees")
    tiling: IntProperty(name="Tiling", default=1, min=1, max=32, update=_changed)


class PTGNodePosterize(PTGNode, Node):
    """Quantize values into flat bands for a painted, cel-like look"""
    bl_idname = "PTGNodePosterize"
    bl_label = "Posterize"
    op_id = "POSTERIZE"
    in_sockets = (("PTGSocketColor", "Input"),)
    out_socket = ("PTGSocketColor", "Result")

    steps: IntProperty(name="Steps", default=4, min=2, max=32, update=_changed)


class PTGNodeHeightToLight(PTGNode, Node):
    """Paint directional light and cavity shading into a base color"""
    bl_idname = "PTGNodeHeightToLight"
    bl_label = "Height to Light"
    op_id = "HEIGHT_TO_LIGHT"
    in_sockets = (("PTGSocketGray", "Height"), ("PTGSocketColor", "Base Color"))
    out_socket = ("PTGSocketColor", "Color")

    angle: FloatProperty(name="Light Angle", default=135.0, min=0.0, max=360.0, update=_changed,
                         description="Direction the light comes from, in degrees (90 = top of the texture)")
    elevation: FloatProperty(name="Elevation", default=45.0, min=1.0, max=90.0, update=_changed)
    depth: FloatProperty(name="Depth", default=1.0, min=0.0, max=20.0, update=_changed,
                         description="How strongly the height shapes the light")
    light: FloatProperty(name="Light", default=0.6, min=0.0, max=1.0, update=_changed)
    cavity: FloatProperty(name="Cavity", default=0.4, min=0.0, max=1.0, update=_changed,
                          description="Darken low areas")


class PTGNodeEdgeHighlight(PTGNode, Node):
    """Bright painted rims on raised edges"""
    bl_idname = "PTGNodeEdgeHighlight"
    bl_label = "Edge Highlight"
    op_id = "EDGE_HIGHLIGHT"
    in_sockets = (("PTGSocketGray", "Height"), ("PTGSocketColor", "Base Color"))
    out_socket = ("PTGSocketColor", "Color")

    color: FloatVectorProperty(name="Color", subtype="COLOR", size=4, min=0.0, max=1.0,
                               default=(1.0, 0.9, 0.7, 1.0), update=_changed)
    width: FloatProperty(name="Width", default=0.01, min=0.001, max=0.1, precision=3, update=_changed,
                         description="Rim width as a fraction of the texture")
    threshold: FloatProperty(name="Threshold", default=0.01, min=0.0, max=0.5, precision=3, update=_changed)
    softness: FloatProperty(name="Softness", default=0.05, min=0.001, max=0.5, precision=3, update=_changed)
    strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=1.0, update=_changed)


CHANNELS = [
    ("BASE_COLOR", "Base Color", ""),
    ("ROUGHNESS", "Roughness", ""),
    ("METALLIC", "Metallic", ""),
    ("NORMAL", "Normal", ""),
    ("HEIGHT", "Height", ""),
    ("AO", "Ambient Occlusion", ""),
    ("EMISSION", "Emission", ""),
]


class PTGNodeOutput(PTGNode, Node):
    """A material channel produced by this graph"""
    bl_idname = "PTGNodeOutput"
    bl_label = "Output"
    op_id = "OUTPUT"
    in_sockets = (("PTGSocketColor", "Input"),)
    out_socket = None

    channel: EnumProperty(name="Channel", items=CHANNELS, update=_changed)

    def draw_buttons(self, context, layout):
        self.draw_thumbnail(layout)
        layout.prop(self, "channel", text="")

    def draw_label(self):
        return f"Output: {dict((c[0], c[1]) for c in CHANNELS)[self.channel]}"


NODE_CLASSES = (
    PTGNodePerlin,
    PTGNodeVoronoi,
    PTGNodeShape,
    PTGNodeTile,
    PTGNodeGradient,
    PTGNodeBlur,
    PTGNodeWarp,
    PTGNodeTransform,
    PTGNodeBlend,
    PTGNodeLevels,
    PTGNodeNormal,
    PTGNodeGradientMap,
    PTGNodePosterize,
    PTGNodeHeightToLight,
    PTGNodeEdgeHighlight,
    PTGNodeOutput,
)

classes = (PTGTextureGraph, PTGSocketGray, PTGSocketColor) + NODE_CLASSES
