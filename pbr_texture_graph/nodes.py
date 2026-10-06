"""The Texture Graph node tree, its sockets and nodes."""

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty
from bpy.types import Node, NodeSocket, NodeTree

from .core.ops import OPS

TREE_ID = "PTGTextureGraph"


def _changed(self, context):
    from . import evaluate
    evaluate.schedule(self.id_data)


# ---------------------------------------------------------------------------
# Tree

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
    auto_update: BoolProperty(name="Auto Update", default=True, description="Re-evaluate whenever the graph changes")
    material: PointerProperty(name="Material", type=bpy.types.Material)

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
        for p in OPS[self.op_id].params:
            layout.prop(self, p.name)


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
        layout.prop(self, "channel", text="")

    def draw_label(self):
        return f"Output: {dict((c[0], c[1]) for c in CHANNELS)[self.channel]}"


NODE_CLASSES = (
    PTGNodePerlin,
    PTGNodeShape,
    PTGNodeBlend,
    PTGNodeLevels,
    PTGNodeNormal,
    PTGNodeGradientMap,
    PTGNodeOutput,
)

classes = (PTGTextureGraph, PTGSocketGray, PTGSocketColor) + NODE_CLASSES
