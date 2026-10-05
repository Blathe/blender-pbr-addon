import bpy
from bpy.types import Operator

from . import evaluate
from .material import assign_material, build_material
from .nodes import TREE_ID, PTGNode


def _edit_tree(context):
    space = context.space_data
    tree = getattr(space, "edit_tree", None)
    if tree is not None and tree.bl_idname == TREE_ID:
        return tree
    return None


class PTG_OT_update(Operator):
    """Evaluate the graph now"""
    bl_idname = "ptg.update"
    bl_label = "Update Now"

    @classmethod
    def poll(cls, context):
        return _edit_tree(context) is not None

    def execute(self, context):
        evaluate.evaluate_tree(_edit_tree(context))
        return {"FINISHED"}


class PTG_OT_create_material(Operator):
    """Build a Principled BSDF material from this graph's outputs and assign it to the active object"""
    bl_idname = "ptg.create_material"
    bl_label = "Create Material"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _edit_tree(context) is not None

    def execute(self, context):
        tree = _edit_tree(context)
        images = evaluate.evaluate_tree(tree)
        if not images:
            self.report({"WARNING"}, "Add an Output node first")
            return {"CANCELLED"}
        mat = build_material(tree, images)
        if not assign_material(context.active_object, mat):
            self.report({"INFO"}, f"Created {mat.name}; select a mesh to assign it")
        return {"FINISHED"}


class PTG_OT_view_node(Operator):
    """Show the active node's result in the Image Editor"""
    bl_idname = "ptg.view_node"
    bl_label = "View Active Node"

    @classmethod
    def poll(cls, context):
        tree = _edit_tree(context)
        return tree is not None and isinstance(tree.nodes.active, PTGNode)

    def execute(self, context):
        tree = _edit_tree(context)
        img = evaluate.preview_node(tree, tree.nodes.active)
        shown = False
        for area in context.screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.spaces.active.image = img
                shown = True
        if not shown:
            self.report({"INFO"}, f"Open an Image Editor to see \"{img.name}\"")
        return {"FINISHED"}


def build_example(tree):
    """A stylized stone-tile graph that exercises every Phase 1 node."""
    nodes, links = tree.nodes, tree.links

    def add(idname, x, y, **props):
        node = nodes.new(idname)
        node.location = (x, y)
        for key, value in props.items():
            setattr(node, key, value)
        return node

    tiles = add("PTGNodeShape", -900, 300, shape="SQUARE", tiling=4, size=0.92, softness=0.35)
    noise = add("PTGNodePerlin", -900, -50, scale=8, octaves=5, persistence=0.55)
    height = add("PTGNodeBlend", -600, 150, mode="MULTIPLY", opacity=0.6)
    levels = add("PTGNodeLevels", -350, 150, in_low=0.05, in_high=0.95)
    color = add("PTGNodeGradientMap", -50, 450,
                color_low=(0.04, 0.04, 0.09, 1.0), color_mid=(0.22, 0.24, 0.33, 1.0),
                color_high=(0.75, 0.66, 0.5, 1.0), mid_position=0.55)
    normal = add("PTGNodeNormal", -50, 150, intensity=2.0)
    rough = add("PTGNodeLevels", -50, -150, out_low=0.6, out_high=0.9)
    links.new(noise.outputs[0], height.inputs["Foreground"])
    links.new(tiles.outputs[0], height.inputs["Background"])
    links.new(height.outputs[0], levels.inputs[0])
    links.new(levels.outputs[0], color.inputs[0])
    links.new(levels.outputs[0], normal.inputs[0])
    links.new(levels.outputs[0], rough.inputs[0])

    for channel, src, y in (("BASE_COLOR", color, 450), ("NORMAL", normal, 150),
                            ("ROUGHNESS", rough, -150), ("HEIGHT", levels, -400)):
        out = add("PTGNodeOutput", 250, y, channel=channel)
        links.new(src.outputs[0], out.inputs[0])


class PTG_OT_new_example(Operator):
    """Create a Texture Graph with a stylized stone-tile example"""
    bl_idname = "ptg.new_example"
    bl_label = "New Example Graph"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        tree = bpy.data.node_groups.new("Stone Tiles", TREE_ID)
        build_example(tree)
        space = context.space_data
        if space is not None and space.type == "NODE_EDITOR":
            space.node_tree = tree
        evaluate.schedule(tree)
        return {"FINISHED"}


classes = (PTG_OT_update, PTG_OT_create_material, PTG_OT_view_node, PTG_OT_new_example)
