from bpy.types import Panel
from nodeitems_utils import NodeCategory, NodeItem, register_node_categories, unregister_node_categories

from .nodes import TREE_ID


class PTG_PT_graph(Panel):
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Texture Graph"
    bl_label = "Texture Graph"

    @classmethod
    def poll(cls, context):
        return getattr(context.space_data, "tree_type", "") == TREE_ID

    def draw(self, context):
        layout = self.layout
        tree = context.space_data.edit_tree
        if tree is None:
            layout.operator("ptg.new_example", icon="ADD")
            return

        col = layout.column()
        col.prop(tree, "resolution")
        col.prop(tree, "backend")
        col.prop(tree, "auto_update")

        col = layout.column(align=True)
        col.operator("ptg.update", icon="FILE_REFRESH")
        col.operator("ptg.view_node", icon="IMAGE_DATA")
        col.operator("ptg.create_material", icon="MATERIAL")
        if tree.material:
            layout.label(text=f"Material: {tree.material.name}", icon="MATERIAL")

        box = layout.box()
        if tree.status_backend:
            box.label(text=f"Last update: {tree.status_ms:.1f} ms on {tree.status_backend}")
            box.label(text=f"Nodes re-run: {tree.status_nodes}")
        if tree.status_error:
            for i, line in enumerate(_wrap(tree.status_error, 40)):
                box.label(text=line, icon="ERROR" if i == 0 else "BLANK1")
        layout.operator("ptg.new_example", icon="ADD")


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for word in words:
        if line and len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out[:6]


class PTGCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == TREE_ID


CATEGORIES = [
    PTGCategory("PTG_GENERATORS", "Generators", items=[NodeItem("PTGNodePerlin"), NodeItem("PTGNodeShape")]),
    PTGCategory("PTG_FILTERS", "Filters", items=[
        NodeItem("PTGNodeBlend"), NodeItem("PTGNodeLevels"), NodeItem("PTGNodeGradientMap"), NodeItem("PTGNodeNormal"),
    ]),
    PTGCategory("PTG_OUTPUT", "Output", items=[NodeItem("PTGNodeOutput")]),
]

classes = (PTG_PT_graph,)


def register():
    register_node_categories("PTG_NODES", CATEGORIES)


def unregister():
    unregister_node_categories("PTG_NODES")
