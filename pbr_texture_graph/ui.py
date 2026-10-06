from bpy.types import Panel
from nodeitems_utils import NodeCategory, NodeItem, register_node_categories, unregister_node_categories

from . import evaluate
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
        col.prop(tree, "draft_resolution")
        col.prop(tree, "backend")
        col.prop(tree, "show_thumbnails")
        col.prop(tree, "auto_update")

        col = layout.column(align=True)
        col.operator("ptg.update", icon="FILE_REFRESH")
        col.operator("ptg.view_node", icon="IMAGE_DATA")
        col.operator("ptg.create_material", icon="MATERIAL")
        if tree.material:
            layout.label(text=f"Material: {tree.material.name}", icon="MATERIAL")

        box = layout.box()
        box.label(text="Preview", icon="SHADING_TEXTURE")
        col = box.column()
        col.prop(tree, "preview_shape")
        col.prop(tree, "preview_tiling")
        col.prop(tree, "preview_displacement")
        box.operator("ptg.preview_on_model", icon="MESH_UVSPHERE")

        box = layout.box()
        box.label(text="Export", icon="EXPORT")
        col = box.column()
        col.prop(tree, "export_preset")
        col.prop(tree, "export_format")
        col.prop(tree, "export_resolution")
        col.prop(tree, "export_directory")
        col.prop(tree, "export_template")
        if tree.export_preset == "UNREAL":
            sub = col.column(align=True)
            sub.scale_y = 0.8
            sub.label(text="Keep Normal nodes on OpenGL;", icon="INFO")
            sub.label(text="the preset converts to DirectX.", icon="BLANK1")
        box.operator("ptg.export_textures", icon="EXPORT")

        st = evaluate.status(tree)
        box = layout.box()
        if st["backend"]:
            kind = "draft" if st["draft"] else "full"
            box.label(text=f"Last update: {st['total_ms']:.1f} ms on {st['backend']} ({kind} {st['size']} px)")
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"Graph: {st['graph_ms']:.1f} ms, {st['nodes']} nodes re-run")
            col.label(text=f"Readback: {st['readback_ms']:.1f} ms, write: {st['write_ms']:.1f} ms")
            col.label(text=f"Images updated: {st['images']}, updates/sec: {len(st['recent'])}")
            col.label(text=f"Thumbnails: {st['thumbs']} in {st['thumb_ms']:.1f} ms")
        if st["error"]:
            for i, line in enumerate(_wrap(st["error"], 40)):
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
    PTGCategory("PTG_GENERATORS", "Generators", items=[
        NodeItem("PTGNodePerlin"), NodeItem("PTGNodeVoronoi"), NodeItem("PTGNodeShape"),
        NodeItem("PTGNodeTile"), NodeItem("PTGNodeGradient"), NodeItem("PTGNodeTileSampler"),
    ]),
    PTGCategory("PTG_SHAPES", "Shapes", items=[
        NodeItem("PTGNodeFloodFill"), NodeItem("PTGNodeFloodFillGray"), NodeItem("PTGNodeFloodFillColor"),
        NodeItem("PTGNodeDistance"), NodeItem("PTGNodeBevel"),
    ]),
    PTGCategory("PTG_FILTERS", "Filters", items=[
        NodeItem("PTGNodeBlend"), NodeItem("PTGNodeLevels"), NodeItem("PTGNodeGradientMap"),
        NodeItem("PTGNodeBlur"), NodeItem("PTGNodeWarp"), NodeItem("PTGNodeTransform"), NodeItem("PTGNodeNormal"),
    ]),
    PTGCategory("PTG_STYLIZED", "Stylized", items=[
        NodeItem("PTGNodeHeightToLight"), NodeItem("PTGNodeEdgeHighlight"), NodeItem("PTGNodePosterize"),
    ]),
    PTGCategory("PTG_OUTPUT", "Output", items=[NodeItem("PTGNodeOutput")]),
]

classes = (PTG_PT_graph,)


def register():
    register_node_categories("PTG_NODES", CATEGORIES)


def unregister():
    unregister_node_categories("PTG_NODES")
