"""Headless Blender tests (run with the `bpy` module from PyPI).
GPU drawing is unavailable in background mode, so these exercise the CPU
fallback and all Blender-side wiring."""

import bpy
import numpy as np
import pytest

import pbr_texture_graph
from pbr_texture_graph import evaluate, operators
from pbr_texture_graph.material import assign_material, build_material
from pbr_texture_graph.nodes import TREE_ID


@pytest.fixture(scope="module", autouse=True)
def addon():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    pbr_texture_graph.register()
    yield
    pbr_texture_graph.unregister()


@pytest.fixture
def tree():
    t = bpy.data.node_groups.new("Stone Tiles", TREE_ID)
    t.resolution = "256"
    operators.build_example(t)
    yield t
    evaluate.clear_all()
    bpy.data.node_groups.remove(t)


def test_example_graph_evaluates_to_images(tree):
    images = evaluate.evaluate_tree(tree)
    assert set(images) == {"BASE_COLOR", "NORMAL", "ROUGHNESS", "HEIGHT"}
    for img in images.values():
        assert tuple(img.size) == (256, 256)
        px = np.empty(256 * 256 * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        assert px.std() > 0.01
    assert images["BASE_COLOR"].colorspace_settings.name == "Linear Rec.709"
    assert images["NORMAL"].colorspace_settings.name == "Non-Color"
    st = evaluate.status(tree)
    assert st["backend"] == "CPU"  # GPU is unavailable in background mode
    assert "GPU unavailable" in st["error"]


def test_param_change_reruns_only_downstream(tree):
    evaluate.evaluate_tree(tree)
    st = evaluate.status(tree)
    assert st["nodes"] == len(tree.nodes)
    assert st["images"] == 4
    tree.nodes["Normal"].intensity = 5.0
    evaluate.evaluate_tree(tree)
    assert st["nodes"] == 2  # Normal and its Output
    assert st["images"] == 1  # only the Normal image is rewritten
    evaluate.evaluate_tree(tree)
    assert st["nodes"] == 0 and st["images"] == 0


def test_evaluation_does_not_reschedule_itself(tree):
    evaluate._pending.clear()
    evaluate.evaluate_tree(tree)
    assert not evaluate._pending


def test_resolution_change_resizes_images(tree):
    evaluate.evaluate_tree(tree)
    tree.resolution = "512"
    images = evaluate.evaluate_tree(tree)
    assert tuple(images["HEIGHT"].size) == (512, 512)


def test_reroute_is_followed(tree):
    levels = tree.nodes["Levels"]
    out = next(n for n in tree.nodes if n.bl_idname == "PTGNodeOutput" and n.channel == "HEIGHT")
    reroute = tree.nodes.new("NodeReroute")
    tree.links.new(levels.outputs[0], reroute.inputs[0])
    tree.links.new(reroute.outputs[0], out.inputs[0])
    specs, _ = evaluate.extract_graph(tree)
    assert specs[out.name].inputs == {"input": "Levels"}


def test_material_is_wired(tree):
    images = evaluate.evaluate_tree(tree)
    mat = build_material(tree, images)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    for name in ("Base Color", "Roughness", "Normal"):
        assert bsdf.inputs[name].is_linked, name
    out = mat.node_tree.nodes["Material Output"]
    assert out.inputs["Displacement"].is_linked

    mesh = bpy.data.meshes.new("m")
    obj = bpy.data.objects.new("o", mesh)
    assert assign_material(obj, mat)
    assert obj.active_material == mat
    # Rebuilding reuses the same material.
    assert build_material(tree, images) == mat


def test_preview_node(tree):
    img = evaluate.preview_node(tree, tree.nodes["Perlin Noise"])
    assert img.name == evaluate.PREVIEW_IMAGE
    assert tuple(img.size) == (256, 256)


def test_set_uniform_skips_optimized_out_uniforms():
    from pbr_texture_graph.core.gpu_backend import set_uniform

    def missing(name, value):
        raise ValueError(f"GPUShader.uniform_int: uniform {name} not found")

    set_uniform(missing, "ptg_size", (1, 1))  # does not raise

    def broken(name, value):
        raise TypeError("expected a sequence")

    with pytest.raises(TypeError):
        set_uniform(broken, "ptg_size", (1, 1))


def test_draft_updates_only_changed_outputs_then_refines(tree):
    tree.resolution = "512"
    tree.draft_resolution = "128"
    evaluate.evaluate_tree(tree)
    st = evaluate.status(tree)
    assert st["images"] == 4

    tree.nodes["Normal"].intensity = 4.0
    images = evaluate.evaluate_tree(tree, draft=True)
    assert st["draft"] and st["images"] == 1
    assert tuple(images["NORMAL"].size) == (128, 128)
    assert tuple(images["BASE_COLOR"].size) == (512, 512)

    images = evaluate.evaluate_tree(tree)
    assert not st["draft"] and st["images"] == 1
    assert tuple(images["NORMAL"].size) == (512, 512)


def test_draft_is_skipped_when_not_smaller(tree):
    tree.draft_resolution = "512"  # tree.resolution is 256
    evaluate.evaluate_tree(tree, draft=True)
    assert not evaluate.status(tree)["draft"]


def test_scheduler_runs_draft_then_full(tree):
    tree.resolution = "512"
    tree.draft_resolution = "128"
    evaluate.evaluate_tree(tree)
    tree.nodes["Levels"].gamma = 1.4
    evaluate.schedule(tree)
    evaluate._run_pending()
    st = evaluate.status(tree)
    assert st["draft"]
    assert tree.name in evaluate._last_edit
    evaluate._last_edit[tree.name] -= 10  # pretend the user stopped editing
    evaluate._settle()
    assert not st["draft"] and st["size"] == 512
    assert tree.name not in evaluate._last_edit
    for timer in (evaluate._run_pending, evaluate._settle):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)


def test_thumbnails_for_every_node_refresh_only_changed(tree):
    from pbr_texture_graph import thumbnails

    evaluate.evaluate_tree(tree)
    st = evaluate.status(tree)
    assert st["thumbs"] == len(tree.nodes)
    for node in tree.nodes:
        # Icons only get ids with a UI; headless, check the stored pixels.
        preview = thumbnails.get(tree, node.name)
        assert preview is not None and tuple(preview.image_size) == (128, 128), node.name
    tree.nodes["Normal"].intensity = 6.0
    evaluate.evaluate_tree(tree)
    assert st["thumbs"] == 2  # Normal and its Output


def test_unconnected_node_still_gets_a_thumbnail(tree):
    from pbr_texture_graph import thumbnails

    lone = tree.nodes.new("PTGNodePerlin")
    evaluate.evaluate_tree(tree)
    assert thumbnails.get(tree, lone.name) is not None


def test_thumbnails_off_skips_unconnected_nodes(tree):
    tree.show_thumbnails = False
    tree.nodes.new("PTGNodePerlin")
    evaluate.evaluate_tree(tree)
    assert evaluate.status(tree)["thumbs"] == 0


@pytest.mark.parametrize("idname", [
    "PTGNodeVoronoi", "PTGNodeTile", "PTGNodeGradient", "PTGNodeBlur", "PTGNodeWarp", "PTGNodeTransform",
    "PTGNodeFloodFill", "PTGNodeTileSampler", "PTGNodeDistance", "PTGNodeBevel",
])
def test_new_nodes_evaluate_into_an_output(tree, idname):
    node = tree.nodes.new(idname)
    out = next(n for n in tree.nodes if n.bl_idname == "PTGNodeOutput" and n.channel == "HEIGHT")
    if node.inputs:
        tree.links.new(tree.nodes["Levels"].outputs[0], node.inputs[0])
    tree.links.new(node.outputs[0], out.inputs[0])
    images = evaluate.evaluate_tree(tree)
    assert not evaluate.status(tree)["error"].startswith("Unknown")
    px = np.empty(256 * 256 * 4, dtype=np.float32)
    images["HEIGHT"].pixels.foreach_get(px)
    assert px.std() > 0.005, idname


def test_export_presets_write_expected_files(tree, tmp_path):
    from pbr_texture_graph.export_blender import export_tree
    from tests.test_export import decode_png

    paths = export_tree(tree, str(tmp_path / "gltf"), "GLTF", "PNG8", 64, "{name}_{map}")
    names = sorted(p.rsplit("/", 1)[1] for p in paths)
    assert names == ["Stone Tiles_BaseColor.png", "Stone Tiles_Height.png",
                     "Stone Tiles_Normal.png", "Stone Tiles_ORM.png"]
    pixels, bits, chunks = decode_png((tmp_path / "gltf" / "Stone Tiles_BaseColor.png").read_bytes())
    assert pixels.shape == (64, 64, 3) and bits == 8 and b"sRGB" in chunks
    assert pixels.std() > 2
    height, bits, _ = decode_png((tmp_path / "gltf" / "Stone Tiles_Height.png").read_bytes())
    assert bits == 16 and height.shape == (64, 64, 1)
    orm, _, _ = decode_png((tmp_path / "gltf" / "Stone Tiles_ORM.png").read_bytes())
    assert (orm[:, :, 0] == 255).all() and (orm[:, :, 2] == 0).all()  # no AO, no metallic
    assert orm[:, :, 1].std() > 1

    unreal = export_tree(tree, str(tmp_path / "ue"), "UNREAL", "PNG16", 64, "{map}")
    gl, _, _ = decode_png((tmp_path / "gltf" / "Stone Tiles_Normal.png").read_bytes())
    dx, _, _ = decode_png((tmp_path / "ue" / "Normal.png").read_bytes())
    assert len(unreal) == 4
    np.testing.assert_allclose(dx[:, :, 1] / 65535.0, 1.0 - gl[:, :, 1] / 255.0, atol=0.003)
    # Exporting leaves no extra evaluator or image behind.
    assert (tree.name, "export") not in evaluate._evaluators


def test_export_exr(tree, tmp_path):
    from pbr_texture_graph.export_blender import export_tree

    before = len(bpy.data.images)
    paths = export_tree(tree, str(tmp_path), "SEPARATE", "EXR", 32, "{map}")
    assert sorted(p.rsplit("/", 1)[1] for p in paths) == [
        "BaseColor.exr", "Height.exr", "Normal.exr", "Roughness.exr"]
    assert len(bpy.data.images) == before
    img = bpy.data.images.load(str(tmp_path / "Roughness.exr"))
    try:
        assert tuple(img.size) == (32, 32)
        px = np.empty(32 * 32 * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        assert 0.55 < px[0::4].mean() < 0.95  # roughness Levels maps into 0.6..0.9
    finally:
        bpy.data.images.remove(img)



def test_export_folder_accepts_blend_relative_paths(tree, recwarn):
    tree.export_directory = "//textures/"
    assert tree.export_directory == "//textures/"
    assert not [w for w in recwarn if "blend relative" in str(w.message)]


@pytest.mark.parametrize("shape", ["TORUS", "SPHERE", "CUBE", "CYLINDER", "PLANE"])
def test_preview_object_has_material_and_uvs(tree, shape):
    from pbr_texture_graph import preview

    build_material(tree, evaluate.evaluate_tree(tree))
    tree.preview_shape = shape
    obj = preview.preview_object(tree, bpy.context.scene)
    try:
        assert obj.name in bpy.context.scene.objects
        assert obj.active_material == tree.material
        assert obj.modifiers["PTG Detail"].subdivision_type == "SIMPLE"
        uv = np.empty(len(obj.data.loops) * 2, dtype=np.float32)
        obj.data.uv_layers.active.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        assert uv.min() >= -1e-5 and uv[:, 0].max() > 0.99 and uv[:, 1].max() > 0.99
        # No face stretches across a UV seam (cylinder caps are one tile each).
        for poly in obj.data.polygons:
            if shape == "CYLINDER" and abs(poly.normal.z) > 0.9:
                continue
            span = uv[poly.loop_start:poly.loop_start + poly.loop_total].ptp(axis=0)
            assert (span < 0.5).all(), poly.index
    finally:
        bpy.data.objects.remove(obj)


def test_preview_shape_and_settings_update_in_place(tree):
    from pbr_texture_graph import preview

    mat = build_material(tree, evaluate.evaluate_tree(tree))
    assert mat.displacement_method == "BOTH"
    obj = preview.preview_object(tree, bpy.context.scene)
    old_mesh = obj.data.name
    tree.preview_shape = "CUBE"
    assert obj.data.name != old_mesh and old_mesh not in bpy.data.meshes
    assert obj.active_material == mat
    # Calling again reuses the same object.
    assert preview.preview_object(tree, bpy.context.scene) == obj

    tree.preview_tiling = 3.0
    tree.preview_displacement = 0.2
    assert tuple(mat.node_tree.nodes["PTG Tiling"].inputs["Scale"].default_value) == (3.0, 3.0, 1.0)
    assert mat.node_tree.nodes["PTG Displacement"].inputs["Scale"].default_value == pytest.approx(0.2)
    # Rebuilding the material keeps the settings and every image follows the tiling.
    build_material(tree, evaluate.evaluate_tree(tree))
    nodes = mat.node_tree.nodes
    assert tuple(nodes["PTG Tiling"].inputs["Scale"].default_value) == (3.0, 3.0, 1.0)
    for node in nodes:
        if node.bl_idname == "ShaderNodeTexImage":
            assert node.inputs["Vector"].is_linked
    bpy.data.objects.remove(obj)


def test_torus_uvs_wrap_seamlessly(tree):
    """Where the torus closes on itself, the UVs on both sides differ by
    whole tiles, so a tileable texture continues without a seam."""
    from pbr_texture_graph import preview

    mesh = preview.build_mesh("t", "TORUS")
    uv = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    per_vert = {}
    for loop, (u, v) in zip(mesh.loops, uv):
        per_vert.setdefault(loop.vertex_index, set()).add((round(u % 1.0, 4) % 1.0, round(v % 1.0, 4) % 1.0))
    assert all(len(values) == 1 for values in per_vert.values())
    assert len(mesh.vertices) == len(per_vert) and not any(e.use_seam for e in mesh.edges)
    bpy.data.meshes.remove(mesh)



@pytest.mark.parametrize("idname", ["PTGNodeFloodFillGray", "PTGNodeFloodFillColor"])
def test_flood_fill_chain(tree, idname):
    fill = tree.nodes.new("PTGNodeFloodFill")
    to = tree.nodes.new(idname)
    tree.links.new(tree.nodes["Shape"].outputs[0], fill.inputs[0])
    tree.links.new(fill.outputs[0], to.inputs[0])
    out = next(n for n in tree.nodes if n.bl_idname == "PTGNodeOutput" and n.channel == "BASE_COLOR")
    tree.links.new(to.outputs[0], out.inputs[0])
    images = evaluate.evaluate_tree(tree)
    assert evaluate.status(tree)["error"].startswith("GPU unavailable")  # nothing else went wrong
    px = np.empty(256 * 256 * 4, dtype=np.float32)
    images["BASE_COLOR"].pixels.foreach_get(px)
    assert px.reshape(-1, 4)[:, 0].std() > 0.05
