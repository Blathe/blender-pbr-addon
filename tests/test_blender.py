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
