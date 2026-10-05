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
    assert tree.status_backend == "CPU"  # GPU is unavailable in background mode
    assert "GPU unavailable" in tree.status_error


def test_param_change_reruns_only_downstream(tree):
    evaluate.evaluate_tree(tree)
    assert tree.status_nodes == len([n for n in tree.nodes])
    tree.nodes["Normal"].intensity = 5.0
    evaluate.evaluate_tree(tree)
    assert tree.status_nodes == 2  # Normal and its Output


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
