"""Compile every op's fragment shader with glslangValidator, declared the way
Blender's GPUShaderCreateInfo declares push constants and samplers. Skipped
when glslangValidator is not installed."""

import shutil
import subprocess

import pytest

from pbr_texture_graph.core.gpu_backend import THUMB_GLSL
from pbr_texture_graph.core.ops import GLSL_COMMON, OPS

GLSL_TYPES = {"float": "float", "int": "int", "color": "vec4"}
VALIDATOR = shutil.which("glslangValidator")


def fragment_source(op):
    lines = ["#version 330 core", "uniform ivec2 ptg_size;"]
    for name in op.inputs:
        lines += [f"uniform sampler2D in_{name};", f"uniform int has_{name};"]
    for p in op.params:
        lines.append(f"uniform {GLSL_TYPES[p.kind]} p_{p.name};")
    lines.append("out vec4 fragColor;")
    return "\n".join(lines) + GLSL_COMMON + op.glsl


@pytest.mark.skipif(VALIDATOR is None, reason="glslangValidator not installed")
@pytest.mark.parametrize("op_id", sorted(OPS))
def test_shader_compiles(op_id, tmp_path):
    path = tmp_path / f"{op_id}.frag"
    path.write_text(fragment_source(OPS[op_id]))
    result = subprocess.run([VALIDATOR, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(VALIDATOR is None, reason="glslangValidator not installed")
def test_thumbnail_shader_compiles(tmp_path):
    path = tmp_path / "thumb.frag"
    path.write_text("#version 330 core\nuniform ivec2 src_size;\nuniform ivec2 dst_size;\n"
                    "uniform sampler2D src;\nout vec4 fragColor;\n" + THUMB_GLSL)
    result = subprocess.run([VALIDATOR, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
