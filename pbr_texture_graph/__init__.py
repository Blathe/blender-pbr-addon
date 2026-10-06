# SPDX-License-Identifier: GPL-3.0-or-later
"""PBR Texture Graph: a node-based PBR texture generator for Blender."""

import bpy

from . import evaluate, nodes, operators, ui

_classes = nodes.classes + operators.classes + ui.classes


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    ui.register()
    evaluate.register()


def unregister():
    evaluate.unregister()
    ui.unregister()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
