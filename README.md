# PBR Texture Graph

A Substance Designer-style node editor inside Blender that generates tileable PBR texture maps, aimed at stylized, hand-painted game assets.

Status: Phase 2 (MVP) in progress. Targets Blender 4.5 LTS (works on 4.2+).

## Install

1. Download the zip from the latest CI run or build it with `python scripts/build.py`.
2. In Blender, drag the zip into the window, or use Edit > Preferences > Get Extensions > the dropdown at the top right > Install from Disk.

## Try it

1. Open a Shader Editor or any editor and switch its type to **Texture Graph**.
2. Open the sidebar (N), go to the **Texture Graph** tab and click **New Example Graph**.
3. Select a mesh and click **Create Material**. Switch the 3D viewport to Material Preview.
4. Tweak any node; the material updates live. The sidebar shows how long the last update took and whether it ran on the GPU or CPU.
5. Every node shows a thumbnail of its result (toggle with **Thumbnails** in the sidebar). Select a node and click **View Active Node** to see it full size in an Image Editor.

## Nodes

| Node | What it does |
| --- | --- |
| Perlin Noise | Tileable fractal Perlin noise |
| Voronoi | Tileable cellular noise: distance, second distance, edges or flat cells |
| Shape | Tiled circle, square or diamond with soft edges |
| Tile Generator | Bricks or tiles with gaps, bevels and per-tile height variation |
| Gradient | Linear, mirrored or radial gradient |
| Blur | Gaussian blur with a resolution-independent radius |
| Warp | Pushes the input along the slopes of a warp map |
| Transform | Offset, rotate and tile the input |
| Blend | Normal, add, multiply, screen, overlay, subtract, darken, lighten, with an optional mask |
| Levels | Input range, gamma and output range |
| Gradient Map | Grayscale to color through three stops |
| Normal | Normal map from height (OpenGL or DirectX) |
| Height to Light | Paints directional light and cavity shading into a base color from height |
| Edge Highlight | Bright painted rims on raised edges |
| Posterize | Quantizes values into flat bands |
| Output | Sends its input to a material channel (Base Color, Roughness, Metallic, Normal, Height, AO, Emission) |

## How it works

Each node is a GLSL fragment shader rendered offscreen at the graph resolution (`core/gpu_backend.py`), with a numpy implementation of the same op (`core/ops.py`) used as the reference in tests and as the fallback when no GPU is available. The evaluator (`core/graph.py`) caches every node's result and re-runs only what changed. Output nodes are written into Blender float images that the generated Principled BSDF material uses.

## Development

```
pip install "bpy==4.5.*" numpy pytest
python -m pytest tests
```

The tests run Blender headless, where GPU drawing is unavailable, so they cover the CPU path and the Blender wiring. Shaders are compile-checked with `glslangValidator` when it is installed. The GPU path is verified by hand in Blender.
