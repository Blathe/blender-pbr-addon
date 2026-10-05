"""Package the add-on as a Blender extension zip in dist/."""

import pathlib
import tomllib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "pbr_texture_graph"


def main():
    manifest = tomllib.loads((SRC / "blender_manifest.toml").read_text())
    out = ROOT / "dist" / f"{manifest['id']}-{manifest['version']}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(SRC))
    print(out)


if __name__ == "__main__":
    main()
