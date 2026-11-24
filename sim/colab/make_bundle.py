"""Pack what the Colab notebook needs (~75 MB) into sim/colab/vla_bundle.zip.

    cd sim
    /opt/anaconda3/envs/lerobot/bin/python colab/make_bundle.py

Contents (paths kept relative to the coding/ folder, so the scene's relative mesh paths work):
  sim/*.py                                  renderer, export, eval and their helpers
  sim/eval/cube_poses.json                  start poses (sim evaluation)
  sim/calib/joint_map.json, sim/calib/intrinsics/wrist.json
  sim/mujoco_menagerie/robotstudio_so101/   the scene (no *.before_* backups)
  models/                                   the cube mesh
  sim/demos/cube/trimmed/*.npz              the 148 demos

Upload the zip to Google Drive, in the folder the notebook's DRIVE_DIR names (default
MyDrive/so101_vla/).
"""
import pathlib
import zipfile

SIM = pathlib.Path(__file__).resolve().parent.parent
CODING = SIM.parent
OUT = SIM / "colab" / "vla_bundle.zip"


def files():
    yield from sorted(SIM.glob("*.py"))
    yield SIM / "eval/cube_poses.json"
    yield SIM / "calib/joint_map.json"
    yield SIM / "calib/intrinsics/wrist.json"
    for f in sorted((SIM / "mujoco_menagerie/robotstudio_so101").rglob("*")):
        if f.is_file() and ".before_" not in f.name and "__pycache__" not in f.parts and f.name != ".DS_Store":
            yield f
    for f in sorted((CODING / "models").rglob("*")):
        if f.is_file() and f.name != ".DS_Store":
            yield f
    yield from sorted((SIM / "demos/cube/trimmed").glob("*.npz"))


def main():
    n, size = 0, 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files():
            z.write(f, f.relative_to(CODING).as_posix())
            n += 1
            size += f.stat().st_size
    demos = len(list((SIM / "demos/cube/trimmed").glob("*.npz")))
    print(f"{n} files ({size / 1e6:.0f} MB, {demos} demos) -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
