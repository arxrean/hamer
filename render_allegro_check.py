"""Quick validation render of a retargeted Allegro pose (Stage 3a).

Loads one frame's qpos from demo_out/robot/, applies it to the Menagerie Allegro
right hand, and renders a few canonical views so we can visually confirm the
retargeting matches the human grasp. Camera/world alignment for the overlay
(Stage 3b) is handled separately.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # headless offscreen rendering

import argparse
from pathlib import Path

import numpy as np
import mujoco
import cv2

# dex-retargeting (pinocchio) joint name -> Menagerie MJCF joint name
DEX2MJ = {
    "joint_0.0": "ffj0", "joint_1.0": "ffj1", "joint_2.0": "ffj2", "joint_3.0": "ffj3",   # index
    "joint_4.0": "mfj0", "joint_5.0": "mfj1", "joint_6.0": "mfj2", "joint_7.0": "mfj3",   # middle
    "joint_8.0": "rfj0", "joint_9.0": "rfj1", "joint_10.0": "rfj2", "joint_11.0": "rfj3", # ring
    "joint_12.0": "thj0", "joint_13.0": "thj1", "joint_14.0": "thj2", "joint_15.0": "thj3", # thumb
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allegro_xml", type=str,
                    default=os.path.expanduser("~/mujoco_menagerie/wonik_allegro/right_hand.xml"))
    ap.add_argument("--frame_npz", type=str, default="demo_out/robot/frame_0001_0.npz")
    ap.add_argument("--out_dir", type=str, default="demo_out/robot_check")
    ap.add_argument("--res", type=int, default=480, help="<=640 wide, <=480 tall for default framebuffer")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(args.allegro_xml)
    data = mujoco.MjData(model)

    d = np.load(args.frame_npz)
    names = [str(x) for x in d["joint_names"]]
    qpos = d["qpos"]
    # Set each joint by name (robust to qpos ordering).
    for dex_name, val in zip(names, qpos):
        jid = model.joint(DEX2MJ[dex_name]).id
        data.qpos[model.jnt_qposadr[jid]] = float(val)
    mujoco.mj_forward(model, data)

    # Center the free camera on the hand.
    center = data.xpos[1:].mean(axis=0)
    renderer = mujoco.Renderer(model, height=args.res, width=args.res)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.distance = 0.22
    cam.elevation = -20
    stem = Path(args.frame_npz).stem
    for az in (0, 90, 180, 270):
        cam.azimuth = az
        renderer.update_scene(data, camera=cam)
        img = renderer.render()  # RGB uint8
        out = os.path.join(args.out_dir, f"{stem}_az{az:03d}.png")
        cv2.imwrite(out, img[:, :, ::-1])
        print("wrote", out)


if __name__ == "__main__":
    main()
