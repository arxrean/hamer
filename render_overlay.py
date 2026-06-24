"""Stage 3b: render the retargeted Allegro hand from the HaMeR camera and overlay it
on the original frame(s).

Places a mocap-mounted MuJoCo camera so the robot (whose base is welded at the origin
with the Menagerie palm orientation) appears in the image where the human wrist was,
using HaMeR's estimated intrinsics.

Locked conventions (validated on frame 1):
  - extra_base_rot = [0,0,0]   (Menagerie palm 'id' orientation)
  - depth_scale    = 1.0       (true Allegro size; fingertips reach the interaction point)

Single frame:  python render_overlay.py --robot_npz demo_out/robot/frame_0001_0.npz
All frames:    python render_overlay.py --batch
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import numpy as np
import mujoco
import cv2

from dex_retargeting.constants import OPERATOR2MANO, HandType

DEX2MJ = {
    "joint_0.0": "ffj0", "joint_1.0": "ffj1", "joint_2.0": "ffj2", "joint_3.0": "ffj3",
    "joint_4.0": "mfj0", "joint_5.0": "mfj1", "joint_6.0": "mfj2", "joint_7.0": "mfj3",
    "joint_8.0": "rfj0", "joint_9.0": "rfj1", "joint_10.0": "rfj2", "joint_11.0": "rfj3",
    "joint_12.0": "thj0", "joint_13.0": "thj1", "joint_14.0": "thj2", "joint_15.0": "thj3",
}

# OpenCV camera (look +Z, +Y down) -> MuJoCo camera (look -Z, +Y up): flip Y and Z axes.
CV2MJ = np.diag([1.0, -1.0, -1.0])


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


def euler_deg_to_mat(rx, ry, rz):
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx); cy, sy = np.cos(ry), np.sin(ry); cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def render_frame(model, data, renderer, cam_id, mocap_id, Rp, robot_npz, hamer_npz, depth_scale,
                 wrist_offset=(0.0, 0.0, 0.0)):
    """Pose fingers + camera for one frame; return (rgb HxWx3, alpha HxW float)."""
    rob = np.load(robot_npz)
    ham = np.load(hamer_npz)
    qpos = rob["qpos"]
    names = [str(x) for x in rob["joint_names"]]
    wrist_frame = rob["wrist_frame"].astype(np.float64)
    wrist_t = rob["wrist_t"].astype(np.float64)

    for nm, v in zip(names, qpos):
        jid = model.joint(DEX2MJ[nm]).id
        data.qpos[model.jnt_qposadr[jid]] = float(v)

    M = wrist_frame @ OPERATOR2MANO[HandType.right]       # canonical(robot base) -> OpenCV cam
    R_c = M @ Rp.T                                        # world -> OpenCV cam
    t_c = wrist_t * depth_scale + np.asarray(wrist_offset)  # size match + translation nudge
    data.mocap_pos[mocap_id] = -R_c.T @ t_c
    data.mocap_quat[mocap_id] = mat2quat((R_c.T) @ CV2MJ)
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera="cam")
    rgb = renderer.render()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="cam")
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    alpha = (seg[:, :, 0] >= 0).astype(np.float32)
    return rgb, alpha, ham


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.expanduser("~/mujoco_menagerie/wonik_allegro/allegro_overlay.xml"))
    ap.add_argument("--robot_npz", default="demo_out/robot/frame_0001_0.npz")
    ap.add_argument("--hamer_npz", default="demo_out/frame_0001_0.npz")
    ap.add_argument("--image", default="example_data/trim/frame_0001.jpg")
    ap.add_argument("--batch", action="store_true", help="render every frame in --robot_dir")
    ap.add_argument("--robot_dir", default="demo_out/robot")
    ap.add_argument("--hamer_dir", default="demo_out")
    ap.add_argument("--image_dir", default="example_data/trim")
    ap.add_argument("--render_dir", default="demo_out/robot_render", help="RGBA robot renders (for Stage 4)")
    ap.add_argument("--overlay_dir", default="demo_out/overlay_all", help="composited overlays (inspection)")
    ap.add_argument("--out_dir", default="demo_out/overlay", help="single-frame output dir")
    ap.add_argument("--scale", type=float, default=1.0, help="render at scale*full_res (keeps 16:9)")
    ap.add_argument("--alpha", type=float, default=0.95, help="robot opacity in overlay")
    ap.add_argument("--depth_scale", type=float, default=1.0)
    ap.add_argument("--extra_base_rot", type=float, nargs=3, default=[0, 0, 0])
    ap.add_argument("--wrist_offset", type=float, nargs=3, default=[0.02, -0.02, 0.0],
                    help="camera-frame translation nudge (m) so fingertips meet the object")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    cam_id = model.camera("cam").id
    mocap_id = model.body("cambody").mocapid[0]

    # Intrinsics + clipping are constant across frames (same focal/img_size). Read from first hamer npz.
    if args.batch:
        first_robot = sorted(Path(args.robot_dir).glob("*_*.npz"))[0]
        ham0 = np.load(str(Path(args.hamer_dir) / first_robot.name))
    else:
        ham0 = np.load(args.hamer_npz)
    focal = float(ham0["focal_length"])
    W_full, H_full = [int(x) for x in ham0["img_size"]]
    fovy = np.rad2deg(2.0 * np.arctan(H_full / (2.0 * focal)))
    model.cam_fovy[cam_id] = fovy
    model.stat.extent = 1.0
    model.vis.map.znear = 0.01
    model.vis.map.zfar = 200.0
    mujoco.mj_forward(model, data)
    Rp = data.body("palm").xmat.reshape(3, 3) @ euler_deg_to_mat(*args.extra_base_rot)

    W = int(round(W_full * args.scale)); H = int(round(H_full * args.scale))
    renderer = mujoco.Renderer(model, height=H, width=W)
    print(f"fovy={fovy:.3f} deg  render={W}x{H}  depth_scale={args.depth_scale}")

    if not args.batch:
        rgb, alpha, ham = render_frame(model, data, renderer, cam_id, mocap_id, Rp,
                                       args.robot_npz, args.hamer_npz, args.depth_scale,
                                       wrist_offset=args.wrist_offset)
        os.makedirs(args.out_dir, exist_ok=True)
        bg = cv2.resize(cv2.imread(args.image), (W, H)).astype(np.float32)
        a = (alpha * args.alpha)[:, :, None]
        overlay = (bg * (1 - a) + rgb[:, :, ::-1].astype(np.float32) * a).astype(np.uint8)
        stem = Path(args.robot_npz).stem
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_overlay.png"), overlay)
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_robot.png"),
                    np.dstack([rgb[:, :, ::-1], (alpha * 255).astype(np.uint8)]))
        print(f"wrote {args.out_dir}/{stem}_overlay.png")
        return

    os.makedirs(args.render_dir, exist_ok=True)
    os.makedirs(args.overlay_dir, exist_ok=True)
    robot_files = sorted(Path(args.robot_dir).glob("*_*.npz"))
    for rf in robot_files:
        frame_stem = rf.name.replace("_0.npz", "")          # frame_0001
        hamer_npz = Path(args.hamer_dir) / rf.name           # demo_out/frame_0001_0.npz
        image = Path(args.image_dir) / f"{frame_stem}.jpg"
        rgb, alpha, _ = render_frame(model, data, renderer, cam_id, mocap_id, Rp,
                                     str(rf), str(hamer_npz), args.depth_scale,
                                     wrist_offset=args.wrist_offset)
        # RGBA robot render (full res) for compositing in Stage 4
        cv2.imwrite(str(Path(args.render_dir) / f"{frame_stem}.png"),
                    np.dstack([rgb[:, :, ::-1], (alpha * 255).astype(np.uint8)]))
        # overlay for inspection
        bg = cv2.resize(cv2.imread(str(image)), (W, H)).astype(np.float32)
        a = (alpha * args.alpha)[:, :, None]
        overlay = (bg * (1 - a) + rgb[:, :, ::-1].astype(np.float32) * a).astype(np.uint8)
        cv2.imwrite(str(Path(args.overlay_dir) / f"{frame_stem}.png"), overlay)
        print(f"  {frame_stem}: alpha_px={int(alpha.sum())}")
    print(f"Done. {len(robot_files)} frames -> {args.render_dir} (RGBA) + {args.overlay_dir} (overlay)")


if __name__ == "__main__":
    main()
