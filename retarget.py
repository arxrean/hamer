"""Retarget HaMeR MANO hand poses to Allegro robot qpos via dex-retargeting.

Reads the per-hand .npz files produced by demo.py (keypoints_3d in OpenPose/MediaPipe
order, which dex-retargeting expects directly) and writes per-frame Allegro joint
configurations plus the wrist 6-DOF pose for placing the hand in simulation.

Stage 2 of the human->robot inpainting pipeline.
"""
import argparse
import os
from pathlib import Path

import numpy as np

from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
    OPERATOR2MANO,
)


def estimate_frame_from_hand_points(kp: np.ndarray) -> np.ndarray:
    """Wrist coordinate frame (MANO convention) from keypoints 0/5/9 (wrist, index-MCP,
    middle-MCP). Ported verbatim from dex-retargeting's SingleHandDetector so HaMeR
    keypoints are canonicalized exactly the way the robot configs expect."""
    pts = kp[[0, 5, 9], :]
    x_vector = pts[0] - pts[2]
    p = pts - pts.mean(axis=0, keepdims=True)
    _, _, v = np.linalg.svd(p)
    normal = v[2, :]
    x = x_vector - (x_vector @ normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if (z @ (pts[1] - pts[2])) < 0:
        normal, z = -normal, -z
    return np.stack([x, normal, z], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Retarget HaMeR MANO -> Allegro hand qpos")
    parser.add_argument("--npz_dir", type=str, default="demo_out",
                        help="Dir with HaMeR per-hand .npz files")
    parser.add_argument("--urdf_dir", type=str, required=True,
                        help="dex-retargeting robot assets dir, i.e. "
                             "<dex-retargeting repo>/assets/robots/hands")
    parser.add_argument("--out_dir", type=str, default="demo_out/robot",
                        help="Output dir for robot qpos")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Point dex-retargeting at the robot URDF/mesh assets, then build the Allegro
    # right-hand DexPilot retargeting: adds fingertip-pinch terms so grasps close
    # crisply without over-curling the open reach. ref_value below is built generically
    # from the config's target_link_human_indices, so DexPilot's extra pinch pairs work.
    # low_pass_alpha=1.0 disables temporal smoothing (alpha<1 lags the grasp behind the
    # human; we keep the robot in phase and can smooth later if needed).
    RetargetingConfig.set_default_urdf_dir(args.urdf_dir)
    config_path = get_default_config_path(RobotName.allegro, RetargetingType.dexpilot, HandType.right)
    cfg = RetargetingConfig.load_from_file(config_path)
    cfg.low_pass_alpha = 1.0
    retargeting = cfg.build()

    joint_names = list(retargeting.optimizer.robot.dof_joint_names)
    indices = np.asarray(retargeting.optimizer.target_link_human_indices)  # (2, N)
    origin_idx, task_idx = indices[0], indices[1]
    print(f"Robot dof={len(joint_names)}; human indices origin={origin_idx} task={task_idx}")

    # Per-hand pose files only; sort by name so frames are processed in temporal order
    # (the retargeter's filter + warm-start rely on sequential ordering).
    npz_files = sorted(Path(args.npz_dir).glob("*_*.npz"))
    print(f"Found {len(npz_files)} hand pose files in {args.npz_dir}")

    all_qpos = []
    for f in npz_files:
        d = np.load(f)
        kp = d["keypoints_3d"].astype(np.float64)         # (21, 3) OpenPose order, camera frame
        rel = kp - kp[0]                                   # wrist-centered

        # Canonicalize into the MANO operator frame exactly as dex-retargeting expects:
        # estimate the wrist frame from the keypoints (NOT from MANO global_orient) and
        # apply the fixed OPERATOR2MANO rotation. This is the validated transform.
        wrist_frame = estimate_frame_from_hand_points(rel)            # (3, 3) camera-frame wrist axes
        joint_pos = rel @ wrist_frame @ OPERATOR2MANO[HandType.right]  # (21, 3) canonical hand

        ref_value = joint_pos[task_idx, :] - joint_pos[origin_idx, :]  # (N, 3) wrist->fingertip vectors
        qpos = retargeting.retarget(ref_value)                         # full robot dof, filtered

        np.savez(
            Path(args.out_dir) / f.name,
            qpos=qpos.astype(np.float32),                  # (dof,) robot joint angles
            joint_names=np.asarray(joint_names),           # qpos order
            wrist_frame=wrist_frame.astype(np.float32),    # (3, 3) estimated wrist axes (camera frame)
            global_orient_rotmat=d["global_orient_rotmat"].astype(np.float32),  # (3, 3) MANO wrist, camera frame
            wrist_t=d["cam_t_full"].astype(np.float32),    # (3,) wrist translation, camera frame
            is_right=d["is_right"],
        )
        all_qpos.append(qpos)

    if all_qpos:
        np.save(Path(args.out_dir) / "all_qpos.npy", np.stack(all_qpos).astype(np.float32))
        with open(Path(args.out_dir) / "joint_names.txt", "w") as fh:
            fh.write("\n".join(joint_names) + "\n")
    print(f"Saved {len(all_qpos)} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
