"""Stage 3b (v2): per-frame pose-solve alignment of the Allegro hand to the human MANO hand.

For each frame, solve the robot base pose (rotation + translation + uniform scale) that
makes the robot's projected wrist + thumb + index (+ middle/ring, low weight) land on the
human MANO keypoints. This replaces the manual extra_base_rot / depth_scale / wrist_offset
tuning and fixes the wrist-convention bug (human world point = keypoints_3d + cam_t_full).

Outputs:
  demo_out/robot_render_aligned/frame_XXXX.png  RGBA robot render (Stage 4 input)
  demo_out/overlay_aligned/frame_XXXX.png       overlay with thumb/index markers (check)
  demo_out/overlay_aligned/_montage.png         sequence montage
  demo_out/align_params.npy                      (N,7) solved [rx,ry,rz,dx,dy,dz,scale]
  demo_out/align_err.png                         per-frame thumb/index error plot
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import numpy as np
import mujoco
import cv2
from scipy.optimize import least_squares

import render_overlay as ro  # DEX2MJ, mat2quat, euler_deg_to_mat, CV2MJ
from dex_retargeting.constants import OPERATOR2MANO, HandType

# robot keypoint bodies + tip geom z-offset; matched to human OpenPose indices
ROBOT_TIPS = [("palm", 0.0, 0), ("th_tip", 0.0423, 4), ("ff_tip", 0.0267, 8),
              ("mf_tip", 0.0267, 12), ("rf_tip", 0.0267, 16)]
WEIGHTS = np.array([1.0, 4.0, 4.0, 0.8, 0.8])  # wrist, thumb, index, middle, ring


def solve_and_render(model, data, renderer, cam_id, mocap_id, palm, robot_npz, hamer_npz,
                     image_path, focal, W, H, scale, x0):
    ham = np.load(hamer_npz)
    rob = np.load(robot_npz)
    kp = ham["keypoints_3d"].astype(float)
    ct = ham["cam_t_full"].astype(float)
    wframe = rob["wrist_frame"].astype(float)

    # pose fingers
    for nm, v in zip([str(x) for x in rob["joint_names"]], rob["qpos"]):
        jid = model.joint(ro.DEX2MJ[nm]).id
        data.qpos[model.jnt_qposadr[jid]] = float(v)
    mujoco.mj_forward(model, data)

    def tipworld(body, oz):
        b = data.body(body)
        return b.xpos + b.xmat.reshape(3, 3) @ np.array([0, 0, oz])
    rob_pts = [tipworld(b, oz) for (b, oz, _) in ROBOT_TIPS]
    M = wframe @ OPERATOR2MANO[HandType.right]

    def projh(i):
        P = kp[i] + ct
        return np.array([focal * P[0] / P[2] + W / 2, focal * P[1] / P[2] + H / 2])
    htargets = [projh(hi) for (_, _, hi) in ROBOT_TIPS]

    def cam(p):
        rx, ry, rz, dx, dy, dz, s = p
        R_c = M @ (palm @ ro.euler_deg_to_mat(rx, ry, rz)).T
        t_c = (ct + kp[0]) * s + np.array([dx, dy, dz])
        return R_c, t_c

    def proj(R_c, t_c, pw):
        pc = R_c @ pw + t_c
        return np.array([focal * pc[0] / pc[2] + W / 2, focal * pc[1] / pc[2] + H / 2])

    def resid(p):
        R_c, t_c = cam(p)
        r = []
        for pw, ht, w in zip(rob_pts, htargets, WEIGHTS):
            r += list(w * (proj(R_c, t_c, pw) - ht) / 50.0)
        return r

    sol = least_squares(resid, x0, bounds=([-90, -90, -90, -3, -3, -8, 0.4],
                                           [90, 90, 90, 3, 3, 8, 3.5]))
    p = sol.x
    R_c, t_c = cam(p)
    err_th = np.linalg.norm(proj(R_c, t_c, rob_pts[1]) - htargets[1])
    err_idx = np.linalg.norm(proj(R_c, t_c, rob_pts[2]) - htargets[2])

    data.mocap_pos[mocap_id] = -R_c.T @ t_c
    data.mocap_quat[mocap_id] = ro.mat2quat((R_c.T) @ ro.CV2MJ)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="cam"); rgb = renderer.render()
    renderer.enable_segmentation_rendering(); renderer.update_scene(data, camera="cam")
    seg = renderer.render(); renderer.disable_segmentation_rendering()
    alpha = (seg[:, :, 0] >= 0).astype(np.float32)
    return p, (err_th, err_idx), rgb, alpha, htargets, (R_c, t_c, rob_pts, proj)


def precompute(model, data, palm, robot_npz, hamer_npz, focal, W, H):
    """Return (rob_pts[5x3 world], M[3x3], htargets[5x2 px]) for one frame."""
    ham = np.load(hamer_npz); rob = np.load(robot_npz)
    kp = ham["keypoints_3d"].astype(float); ct = ham["cam_t_full"].astype(float)
    wframe = rob["wrist_frame"].astype(float)
    for nm, v in zip([str(x) for x in rob["joint_names"]], rob["qpos"]):
        jid = model.joint(ro.DEX2MJ[nm]).id
        data.qpos[model.jnt_qposadr[jid]] = float(v)
    mujoco.mj_forward(model, data)
    def tw(b, oz):
        bb = data.body(b); return bb.xpos + bb.xmat.reshape(3, 3) @ np.array([0, 0, oz])
    rob_pts = np.array([tw(b, oz) for (b, oz, _) in ROBOT_TIPS])
    M = wframe @ OPERATOR2MANO[HandType.right]
    def projh(i):
        P = kp[i] + ct; return np.array([focal * P[0] / P[2] + W / 2, focal * P[1] / P[2] + H / 2])
    htargets = np.array([projh(hi) for (_, _, hi) in ROBOT_TIPS])
    return rob_pts, M, htargets, ct + kp[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.expanduser("~/mujoco_menagerie/wonik_allegro/allegro_overlay.xml"))
    ap.add_argument("--robot_dir", default="demo_out/robot")
    ap.add_argument("--hamer_dir", default="demo_out")
    ap.add_argument("--image_dir", default="example_data/trim")
    ap.add_argument("--render_dir", default="demo_out/robot_render_aligned")
    ap.add_argument("--overlay_dir", default="demo_out/overlay_aligned")
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.render_dir, exist_ok=True)
    os.makedirs(args.overlay_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    cam_id = model.camera("cam").id
    mocap_id = model.body("cambody").mocapid[0]
    robot_files = sorted(Path(args.robot_dir).glob("*_*.npz"))
    ham0 = np.load(str(Path(args.hamer_dir) / robot_files[0].name))
    focal = float(ham0["focal_length"]); W, H = [int(x) for x in ham0["img_size"]]
    model.cam_fovy[cam_id] = np.rad2deg(2 * np.arctan(H / (2 * focal)))
    model.stat.extent = 1.0; model.vis.map.znear = 0.01; model.vis.map.zfar = 300.0
    mujoco.mj_forward(model, data)
    palm = data.body("palm").xmat.reshape(3, 3)
    sc = args.scale; Wr, Hr = int(W * sc), int(H * sc)
    renderer = mujoco.Renderer(model, height=Hr, width=Wr)

    # Precompute per-frame geometry
    F = [precompute(model, data, palm, str(rf), str(Path(args.hamer_dir) / rf.name), focal, W, H)
         for rf in robot_files]
    N = len(F)

    def cam(rot, s, off, M, wrist):
        R_c = M @ (palm @ ro.euler_deg_to_mat(*rot)).T
        return R_c, wrist * s + off
    def proj(R_c, t_c, pw):
        pc = R_c @ pw + t_c; return np.array([focal * pc[0] / pc[2] + W / 2, focal * pc[1] / pc[2] + H / 2])

    # Joint solve: SHARED rotation(3)+scale(1) + PER-FRAME in-plane translation(2N).
    # Depth comes from the shared scale (2D projection can't constrain per-frame depth).
    def unpack(x):
        off2 = x[4:].reshape(N, 2)
        offs = np.concatenate([off2, np.zeros((N, 1))], axis=1)
        return x[:3], x[3], offs
    def resid(x):
        rot, s, offs = unpack(x); r = []
        for i, (rob_pts, M, ht, wrist) in enumerate(F):
            R_c, t_c = cam(rot, s, offs[i], M, wrist)
            for pw, h, w in zip(rob_pts, ht, WEIGHTS):
                r += list(w * (proj(R_c, t_c, pw) - h) / 50.0)
        return r
    x0 = np.concatenate([[-33, 14, -36, 1.0], np.zeros(2 * N)])
    sol = least_squares(resid, x0, method="trf")
    rot, s, offs = unpack(sol.x)
    print(f"shared rot={np.round(rot,1)} scale={s:.2f}")

    params = np.zeros((N, 7)); errs = []; tiles = []
    for i, rf in enumerate(robot_files):
        stem = rf.name.replace("_0.npz", "")
        rob_pts, M, ht, wrist = F[i]
        R_c, t_c = cam(rot, s, offs[i], M, wrist)
        params[i] = [rot[0], rot[1], rot[2], offs[i][0], offs[i][1], offs[i][2], s]
        eth = np.linalg.norm(proj(R_c, t_c, rob_pts[1]) - ht[1])
        ei = np.linalg.norm(proj(R_c, t_c, rob_pts[2]) - ht[2]); errs.append((eth, ei))
        # re-pose fingers for this frame and render
        rob = np.load(str(rf))
        for nm, v in zip([str(x) for x in rob["joint_names"]], rob["qpos"]):
            jid = model.joint(ro.DEX2MJ[nm]).id; data.qpos[model.jnt_qposadr[jid]] = float(v)
        data.mocap_pos[mocap_id] = -R_c.T @ t_c; data.mocap_quat[mocap_id] = ro.mat2quat((R_c.T) @ ro.CV2MJ)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="cam"); rgb = renderer.render()
        renderer.enable_segmentation_rendering(); renderer.update_scene(data, camera="cam")
        seg = renderer.render(); renderer.disable_segmentation_rendering()
        alpha = (seg[:, :, 0] >= 0).astype(np.float32)
        img = str(Path(args.image_dir) / f"{stem}.jpg")
        bg = cv2.resize(cv2.imread(img), (Wr, Hr)).astype(np.float32)
        a = (alpha * 0.95)[:, :, None]
        overlay = (bg * (1 - a) + rgb[:, :, ::-1].astype(np.float32) * a).astype(np.uint8)
        cv2.imwrite(str(Path(args.render_dir) / f"{stem}.png"),
                    np.dstack([rgb[:, :, ::-1], (alpha * 255).astype(np.uint8)]))
        for P, c in [(ht[1], (0, 255, 0)), (ht[2], (0, 255, 0)),
                     (proj(R_c, t_c, rob_pts[1]), (0, 0, 255)), (proj(R_c, t_c, rob_pts[2]), (0, 0, 255))]:
            cv2.circle(overlay, (int(P[0] * sc), int(P[1] * sc)), 7, c, -1)
        cv2.imwrite(str(Path(args.overlay_dir) / f"{stem}.png"), overlay)
        print(f"  {stem}: thumb={eth:.0f}px index={ei:.0f}px")

    np.save("demo_out/align_params.npy", params)
    errs = np.array(errs)
    for i in [0, N // 3, 2 * N // 3, N - 1]:
        st = robot_files[i].name.replace('_0.npz', '')
        t = cv2.resize(cv2.imread(str(Path(args.overlay_dir) / f"{st}.png")), (640, 360))
        cv2.putText(t, st, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2); tiles.append(t)
    cv2.imwrite(str(Path(args.overlay_dir) / "_montage.png"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    print(f"thumb err: mean={errs[:,0].mean():.0f}px max={errs[:,0].max():.0f}px | index mean={errs[:,1].mean():.0f}px max={errs[:,1].max():.0f}px")
    print(f"Done -> {args.render_dir}, {args.overlay_dir}")


if __name__ == "__main__":
    main()
