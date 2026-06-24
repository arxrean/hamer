"""Stage 3 final: Franka+Allegro render with fingertip-contact-faithful wrist pose,
zero-lag temporal smoothing, and per-frame IK.

Pass 1: per frame, solve the wrist pose that lands thumb+index on the human fingertips
        (regime B), and read the retargeted finger qpos.
Smooth: centered (non-causal) moving average over wrist position, wrist orientation
        (quaternion), and finger qpos -> removes jitter WITHOUT lag.
Pass 2: IK the Franka arm to each smoothed wrist pose, render RGBA + overlay.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
from pathlib import Path
import numpy as np
import mujoco
import cv2
from scipy.optimize import least_squares
import render_overlay as ro
from dex_retargeting.constants import OPERATOR2MANO, HandType

ROT_CONST = (-39.9, 1.2, -25.4)
UDIR = np.array([0.5, -0.35, 0.0]); UDIR /= np.linalg.norm(UDIR)


def build():
    fr = os.path.expanduser("~/mujoco_menagerie/franka_emika_panda")
    al = os.path.expanduser("~/mujoco_menagerie/wonik_allegro")
    p = mujoco.MjSpec.from_file(os.path.join(fr, "panda_nohand.xml"))
    h = mujoco.MjSpec.from_file(os.path.join(al, "right_hand.xml"))
    p.attach(h, prefix="al_", site="attachment_site"); p.body("link0").add_freejoint()
    p.worldbody.add_body(name="cambody", pos=[0, 0, 0], quat=[0, 1, 0, 0]).add_camera(name="cam", fovy=45)
    cube = p.worldbody.add_body(name="cube"); cube.mocap = True   # green object, posed per frame
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.012, 0.012],
                  rgba=[0.42, 0.62, 0.18, 1.0], group=4)          # group 4 -> toggle on/off
    p.worldbody.add_light(pos=[0.3, -0.3, 1.0], diffuse=[.6, .6, .6])
    p.worldbody.add_light(pos=[-0.3, 0.3, 1.0], diffuse=[.5, .5, .5])
    p.visual.headlight.ambient = [.5, .5, .5]; p.visual.headlight.diffuse = [.4, .4, .4]
    p.visual.global_.offwidth = 1920; p.visual.global_.offheight = 1080
    return p.compile()


def smooth_vec(A, w):
    N = len(A); out = np.zeros_like(A); h = w // 2
    for i in range(N):
        out[i] = A[max(0, i - h):min(N, i + h + 1)].mean(0)
    return out


def smooth_quat(Q, w):
    Q = Q.copy()
    for i in range(1, len(Q)):
        if Q[i] @ Q[i - 1] < 0:
            Q[i] = -Q[i]
    N = len(Q); out = np.zeros_like(Q); h = w // 2
    for i in range(N):
        acc = Q[max(0, i - h):min(N, i + h + 1)].sum(0); out[i] = acc / np.linalg.norm(acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", default="demo_out/franka_smooth")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    m = build(); d = mujoco.MjData(m); cam_id = m.camera("cam").id
    files = sorted(Path("demo_out").glob("frame_*_0.npz"))
    ham0 = np.load(str(files[0])); focal = float(ham0["focal_length"]); W, H = [int(x) for x in ham0["img_size"]]
    m.cam_fovy[cam_id] = np.rad2deg(2 * np.arctan(H / (2 * focal))); m.stat.extent = 1.0
    m.vis.map.znear = 0.01; m.vis.map.zfar = 300.0
    badr = [m.jnt_qposadr[i] for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE][0]
    ajid = [m.joint(f"joint{i}").id for i in range(1, 8)]; aq = [m.jnt_qposadr[j] for j in ajid]
    ad = [m.jnt_dofadr[j] for j in ajid]; pbid = m.body("al_palm").id
    alq = {nm: m.jnt_qposadr[m.joint("al_" + ro.DEX2MJ[nm]).id] for nm in ro.DEX2MJ}
    sc = args.scale; Wr, Hr = int(W * sc), int(H * sc)
    renderer = mujoco.Renderer(m, Hr, Wr)
    vopt_nocube = mujoco.MjvOption(); vopt_nocube.geomgroup[4] = 0   # hide the sim cube (group 4)
    thumb_geoms = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("al_th")])
    def proj(P): return np.array([focal * P[0] / P[2] + W / 2, focal * P[1] / P[2] + H / 2])

    # ---- Pass 1: per-frame B solve ----
    N = len(files); P_pos = np.zeros((N, 3)); P_quat = np.zeros((N, 4)); P_q = np.zeros((N, 16)); HUM = []
    armp = [0, -0.6, 0, -1.8, 0, 1.6, 0.0]
    for k, f in enumerate(files):
        ham = np.load(str(f)); rob = np.load(str(Path("demo_out/robot") / f.name))
        kp = ham["keypoints_3d"].astype(float); ct = ham["cam_t_full"].astype(float); wframe = rob["wrist_frame"].astype(float)
        names = [str(x) for x in rob["joint_names"]]; qv = rob["qpos"]; P_q[k] = qv
        Qb = (wframe @ OPERATOR2MANO[HandType.right]) @ ro.euler_deg_to_mat(*ROT_CONST).T; tpos = kp[0] + ct
        for nm, v in zip(names, qv): d.qpos[alq[nm]] = float(v)
        d.qpos[badr:badr + 3] = tpos; bq0 = np.zeros(4); mujoco.mju_mat2Quat(bq0, Qb.reshape(9)); d.qpos[badr + 3:badr + 7] = bq0
        for i, a in enumerate(aq): d.qpos[a] = armp[i]
        mujoco.mj_forward(m, d); Rp = d.body("al_palm").xmat.reshape(3, 3); pp = d.body("al_palm").xpos.copy()
        def L(b, oz): bb = d.body(b); return Rp.T @ (bb.xpos + bb.xmat.reshape(3, 3) @ np.array([0, 0, oz]) - pp)
        Lth, Lidx, Lw = L("al_th_tip", 0.0423), L("al_ff_tip", 0.0267), np.zeros(3)
        hth, hidx, hw = proj(kp[4] + ct), proj(kp[8] + ct), proj(kp[0] + ct); HUM.append((hth, hidx))
        # optimize rotation(3) + IN-PLANE translation(2) only; depth fixed to human's
        # (2D fingertip matching can't constrain depth -> would jump wildly).
        def res(x):
            R = ro.euler_deg_to_mat(*x[:3]) @ Qb; p = tpos + np.array([x[3], x[4], 0.0]); e = []
            for Lp, htg, wt in [(Lth, hth, 3), (Lidx, hidx, 3), (Lw, hw, 1)]:
                e += list(wt * (proj(R @ Lp + p) - htg) / 50)
            return e
        x = least_squares(res, np.zeros(5)).x
        Rstar = ro.euler_deg_to_mat(*x[:3]) @ Qb; P_pos[k] = tpos + np.array([x[3], x[4], 0.0])
        qq = np.zeros(4); mujoco.mju_mat2Quat(qq, Rstar.reshape(9)); P_quat[k] = qq

    # ---- Smooth (centered, zero-lag) ----
    S_pos = smooth_vec(P_pos, args.window); S_quat = smooth_quat(P_quat, args.window); S_q = smooth_vec(P_q, args.window)
    jit_raw = np.abs(np.diff(P_pos, axis=0)).max(); jit_sm = np.abs(np.diff(S_pos, axis=0)).max()
    print(f"max frame-to-frame wrist-pos jump: raw {jit_raw*1000:.0f}mm -> smoothed {jit_sm*1000:.0f}mm")

    # ---- Pass 2: IK + render ----
    errs = []; armp = [0, -0.6, 0, -1.8, 0, 1.6, 0.0]
    for k, f in enumerate(files):
        stem = f.name.replace("_0.npz", ""); ham = np.load(str(f)); kp = ham["keypoints_3d"]; ct = ham["cam_t_full"]
        Rstar = np.zeros(9); mujoco.mju_quat2Mat(Rstar, S_quat[k]); Rstar = Rstar.reshape(3, 3); pstar = S_pos[k]
        names = [str(x) for x in np.load(str(Path("demo_out/robot")/f.name))["joint_names"]]
        for nm, v in zip(names, S_q[k]): d.qpos[alq[nm]] = float(v)
        d.qpos[badr:badr + 3] = pstar + UDIR * 0.70; bq = np.zeros(4); mujoco.mju_mat2Quat(bq, ro.euler_deg_to_mat(0, -90, 0).reshape(9)); d.qpos[badr + 3:badr + 7] = bq
        for i, a in enumerate(aq): d.qpos[a] = armp[i]
        for it in range(400):
            mujoco.mj_forward(m, d); cp = d.body("al_palm").xpos.copy(); cR = d.body("al_palm").xmat.reshape(3, 3)
            perr = pstar - cp; qd = np.zeros(4); mujoco.mju_mat2Quat(qd, (Rstar @ cR.T).reshape(9)); rerr = np.zeros(3); mujoco.mju_quat2Vel(rerr, qd, 1.0)
            if np.linalg.norm(perr) < 5e-4 and np.linalg.norm(rerr) < 5e-3: break
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, jp, jr, pbid); J = np.vstack([jp[:, ad], jr[:, ad]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([perr, rerr]))
            for i, a in enumerate(aq): d.qpos[a] = np.clip(d.qpos[a] + dq[i], m.jnt_range[ajid[i]][0], m.jnt_range[ajid[i]][1])
        armp = [d.qpos[a] for a in aq]; mujoco.mj_forward(m, d)
        eth = np.linalg.norm(proj(d.body("al_th_tip").xpos + d.body("al_th_tip").xmat.reshape(3, 3) @ [0, 0, 0.0423]) - HUM[k][0])
        ei = np.linalg.norm(proj(d.body("al_ff_tip").xpos + d.body("al_ff_tip").xmat.reshape(3, 3) @ [0, 0, 0.0267]) - HUM[k][1]); errs.append((eth, ei))
        bg = cv2.resize(cv2.imread(f"example_data/trim/{stem}.jpg"), (Wr, Hr)).astype(np.float32)
        os.makedirs(args.out + "_rgba", exist_ok=True); os.makedirs(args.out + "_thumb", exist_ok=True)
        renderer.update_scene(d, camera="cam", scene_option=vopt_nocube); rgb_r = renderer.render()
        renderer.enable_segmentation_rendering(); renderer.update_scene(d, camera="cam", scene_option=vopt_nocube); seg = renderer.render(); renderer.disable_segmentation_rendering()
        gid = seg[:, :, 0]
        ar = (gid >= 0).astype(np.uint8) * 255                       # robot alpha
        thumb = np.isin(gid, thumb_geoms).astype(np.uint8) * 255     # thumb pixels (for occlusion rule)
        cv2.imwrite(f"{args.out}_rgba/{stem}.png", np.dstack([rgb_r[:, :, ::-1], ar]))
        cv2.imwrite(f"{args.out}_thumb/{stem}.png", thumb)
        a = (ar.astype(np.float32) / 255.0 * 0.95)[:, :, None]
        cv2.imwrite(f"{args.out}/{stem}.png", (bg * (1 - a) + rgb_r[:, :, ::-1].astype(np.float32) * a).astype(np.uint8))
    errs = np.array(errs)
    print(f"fingertip px: thumb mean {errs[:,0].mean():.0f} max {errs[:,0].max():.0f} | index mean {errs[:,1].mean():.0f} max {errs[:,1].max():.0f}")
    rows = [cv2.putText(cv2.resize(cv2.imread(f"{args.out}/frame_{i:04d}.png"), (480, 270)), f"f{i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) for i in [1, 5, 9, 13, 17, 21, 25, 28]]
    cv2.imwrite(f"{args.out}/_strip.png", np.vstack([np.hstack(rows[:4]), np.hstack(rows[4:])]))
    j = [cv2.putText(cv2.resize(cv2.imread(f"{args.out}/frame_{i:04d}.png"), (480, 270)), f"f{i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) for i in [7, 8, 9, 10]]
    cv2.imwrite(f"{args.out}/_jump_7to10.png", np.hstack(j))
    print(f"wrote {args.out}/_strip.png and _jump_7to10.png")


if __name__ == "__main__":
    main()
