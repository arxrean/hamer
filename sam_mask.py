"""Stage 4a: SAM 2 video masking of the human hand+arm.

Seeds frame 0 with HaMeR hand keypoints (+ a few forearm points) and propagates
through the clip. Exports, per frame, the image and a dilated binary mask
(white = remove) into a LaMa-ready input folder.

Output: demo_out/lama_in/frame_XXXX.png  +  frame_XXXX_mask.png   (1920x1080)
"""
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2_video_predictor

SAM_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM_CKPT = os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_large.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="example_data/trim")
    ap.add_argument("--hamer_dir", default="demo_out")
    ap.add_argument("--out", default="demo_out/lama_in")
    ap.add_argument("--rw", type=int, default=1920)
    ap.add_argument("--rh", type=int, default=1080)
    ap.add_argument("--dilate", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    src = sorted(Path(args.img_dir).glob("frame_*.jpg"))
    stems = [s.stem for s in src]
    # SAM2 video predictor wants a dir of <idx>.jpg frames
    fdir = "demo_out/sam_frames"; os.makedirs(fdir, exist_ok=True)
    for i, s in enumerate(src):
        cv2.imwrite(f"{fdir}/{i}.jpg", cv2.resize(cv2.imread(str(s)), (args.rw, args.rh)))

    # frame-0 prompts from HaMeR (projected hand keypoints + forearm points)
    ham = np.load(str(Path(args.hamer_dir) / f"{stems[0]}_0.npz"))
    kp = ham["keypoints_3d"]; ct = ham["cam_t_full"]; focal = float(ham["focal_length"]); W, H = [int(x) for x in ham["img_size"]]
    sx, sy = args.rw / W, args.rh / H
    def proj(i):
        P = kp[i] + ct
        return [(focal * P[0] / P[2] + W / 2) * sx, (focal * P[1] / P[2] + H / 2) * sy]
    hand_pts = np.array([proj(i) for i in range(21)], dtype=np.float32)
    wrist = np.array(proj(0))
    arm = np.array([wrist + [150, -40], wrist + [320, -90], wrist + [520, -150]], dtype=np.float32)
    pts = np.concatenate([hand_pts, arm]); labels = np.ones(len(pts), dtype=np.int32)

    predictor = build_sam2_video_predictor(SAM_CFG, SAM_CKPT, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=fdir)
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=pts, labels=labels)
        masks = {}
        for fidx, obj_ids, mlogits in predictor.propagate_in_video(state):
            masks[fidx] = (mlogits[0] > 0).cpu().numpy()[0]

    k = np.ones((args.dilate, args.dilate), np.uint8)
    for i, stem in enumerate(stems):
        img = cv2.imread(f"{fdir}/{i}.jpg")
        mk = (masks[i].astype(np.uint8) * 255)
        if args.dilate > 0:
            mk = cv2.dilate(mk, k)
        cv2.imwrite(f"{args.out}/{stem}.png", img)
        cv2.imwrite(f"{args.out}/{stem}_mask.png", mk)
    print(f"wrote {len(stems)} frame+mask pairs -> {args.out}")


if __name__ == "__main__":
    main()
