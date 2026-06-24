"""Stage 4c: final composite.

For each frame: inpaint only the human pixels NOT covered by the robot
(human_mask AND NOT robot_alpha) with big-lama -> small regions, clean fill;
then alpha-composite the robot RGBA on top.

Inputs:
  demo_out/lama_in/frame_XXXX.png         resized original frame (1920x1080)
  demo_out/lama_in/frame_XXXX_mask.png    human hand+arm mask (SAM2)
  demo_out/franka_smooth_rgba/frame_XXXX.png  robot RGBA (from franka_render.py)
Output:
  demo_out/final/frame_XXXX.png
"""
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import inpaint_lama as il   # load_generator, pad8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lama_in", default="demo_out/lama_in")
    ap.add_argument("--robot_rgba", default="demo_out/franka_smooth_rgba")
    ap.add_argument("--out", default="demo_out/final")
    ap.add_argument("--dilate", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = il.load_generator(device)
    k = np.ones((args.dilate, args.dilate), np.uint8)

    frames = sorted(p for p in Path(args.lama_in).glob("*.png") if not p.stem.endswith("_mask"))
    for fp in frames:
        stem = fp.stem
        img = cv2.imread(str(fp))[:, :, ::-1].astype(np.float32) / 255.0           # RGB [0,1]
        hmask = cv2.imread(str(fp.with_name(stem + "_mask.png")), cv2.IMREAD_GRAYSCALE) > 127
        rgba = cv2.imread(f"{args.robot_rgba}/{stem}.png", cv2.IMREAD_UNCHANGED)    # BGRA
        rgba = cv2.resize(rgba, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        robot_bgr = rgba[:, :, :3].astype(np.float32) / 255.0
        ralpha = (rgba[:, :, 3].astype(np.float32) / 255.0)

        # inpaint only where human is present AND robot does NOT cover it
        inpaint_mask = (hmask & (ralpha < 0.04)).astype(np.uint8) * 255
        inpaint_mask = cv2.dilate(inpaint_mask, k)
        mt = torch.from_numpy((inpaint_mask > 127).astype(np.float32))[None, None].to(device)
        it = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
        itp, h, w = il.pad8(it); mtp, _, _ = il.pad8(mt)
        with torch.no_grad():
            pred = gen(torch.cat([itp * (1 - mtp), mtp], dim=1))
            cleaned = (mtp * pred + (1 - mtp) * itp)[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        cleaned_bgr = cleaned[:, :, ::-1]                                            # RGB->BGR [0,1]

        a = ralpha[:, :, None]
        final = (a * robot_bgr + (1 - a) * cleaned_bgr) * 255.0
        cv2.imwrite(f"{args.out}/{stem}.png", final.astype(np.uint8))
    print(f"composited {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
