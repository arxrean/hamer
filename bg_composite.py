"""Stage 4 (final): clean background + per-frame real cube + robot composite.

- Background: temporal-median plate over frames where each pixel is neither human NOR
  cube (no green ghost); LaMa fills pixels masked in every frame.
- Cube: the REAL per-frame cube (green mask + RGB) is kept as-is (its own pose/appearance).
- Robot: composited over, with the grasp rule -> thumb in front of the cube, cube in
  front of the other (non-thumb) fingers.
Also dumps the per-frame extracted cube for inspection (demo_out/cube_extracted/).
"""
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import inpaint_lama as il

LAMA_IN = "demo_out/lama_in"
ROBOT = "demo_out/franka_smooth_rgba"
THUMB = "demo_out/franka_smooth_thumb"
OUT = "demo_out/final"
CUBE_DIR = "demo_out/cube_extracted"
os.makedirs(OUT, exist_ok=True); os.makedirs(CUBE_DIR, exist_ok=True)
GREEN_LO, GREEN_HI = (30, 30, 25), (95, 255, 255)

frames = sorted(p for p in Path(LAMA_IN).glob("*.png") if not p.stem.endswith("_mask"))
imgs = np.stack([cv2.imread(str(p)).astype(np.float32) for p in frames])
masks = np.stack([cv2.imread(str(p.with_name(p.stem + "_mask.png")), 0) > 127 for p in frames])
N, H, W, _ = imgs.shape

def green_mask(img):
    return cv2.inRange(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV), GREEN_LO, GREEN_HI) > 0

# ---- background plate (exclude human AND cube) ----
cubes = np.stack([green_mask(imgs[j]) for j in range(N)])
valid = (~(masks | cubes))[..., None]
plate = np.nanmedian(np.where(valid, imgs, np.nan), axis=0)
never = np.isnan(plate).any(-1)
print(f"plate built; {int(never.sum())} px masked in all frames (LaMa fallback)")
if never.any():
    plate0 = np.nan_to_num(plate); device = "cuda" if torch.cuda.is_available() else "cpu"; gen = il.load_generator(device)
    it = torch.from_numpy(plate0[:, :, ::-1].copy() / 255.0).permute(2, 0, 1)[None].float().to(device)
    mt = torch.from_numpy(never.astype(np.float32))[None, None].to(device)
    itp, h, w = il.pad8(it); mtp, _, _ = il.pad8(mt)
    with torch.no_grad():
        pred = gen(torch.cat([itp * (1 - mtp), mtp], 1))
        filled = (mtp * pred + (1 - mtp) * itp)[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    plate = np.where(never[..., None], filled[:, :, ::-1] * 255.0, plate0)
plate = plate.astype(np.float32)

def lighting_match(frame, mask, plate, sigma=61):
    v = (~mask).astype(np.float32)
    num = cv2.GaussianBlur((frame - plate) * v[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(v, (0, 0), sigma)[..., None] + 1e-3
    return plate + num / den

def feather(mask):
    s = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((9, 9), np.uint8)).astype(np.float32)
    return cv2.GaussianBlur(s, (0, 0), 9) / 255.0

for i, p in enumerate(frames):
    stem = p.stem
    cube = cubes[i]                                       # real per-frame cube (green)
    # extracted cube dump (cube on neutral gray) for inspection
    ext = np.full_like(imgs[i], 128.0); ext[cube] = imgs[i][cube]
    cv2.imwrite(f"{CUBE_DIR}/{stem}.png", ext.astype(np.uint8))

    rr = cv2.resize(cv2.imread(f"{ROBOT}/{stem}.png", cv2.IMREAD_UNCHANGED), (W, H), interpolation=cv2.INTER_NEAREST)
    robot = rr[:, :, :3].astype(np.float32); a = rr[:, :, 3:4].astype(np.float32) / 255.0
    thumb = cv2.resize(cv2.imread(f"{THUMB}/{stem}.png", 0), (W, H), interpolation=cv2.INTER_NEAREST) > 127

    remove = masks[i] & (~cube)                           # remove human skin, keep cube
    plate_c = lighting_match(imgs[i], masks[i] | cube, plate)   # exclude cube from lighting match
    soft = feather(remove); soft[cube] = 0.0; soft = soft[..., None]
    bg = imgs[i] * (1 - soft) + plate_c * soft

    final = a * robot + (1 - a) * bg                      # robot over background+cube
    over = cube & (a[:, :, 0] > 0.5) & (~thumb)           # cube over non-thumb robot; thumb on top of cube
    final[over] = imgs[i][over]
    cv2.imwrite(f"{OUT}/{stem}.png", np.clip(final, 0, 255).astype(np.uint8))
print(f"composited {N} frames -> {OUT}")
