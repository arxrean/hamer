#!/usr/bin/env bash
# =============================================================================
# Human -> Robot (Allegro-on-Franka) video inpainting pipeline
#
#   Stage 1  HaMeR hand pose       -> per-frame MANO pose + camera   (demo.py)
#   Stage 2  retarget MANO->Allegro-> per-frame robot qpos           (retarget.py)
#   Stage 3  render Franka+Allegro -> robot RGBA + thumb mask        (franka_render.py)
#   Stage 4a SAM2 mask human       -> per-frame hand+arm masks       (sam_mask.py)
#   Stage 4b background + composite-> FINAL frames                   (bg_composite.py)
#
# Input :  example_data/trim/frame_XXXX.jpg   (static-camera clip, right hand)
# Output:  demo_out/final/frame_XXXX.png      (robot grasping, human removed)
#
# Run:     bash run_pipeline.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"                                  # run from the hamer/ repo root

PY=/home/zkou/.conda/envs/hamer/bin/python           # the 'hamer' conda env python
IMG_DIR=example_data/trim                             # input frames
OUT=demo_out                                          # all intermediate + final outputs
URDF_DIR=$HOME/dex-retargeting/assets/robots/hands    # dex-retargeting robot URDFs/meshes
export MUJOCO_GL=egl                                  # headless offscreen rendering

# =============================================================================
# ONE-TIME SETUP ON A NEW MACHINE (run these once, not part of the pipeline)
# -----------------------------------------------------------------------------
# 1) Base HaMeR conda env 'hamer' (torch cu128, detectron2, chumpy, mmcv,
#    ViTPose, xtcocotools rebuilt for numpy2, etc.):   bash setup.sh
#
# 2) Extra Python deps added on top of setup.sh (into the 'hamer' env):
#      $PY -m pip install dex_retargeting mujoco                 # retarget + sim
#      $PY -m pip install --upgrade "PyOpenGL==3.1.7"            # needed for MuJoCo EGL
#      $PY -m pip install easydict kornia scikit-learn           # LaMa generator deps
#    Keep numpy==2.2.6 / opencv-python==4.13 / pillow>=10 — do NOT let any install
#    downgrade numpy (e.g. simple-lama-inpainting pins numpy<2 and breaks the env).
#
# 3) HaMeR model data:
#      bash fetch_demo_data.sh                 # checkpoints -> ./_DATA
#      # place MANO_RIGHT.pkl in _DATA/data/mano/   (register at mano.is.tue.mpg.de)
#
# 4) External repos + checkpoints (clone next to ~, i.e. $HOME), from github:
#      git clone https://github.com/dexsuite/dex-retargeting.git ~/dex-retargeting
#      ( cd ~/dex-retargeting && git submodule update --init )         # -> assets/
#      git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
#      git clone https://github.com/facebookresearch/sam2.git ~/sam2
#      ( cd ~/sam2 && $PY -m pip install -e . && cd checkpoints && ./download_ckpts.sh )
#      git clone https://github.com/advimman/lama.git ~/lama
#      # download big-lama.zip per lama/README and unzip to ~/lama/models/big-lama
#
# Paths hard-coded in the scripts (edit if your layout differs):
#   sam_mask.py     : ~/sam2/checkpoints/sam2.1_hiera_large.pt , config sam2.1_hiera_l.yaml
#   franka_render.py: ~/mujoco_menagerie/{wonik_allegro,franka_emika_panda}
#   inpaint_lama.py : ~/lama  + ~/lama/models/big-lama   (used by bg_composite fallback)
# =============================================================================

# --- Stage 1: HaMeR hand pose (RIGHT hand only) ------------------------------
# demo.py (modified) -> per hand: demo_out/frame_XXXX_0.npz
#   keys: keypoints_3d, vertices, global_orient/hand_pose (aa+rotmat), betas,
#         cam_t_full, focal_length, img_size, box_center/size, is_right
#   + demo_out/frame_XXXX_all.jpg (MANO mesh overlay sanity check)
$PY demo.py --img_folder "$IMG_DIR" --out_folder "$OUT"

# --- Stage 2: retarget MANO -> Allegro (DexPilot, no temporal smoothing) ------
# retarget.py -> demo_out/robot/frame_XXXX_0.npz
#   keys: qpos (16 Allegro joints), joint_names, wrist_frame, wrist_t, is_right
# Key: human keypoints canonicalized via the keypoint-estimated wrist frame +
# OPERATOR2MANO (NOT MANO global_orient); low_pass_alpha=1.0 (grasp stays in phase).
$PY retarget.py --urdf_dir "$URDF_DIR"

# --- Stage 3: render Franka(arm)+Allegro(hand), contact-aligned, smoothed -----
# franka_render.py:
#   * Panda(no-hand) + Allegro via MjSpec.attach; fixed camera = HaMeR intrinsics
#   * per frame: solve wrist pose so thumb+index land on the human fingertips,
#                depth pinned to human (robot scales with human)
#   * zero-lag (centered, window=3) smoothing of wrist pose + qpos
#   * base anchored off-frame upper-right; IK the 7 arm joints to the wrist
#   * outputs: franka_smooth/ (overlay preview), franka_smooth_rgba/ (robot RGBA),
#              franka_smooth_thumb/ (thumb-pixel mask, for the grasp occlusion rule)
$PY franka_render.py --window 3 --scale 0.5 --out "$OUT/franka_smooth"

# --- Stage 4a: mask the human hand+arm with SAM 2 -----------------------------
# sam_mask.py: seed frame 0 with HaMeR hand keypoints (+ forearm points), propagate
# through the clip -> demo_out/lama_in/frame_XXXX.png + _mask.png  (1920x1080).
$PY sam_mask.py --img_dir "$IMG_DIR" --hamer_dir "$OUT" --out "$OUT/lama_in"

# --- Stage 4b: clean background + per-frame cube + robot composite -------------
# bg_composite.py:
#   * background = temporal-median plate over frames where each pixel is neither
#     human NOR cube (no green ghost) + lighting-match (no seams); big-lama fills
#     pixels masked in every frame (via inpaint_lama.load_generator, ~/lama/models/big-lama)
#   * keep the REAL per-frame cube (green mask + RGB)
#   * composite robot with grasp rule: thumb in front of cube, cube over other fingers
#   * also dumps demo_out/cube_extracted/ (per-frame cube cut-out, inspection)
$PY bg_composite.py
echo "Done -> $OUT/final/   (robot grasping the cube on a human-removed background)"
