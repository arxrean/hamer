"""Stage 4b: background inpainting with big-lama, loaded directly (no PL/hydra/albumentations).

Builds LaMa's FFCResNetGenerator from big-lama/config.yaml, loads the generator weights
from big-lama/models/best.ckpt, and inpaints each frame in demo_out/lama_in/ (image +
_mask) -> demo_out/lama_out/. Runs in the hamer env (torch 2.11, numpy 2).
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

LAMA_REPO = os.path.expanduser("~/lama")
BIG_LAMA = os.path.join(LAMA_REPO, "models/big-lama")
sys.path.insert(0, LAMA_REPO)
from omegaconf import OmegaConf
from saicinpainting.training.modules.ffc import FFCResNetGenerator


def load_generator(device):
    cfg = OmegaConf.load(os.path.join(BIG_LAMA, "config.yaml"))
    g = OmegaConf.to_container(cfg.generator, resolve=True)
    g.pop("kind", None)
    gen = FFCResNetGenerator(**g)
    ckpt = torch.load(os.path.join(BIG_LAMA, "models/best.ckpt"), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    gen_sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    gen.load_state_dict(gen_sd, strict=True)
    return gen.to(device).eval()


def pad8(x):
    _, _, h, w = x.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="demo_out/lama_in")
    ap.add_argument("--outdir", default="demo_out/lama_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = load_generator(device)

    frames = sorted(p for p in Path(args.indir).glob("*.png") if not p.stem.endswith("_mask"))
    for fp in frames:
        mp = fp.with_name(fp.stem + "_mask.png")
        img = cv2.imread(str(fp))[:, :, ::-1].astype(np.float32) / 255.0          # RGB [0,1]
        mask = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)  # {0,1}
        it = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
        mt = torch.from_numpy(mask)[None, None].to(device)
        it, h, w = pad8(it); mt, _, _ = pad8(mt)
        with torch.no_grad():
            masked = it * (1 - mt)
            pred = gen(torch.cat([masked, mt], dim=1))      # (1,3,H,W) sigmoid
            out = mt * pred + (1 - mt) * it
        out = (out[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.outdir, fp.name), out[:, :, ::-1])
    print(f"inpainted {len(frames)} frames -> {args.outdir}")


if __name__ == "__main__":
    main()
