"""Inference for Stage-2 Full CARA-v2 fusion.

Output fused images with the same basename as visible images.
"""

from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from dataloader_fusion import create_fusion_dataloader
from model import LYT_DualBranch
from fusion_net_cara_full import FusionNetCARAFull
from utils import load_state_dict_safely


VIS_LOW_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/LLVIP200/VIS_LQ"
IR_NOISY_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/LLVIP200/Infrared_gt_noisy1"
RESTORER_CKPT = "/home/msia/Leeql/LYTFusion/V4.3.1/best_model.pth"
FUSION_CKPT = "/home/msia/Leeql/LYTFusion/Code_CARA/fusion-stagev2/runs_fusion_cara_w/full_108/checkpoints/fusion_cara_full_epoch_0100.pth"
OUT_DIR = "/home/msia/Leeql/LYTFusion/Code_CARA/fusion-stagev2/runs_fusion_cara_w/full_108/results_LLVIP_100"

BATCH_SIZE = 1
NUM_WORKERS = 2
BASE_CHANNELS = 32
RES_SCALE = 0.45
OUTPUT_BASE = "mean"
REJECT_STRENGTH = 0.60
SAVE_MAPS = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(root: str) -> List[str]:
    files = []
    for name in os.listdir(root):
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            files.append(name)
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No image files found in {root}")
    return files


def output_name_from_vis(vis_name: str, out_ext: str = ".png") -> str:
    return os.path.splitext(os.path.basename(vis_name))[0] + out_ext


def _load_fusion_state(model: torch.nn.Module, path: str):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("fusion", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")


@torch.no_grad()
def save_maps(maps: Dict[str, torch.Tensor], out_dir: str, vis_name: str, size):
    os.makedirs(out_dir, exist_ok=True)
    keys = [
        "s1_w_vis", "s1_w_ir", "s1_w_com", "s1_w_rej",
        "s1_conflict", "residual_cue", "ir_hf", "ir_stripe", "struct_conflict", "semantic_prior",
    ]
    imgs = []
    for k in keys:
        if k not in maps:
            continue
        m = F.interpolate(maps[k], size=size, mode="bilinear", align_corners=False).clamp(0, 1)
        imgs.append(m.repeat(1, 3, 1, 1))
    if imgs:
        stem = os.path.splitext(os.path.basename(vis_name))[0]
        save_image(torch.cat(imgs, dim=0), os.path.join(out_dir, f"{stem}_maps.png"), nrow=1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    if SAVE_MAPS:
        os.makedirs(os.path.join(OUT_DIR, "maps"), exist_ok=True)

    vis_names = list_image_files(VIS_LOW_DIR)
    loader = create_fusion_dataloader(
        vis_low_dir=VIS_LOW_DIR,
        ir_noisy_dir=IR_NOISY_DIR,
        crop_size=None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        training=False,
    )
    if len(vis_names) != len(loader.dataset):
        print(
            f"[WARN] Number of VIS files ({len(vis_names)}) != dataset length ({len(loader.dataset)}). "
            "Check dataloader sorting/filtering if output names are misaligned."
        )

    restorer = LYT_DualBranch().to(device)
    load_state_dict_safely(restorer, RESTORER_CKPT)
    restorer.eval()

    fusion = FusionNetCARAFull(BASE_CHANNELS, RES_SCALE, OUTPUT_BASE, REJECT_STRENGTH).to(device)
    _load_fusion_state(fusion, FUSION_CKPT)
    fusion.eval()

    global_idx = 0
    for _, (vis_low, ir_noisy) in enumerate(loader):
        vis_low = vis_low.to(device)
        ir_noisy = ir_noisy.to(device)
        with torch.no_grad():
            vis_hat, ir_hat, vis_conf, ir_conf = restorer(vis_low, ir_noisy)
            _, fused_rgb, maps = fusion(
                vis_hat.clamp(0, 1),
                ir_hat.clamp(0, 1),
                vis_conf.clamp(0, 1),
                ir_conf.clamp(0, 1),
                return_maps=True,
            )

        batch_size = fused_rgb.size(0)
        for b in range(batch_size):
            if global_idx < len(vis_names):
                vis_name = vis_names[global_idx]
                out_name = output_name_from_vis(vis_name, out_ext=".png")
            else:
                vis_name = f"{global_idx:04d}.png"
                out_name = f"{global_idx:04d}.png"

            save_image(fused_rgb[b:b + 1], os.path.join(OUT_DIR, out_name))
            if SAVE_MAPS:
                single_maps = {}
                for k, v in maps.items():
                    if torch.is_tensor(v) and v.size(0) == batch_size:
                        single_maps[k] = v[b:b + 1]
                    else:
                        single_maps[k] = v
                save_maps(single_maps, os.path.join(OUT_DIR, "maps"), vis_name, fused_rgb.shape[-2:])
            print(f"saved {out_name}")
            global_idx += 1


if __name__ == "__main__":
    main()
