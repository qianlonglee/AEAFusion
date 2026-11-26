"""Train Stage-2 Full CARA-v2 fusion network.

Before running, place this file in Code_CARA/fusion-stage/ and edit paths below.
Stage-1 is loaded and frozen. Only the Stage-2 CARA fusion network is trained.
"""

from __future__ import annotations

import os
import random
import time
from typing import Dict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

from dataloader_fusion import create_fusion_dataloader
from model import LYT_DualBranch
from fusion_net_cara_full import FusionNetCARAFull
from losses_fusion_cara_full import (
    fusion_loss_base,
    max_evidence_feature_loss,
    cara_full_regularization,
    low_frequency_consistency_loss,
)
from utils import load_state_dict_safely


# -----------------------------------------------------------------------------
# Paths: edit these before training.
# -----------------------------------------------------------------------------

VIS_LOW_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Visible_LQ"
IR_NOISY_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Infrared_gt_noisy1"
RESTORER_CKPT = "/home/msia/Leeql/LYTFusion/V4.3.1/best_model.pth"
OUT_DIR = "/home/msia/Leeql/LYTFusion/Code_CARA/fusion-stagev2/runs_Hyperparameter"


# -----------------------------------------------------------------------------
# Training config
# -----------------------------------------------------------------------------

SEED = 42
EPOCHS = 2000
BATCH_SIZE = 16
NUM_WORKERS = 4
CROP_SIZE = 256
LR = 1e-4
USE_AMP = False

# Fusion losses.
W_INT = 1.0
W_GRAD = 10.0
LAMBDA_FEAT = 0.05
LAMBDA_CARA = 0.10
LAMBDA_LOW = 0.20

# Full CARA-v2 network.
BASE_CHANNELS = 32
RES_SCALE = 0.45
OUTPUT_BASE = "mean"       # "mean" or "vis"
REJECT_STRENGTH = 0.60     # beta in F_base + (1-beta*w_rej)*F_detail

SAVE_MODEL_EVERY = 100
SAVE_SAMPLE_EVERY = 10
SAVE_BEST_BY_LOSS = True


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _to_rgb(x: torch.Tensor) -> torch.Tensor:
    return x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)


@torch.no_grad()
def save_visual_samples(
    vis_low: torch.Tensor,
    ir_noisy: torch.Tensor,
    vis_hat: torch.Tensor,
    ir_hat: torch.Tensor,
    fused_rgb: torch.Tensor,
    maps: Dict[str, torch.Tensor],
    out_dir: str,
    epoch: int,
):
    os.makedirs(out_dir, exist_ok=True)
    grid = torch.cat([vis_low, _to_rgb(ir_noisy), vis_hat, _to_rgb(ir_hat), fused_rgb], dim=0)
    save_image(grid, os.path.join(out_dir, f"epoch_{epoch:03d}_images.png"), nrow=vis_low.size(0))

    h, w = vis_low.shape[-2:]
    keys = [
        "s1_w_vis", "s1_w_ir", "s1_w_com", "s1_w_rej",
        "s1_conflict", "residual_cue", "ir_hf", "ir_stripe", "struct_conflict", "semantic_prior",
    ]
    map_imgs = []
    for k in keys:
        if k not in maps:
            continue
        m = F.interpolate(maps[k], size=(h, w), mode="bilinear", align_corners=False)
        map_imgs.append(m.clamp(0, 1).repeat(1, 3, 1, 1))
    if map_imgs:
        save_image(torch.cat(map_imgs, dim=0), os.path.join(out_dir, f"epoch_{epoch:03d}_maps.png"), nrow=vis_low.size(0))


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints")
    sample_dir = os.path.join(OUT_DIR, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    print(f"Device: {device} | epochs={EPOCHS} batch={BATCH_SIZE} crop={CROP_SIZE} lr={LR}")
    print(
        f"Full CARA-v2: base={BASE_CHANNELS}, res_scale={RES_SCALE}, "
        f"output_base={OUTPUT_BASE}, reject_strength={REJECT_STRENGTH}"
    )

    train_loader = create_fusion_dataloader(
        vis_low_dir=VIS_LOW_DIR,
        ir_noisy_dir=IR_NOISY_DIR,
        crop_size=CROP_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        training=True,
    )
    print(f"Train loader iters/epoch: {len(train_loader)}")

    # Frozen Stage-1 restoration/confidence provider.
    restorer = LYT_DualBranch().to(device)
    load_state_dict_safely(restorer, RESTORER_CKPT)
    restorer.eval()
    for p in restorer.parameters():
        p.requires_grad = False

    fuse_net = FusionNetCARAFull(
        base_channels=BASE_CHANNELS,
        res_scale=RES_SCALE,
        output_base=OUTPUT_BASE,
        reject_strength=REJECT_STRENGTH,
    ).to(device)
    optimizer = optim.AdamW(fuse_net.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available() and USE_AMP))

    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        fuse_net.train()
        t0 = time.time()
        meters = {"total": 0.0, "fuse": 0.0, "int": 0.0, "grad": 0.0, "feat": 0.0, "cara": 0.0, "low": 0.0}

        for vis_low, ir_noisy in train_loader:
            vis_low = vis_low.to(device, non_blocking=True)
            ir_noisy = ir_noisy.to(device, non_blocking=True)

            with torch.no_grad():
                vis_hat, ir_hat, vis_conf, ir_conf = restorer(vis_low, ir_noisy)
                vis_hat = vis_hat.clamp(0, 1)
                ir_hat = ir_hat.clamp(0, 1)
                vis_conf = vis_conf.clamp(0, 1)
                ir_conf = ir_conf.clamp(0, 1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(torch.cuda.is_available() and USE_AMP)):
                _, fused_rgb, maps = fuse_net(
                    vis_hat,
                    ir_hat,
                    vis_conf,
                    ir_conf,
                    return_feats=True,
                    return_maps=True,
                )
                loss_fuse, loss_int, loss_grad = fusion_loss_base(
                    vi=vis_hat,
                    ir=ir_hat,
                    fu=fused_rgb,
                    w_int=W_INT,
                    w_grad=W_GRAD,
                )
                loss_feat = max_evidence_feature_loss(maps["r2"], maps["t2"], maps["f2"])
                loss_cara, _ = cara_full_regularization(maps)
                loss_low = low_frequency_consistency_loss(vis_hat, ir_hat, fused_rgb)
                loss_total = loss_fuse + LAMBDA_FEAT * loss_feat + LAMBDA_CARA * loss_cara + LAMBDA_LOW * loss_low

            if not torch.isfinite(loss_total):
                raise RuntimeError("Non-finite Stage-2 loss. Check input normalization and loss weights.")

            if USE_AMP and torch.cuda.is_available():
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fuse_net.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(fuse_net.parameters(), max_norm=5.0)
                optimizer.step()

            meters["total"] += float(loss_total.detach())
            meters["fuse"] += float(loss_fuse.detach())
            meters["int"] += float(loss_int.detach())
            meters["grad"] += float(loss_grad.detach())
            meters["feat"] += float(loss_feat.detach())
            meters["cara"] += float(loss_cara.detach())
            meters["low"] += float(loss_low.detach())

        n = max(1, len(train_loader))
        for k in meters:
            meters[k] /= n

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | total={meters['total']:.4f} "
            f"fuse={meters['fuse']:.4f} int={meters['int']:.4f} grad={meters['grad']:.4f} "
            f"feat={meters['feat']:.4f} cara={meters['cara']:.4f} low={meters['low']:.4f} "
            f"| time={time.time() - t0:.1f}s"
        )

        if epoch % SAVE_MODEL_EVERY == 0:
            checkpoint = {"fusion": fuse_net.state_dict(), "epoch": epoch, "meters": meters}
            epoch_path = os.path.join(ckpt_dir, f"fusion_cara_full_epoch_{epoch:04d}.pth")
            torch.save(checkpoint, epoch_path)
            torch.save(checkpoint, os.path.join(ckpt_dir, "fusion_cara_full_last.pth"))

            # Select the best model among checkpoint epochs only.
            if SAVE_BEST_BY_LOSS and meters["total"] < best_loss:
                best_loss = meters["total"]
                torch.save(checkpoint, os.path.join(ckpt_dir, "fusion_cara_full_best.pth"))

            print(f"Saved checkpoint: {epoch_path}")

        if epoch % SAVE_SAMPLE_EVERY == 0 or epoch == 1:
            vis_low_b, ir_noisy_b = next(iter(train_loader))
            vis_low_b = vis_low_b.to(device)
            ir_noisy_b = ir_noisy_b.to(device)
            with torch.no_grad():
                vis_hat_b, ir_hat_b, vis_conf_b, ir_conf_b = restorer(vis_low_b, ir_noisy_b)
                _, fused_rgb_b, maps_b = fuse_net(
                    vis_hat_b.clamp(0, 1),
                    ir_hat_b.clamp(0, 1),
                    vis_conf_b.clamp(0, 1),
                    ir_conf_b.clamp(0, 1),
                    return_maps=True,
                )
            save_visual_samples(vis_low_b, ir_noisy_b, vis_hat_b, ir_hat_b, fused_rgb_b, maps_b, sample_dir, epoch)

    print(f"Done. Last checkpoint: {os.path.join(ckpt_dir, 'fusion_cara_full_last.pth')}")


if __name__ == "__main__":
    main()
