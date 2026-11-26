# ==========================================================
# train_finetune_vis_freeze_ir.py
# - Load a pretrained dual-branch model
# - Freeze IR branch
# - Finetune ONLY the VIS branch (and VIS confidence head)
#
# Run:
#   python train_finetune_vis_freeze_ir.py
# ==========================================================

import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import structural_similarity_index_measure
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import LYT_DualBranch
from dataloader import TriPairedDataset
from losses import CombinedLoss, confidence_soft_target, confidence_weighted_l1, confidence_weighted_grad_l1


# ======================================================================
# ✅ (MANUAL EDIT) only edit these 6 lines
# ======================================================================
GPU_ID = "0"
TRAIN_VIS_LOW_DIR  = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Visible_LQ"
TRAIN_VIS_HIGH_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Visible_HQ"
TRAIN_IR_NOISY_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Infrared_gt_noisy1"
TRAIN_IR_CLEAN_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Infrared_gt"
PRETRAINED_CKPT    = "/home/msia/Leeql/LYT-Net/V4.3.6/runs_mixed1_ir/best_model.pth"
SAVE_ROOT          = "./runs_finetune_vis_freeze_ir"


# ======================================================================
# Hyperparameters (reasonable defaults)
# ======================================================================
LR_VIS = 1e-4                 # finetune lr (VIS only)
NUM_EPOCHS = 120
BATCH_SIZE = 6
CROP_SIZE = 256
VAL_RATIO = 0.1
SPLIT_SEED = 42

USE_AMP = True
GT_MEAN_VIS = True            # keep your current PSNR definition (mean-aligned)

# ----- VIS-only confidence training -----
# (recommended) teach VIS confidence map to predict per-pixel reliability
LAMBDA_CONF_VIS = 0.05
CONF_ALPHA = 10.0

# (optional) confidence-weighted reconstruction (uses DETACHED conf => avoids degeneracy)
LAMBDA_W_VIS = 0.10

# (recommended) use the *frozen* IR branch output as a denoised hint to guide VIS
USE_IR_DENOISED_HINT = True


# ======================================================================
# Utils
# ======================================================================
def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_pixel_value: float = 1.0, gt_mean: bool = True) -> float:
    """PSNR between two BCHW tensors in [0,1]."""
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-12)), 0, 1)

    mse = F.mse_loss(img1, img2, reduction="mean")
    if mse.item() == 0:
        return float("inf")
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return float(psnr.item())


@torch.no_grad()
def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, max_pixel_value: float = 1.0, gt_mean: bool = True) -> float:
    """SSIM between two BCHW tensors in [0,1]."""
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-12)), 0, 1)

    return float(structural_similarity_index_measure(img1, img2, data_range=max_pixel_value).item())


def load_state(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # compatible with either:
    #   torch.save(model.state_dict(), path)
    # or a dict checkpoint containing a state_dict key
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] Loaded: {ckpt_path}")
    if missing:
        print(f"[CKPT] Missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[CKPT] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")


@torch.no_grad()
def validate_vis_only(
    model: LYT_DualBranch,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str = None,
    save_first_n: int = 0,
):
    """Validate VIS finetuning while IR is frozen (report both VIS and IR metrics)."""
    model.eval()
    total_psnr_v = 0.0
    total_ssim_v = 0.0
    total_psnr_i = 0.0
    n = 0

    if save_dir is not None and save_first_n > 0:
        os.makedirs(save_dir, exist_ok=True)

    for idx, (v_low, v_gt, i_noisy, i_gt) in enumerate(dataloader):
        v_low = v_low.to(device)
        v_gt = v_gt.to(device)
        i_noisy = i_noisy.to(device)
        i_gt = i_gt.to(device)

        # Frozen IR hint
        if USE_IR_DENOISED_HINT:
            i_hint = model.ir(i_noisy)  # (B,1,H,W)
        else:
            i_hint = i_noisy

        v_hat = model.vis(v_low, ir_hint=i_hint, return_conf=False)
        if isinstance(v_hat, tuple):
            v_hat = v_hat[0]
        v_hat = torch.clamp(v_hat, 0, 1)

        # IR output for reference metric (doesn't change because IR is frozen)
        i_hat = model.ir(i_noisy)
        if isinstance(i_hat, tuple):
            i_hat = i_hat[0]
        i_hat = torch.clamp(i_hat, 0, 1)

        total_psnr_v += calculate_psnr(v_hat, v_gt, gt_mean=GT_MEAN_VIS)
        total_ssim_v += calculate_ssim(v_hat, v_gt, gt_mean=GT_MEAN_VIS)
        total_psnr_i += calculate_psnr(i_hat, i_gt, gt_mean=False)
        n += 1

        if save_dir is not None and idx < save_first_n:
            save_image(v_hat, os.path.join(save_dir, f"vis_{idx:04d}.png"))
            save_image(i_hat, os.path.join(save_dir, f"ir_{idx:04d}.png"))

    return {
        "psnr_vis": total_psnr_v / max(n, 1),
        "ssim_vis": total_ssim_v / max(n, 1),
        "psnr_ir": total_psnr_i / max(n, 1),
    }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----------------------
    # Dataset + split
    # ----------------------
    transform = transforms.Compose([transforms.ToTensor()])

    base_dataset = TriPairedDataset(
        TRAIN_VIS_LOW_DIR, TRAIN_VIS_HIGH_DIR, TRAIN_IR_NOISY_DIR, TRAIN_IR_CLEAN_DIR,
        transform=transform, crop_size=None, training=False,
    )

    n = len(base_dataset)
    indices = list(range(n))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(indices)
    val_n = max(1, int(n * VAL_RATIO)) if VAL_RATIO > 0 else 0
    val_indices = indices[:val_n]
    train_indices = indices[val_n:]

    train_dataset = TriPairedDataset(
        TRAIN_VIS_LOW_DIR, TRAIN_VIS_HIGH_DIR, TRAIN_IR_NOISY_DIR, TRAIN_IR_CLEAN_DIR,
        transform=transform, crop_size=CROP_SIZE, training=True,
    )
    val_dataset = TriPairedDataset(
        TRAIN_VIS_LOW_DIR, TRAIN_VIS_HIGH_DIR, TRAIN_IR_NOISY_DIR, TRAIN_IR_CLEAN_DIR,
        transform=transform, crop_size=None, training=False,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Train iters: {len(train_loader)} | Val iters: {len(val_loader)} | Total images: {n}")

    # ----------------------
    # Model
    # ----------------------
    model = LYT_DualBranch().to(device)
    load_state(model, PRETRAINED_CKPT)

    # ✅ Freeze IR branch completely
    for p in model.ir.parameters():
        p.requires_grad = False
    model.ir.eval()  # IR stays fixed (no BN updates etc.)

    # ✅ Only VIS params are trainable
    trainable = [p for p in model.vis.parameters() if p.requires_grad]
    assert len(trainable) > 0, "No trainable parameters in VIS branch!"

    optimizer_vis = optim.Adam(model.vis.parameters(), lr=LR_VIS)
    scheduler_vis = CosineAnnealingLR(optimizer_vis, T_max=NUM_EPOCHS)

    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available() and USE_AMP))

    # Losses (VIS only)
    vis_criterion = CombinedLoss(device=device)

    run_dir = SAVE_ROOT
    os.makedirs(run_dir, exist_ok=True)
    ckpt_best = os.path.join(run_dir, "best_vis.pth")
    ckpt_last = os.path.join(run_dir, "last_vis.pth")
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    best_psnr = -1.0
    print("Finetune VIS started (IR frozen).")

    for epoch in range(NUM_EPOCHS):
        model.train()
        model.ir.eval()  # keep frozen
        epoch_loss = 0.0
        epoch_loss_vis = 0.0
        epoch_loss_conf = 0.0
        epoch_loss_wvis = 0.0

        for batch in train_loader:
            v_low, v_gt, i_noisy, _ = batch
            v_low = v_low.to(device)
            v_gt = v_gt.to(device)
            i_noisy = i_noisy.to(device)

            optimizer_vis.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(torch.cuda.is_available() and USE_AMP)):
                # --- Frozen IR hint (optional but recommended) ---
                if USE_IR_DENOISED_HINT:
                    with torch.no_grad():
                        i_hint = model.ir(i_noisy)  # (B,1,H,W)
                        if isinstance(i_hint, tuple):
                            i_hint = i_hint[0]
                else:
                    i_hint = i_noisy

                v_hat, v_conf = model.vis(v_low, ir_hint=i_hint, return_conf=True)

            # Compute losses in fp32 for stability
            with torch.cuda.amp.autocast(enabled=False):
                v_hat_f = torch.clamp(v_hat.float(), 0.0, 1.0)
                v_gt_f = v_gt.float()
                v_conf_f = v_conf.float()

                loss_vis = vis_criterion(v_gt_f, v_hat_f)

                if LAMBDA_CONF_VIS > 0:
                    t_vis = confidence_soft_target(v_hat_f, v_gt_f, alpha=CONF_ALPHA)
                    loss_conf_vis = F.l1_loss(v_conf_f, t_vis)
                else:
                    loss_conf_vis = torch.zeros((), device=device, dtype=torch.float32)

                if LAMBDA_W_VIS > 0:
                    loss_w_vis = confidence_weighted_l1(v_hat_f, v_gt_f, v_conf_f.detach()) + \
                                 confidence_weighted_grad_l1(v_hat_f, v_gt_f, v_conf_f.detach())
                else:
                    loss_w_vis = torch.zeros((), device=device, dtype=torch.float32)

                loss = loss_vis + LAMBDA_CONF_VIS * loss_conf_vis + LAMBDA_W_VIS * loss_w_vis

            if not torch.isfinite(loss).all():
                raise RuntimeError("Non-finite loss detected during VIS finetune.")

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer_vis)
                scaler.update()
            else:
                loss.backward()
                optimizer_vis.step()

            epoch_loss += float(loss.detach().cpu().item())
            epoch_loss_vis += float(loss_vis.detach().cpu().item())
            epoch_loss_conf += float(loss_conf_vis.detach().cpu().item())
            epoch_loss_wvis += float(loss_w_vis.detach().cpu().item())

        scheduler_vis.step()

        # ----- validation -----
        metrics = validate_vis_only(
            model,
            val_loader,
            device,
            save_dir=os.path.join(sample_dir, f"epoch_{epoch+1:04d}") if (epoch + 1) % 25 == 0 else None,
            save_first_n=3,
        )

        psnr_vis = metrics["psnr_vis"]
        ssim_vis = metrics["ssim_vis"]
        psnr_ir = metrics["psnr_ir"]

        print(
            f"ep [{epoch+1:03d}/{NUM_EPOCHS}] "
            f"loss={epoch_loss/len(train_loader):.4f} "
            f"(vis={epoch_loss_vis/len(train_loader):.4f}, conf={epoch_loss_conf/len(train_loader):.4f}, wvis={epoch_loss_wvis/len(train_loader):.4f}) | "
            f"VAL: PSNR(V)={psnr_vis:.3f} SSIM(V)={ssim_vis:.4f}  PSNR(IR,ref)={psnr_ir:.3f}"
        )

        # Save last (whole model state, to keep your test.py compatible)
        torch.save(model.state_dict(), ckpt_last)

        # Save best based on VIS PSNR only
        if psnr_vis > best_psnr:
            best_psnr = psnr_vis
            torch.save(model.state_dict(), ckpt_best)
            print(f"[SAVE] best_vis.pth updated: PSNR(V)={best_psnr:.3f}")

    print(f"Done. Best VIS PSNR = {best_psnr:.3f}")
    print(f"Saved: {ckpt_best}")


if __name__ == "__main__":
    main()
