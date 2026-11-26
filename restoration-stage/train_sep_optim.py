import os
import random
from typing import Dict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import structural_similarity_index_measure
from torchvision.utils import save_image
import torchvision.transforms as transforms

from model import LYT_DualBranch
from losses import (
    CombinedLoss,
    IRLoss,
    confidence_soft_target,
    confidence_weighted_l1,
    confidence_weighted_grad_l1,
)
from dataloader import TriPairedDataset
from torch.utils.data import DataLoader, Subset


# ==========================================================
# Metrics
# ==========================================================

def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """PSNR between two BCHW tensors in [0,1]."""
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-12)), 0, 1)
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse.item() == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """SSIM between two BCHW tensors in [0,1]."""
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-12)), 0, 1)
    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()


@torch.no_grad()
def validate(model, dataloader, device, gt_mean_vis=True, save_dir=None, save_first_n=0) -> Dict[str, float]:
    model.eval()
    total_psnr_v = 0.0
    total_ssim_v = 0.0
    total_psnr_i = 0.0
    n = 0

    if save_dir is not None and save_first_n > 0:
        os.makedirs(save_dir, exist_ok=True)

    for idx, (v_low, v_gt, i_noisy, i_gt) in enumerate(dataloader):
        v_low, v_gt = v_low.to(device), v_gt.to(device)
        i_noisy, i_gt = i_noisy.to(device), i_gt.to(device)

        v_hat, i_hat, _, _ = model(v_low, i_noisy)
        v_hat = torch.clamp(v_hat, 0, 1)
        i_hat = torch.clamp(i_hat, 0, 1)

        total_psnr_v += calculate_psnr(v_hat, v_gt, gt_mean=gt_mean_vis)
        total_ssim_v += calculate_ssim(v_hat, v_gt, gt_mean=gt_mean_vis)
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


# ==========================================================
# Main
# ==========================================================

def main():
    # ---------------------------------------------------------------------
    # Dataset Paths (EDIT THESE 4 PATHS)
    # ---------------------------------------------------------------------
    TRAIN_VIS_LOW_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Visible_LQ"
    TRAIN_VIS_HIGH_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Visible_HQ"
    TRAIN_IR_NOISY_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Infrared_gt_noisy1"
    TRAIN_IR_CLEAN_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/Infrared_gt"

    # ---------------------------------------------------------------------
    # Hyperparameters (you can decouple VIS/IR here)
    # ---------------------------------------------------------------------
    lr_vis = 2e-4
    lr_ir = 2e-4
    num_epochs = 300
    crop_size = 256
    batch_size = 1

    # Loss weights (independent, no cross-gradient)
    lambda_conf_vis = 0.05
    lambda_conf_ir = 0.05
    lambda_w_vis = 0.05
    lambda_w_ir = 0.05

    # Consistency regularization (per-branch, not cross-coupled)
    lambda_cons_vis = 0.05
    lambda_cons_ir = 0.05
    cons_sigma_vis = 0.02
    cons_sigma_ir = 0.02

    ir_grad_weight = 1.0
    conf_alpha = 10.0

    # Keep AMP OFF by default (same reason as original train.py)
    use_amp = False

    # Output folders
    save_root = "runs_sep_optim"
    os.makedirs(save_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | lr_vis={lr_vis} lr_ir={lr_ir} | epochs={num_epochs}")

    # ---------------------------------------------------------------------
    # Dataloaders (auto train/val split)
    # ---------------------------------------------------------------------
    VAL_RATIO = 0.1
    SPLIT_SEED = 42
    GT_MEAN_VIS = True

    transform = transforms.Compose([transforms.ToTensor()])

    # deterministic full-res dataset for stable index split
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
        transform=transform, crop_size=crop_size, training=True,
    )
    val_dataset = TriPairedDataset(
        TRAIN_VIS_LOW_DIR, TRAIN_VIS_HIGH_DIR, TRAIN_IR_NOISY_DIR, TRAIN_IR_CLEAN_DIR,
        transform=transform, crop_size=None, training=False,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    VIS_FILTERS = 32
    IR_FILTERS = 32
    IR_N_RESBLOCKS = 1
    IR_INIT_SCALE = 0.5
    IR_MAX_SCALE = 2.0

    model = LYT_DualBranch(
        vis_filters=VIS_FILTERS,
        ir_filters=IR_FILTERS,
        ir_n_resblocks=IR_N_RESBLOCKS,
        ir_init_scale=IR_INIT_SCALE,
        ir_max_scale=IR_MAX_SCALE,
    ).to(device)

    # ---------------------------------------------------------------------
    # Losses
    # ---------------------------------------------------------------------
    vis_criterion = CombinedLoss(device=device)
    ir_criterion = IRLoss(grad_weight=ir_grad_weight)

    # ---------------------------------------------------------------------
    # ✅ Separate optimizers + schedulers (VIS / IR are fully decoupled)
    # ---------------------------------------------------------------------
    optimizer_vis = optim.Adam(model.vis.parameters(), lr=lr_vis)
    optimizer_ir = optim.Adam(model.ir.parameters(), lr=lr_ir)

    scheduler_vis = CosineAnnealingLR(optimizer_vis, T_max=num_epochs)
    scheduler_ir = CosineAnnealingLR(optimizer_ir, T_max=num_epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available() and use_amp))

    # ---------------------------------------------------------------------
    # Logging / checkpoint
    # ---------------------------------------------------------------------
    sample_dir = os.path.join(save_root, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    ckpt_best_vis = os.path.join(save_root, "best_vis.pth")   # full model state, best by PSNR(V)
    ckpt_best_ir = os.path.join(save_root, "best_ir.pth")     # full model state, best by PSNR(IR)
    ckpt_last = os.path.join(save_root, "last.pth")

    best_psnr_vis = -1.0
    best_psnr_ir = -1.0

    print("Training started (separate optimizers).")

    # -----------------------------------------------------------------
    # Stochastic degradation for consistency regularization
    # -----------------------------------------------------------------
    def _stochastic_degrade_vis(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
        # x: (B,3,H,W) in [0,1]
        b = 0.9 + 0.2 * torch.rand((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype)
        g = 0.9 + 0.2 * torch.rand((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype)
        y = torch.clamp(x * b, 0.0, 1.0)
        y = torch.clamp(y, 1e-6, 1.0) ** g
        if sigma > 0:
            y = torch.clamp(y + torch.randn_like(y) * sigma, 0.0, 1.0)
        return y

    def _stochastic_degrade_ir(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
        # x: (B,1,H,W) in [0,1]
        c = 0.9 + 0.2 * torch.rand((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype)
        y = torch.clamp(x * c, 0.0, 1.0)
        if sigma > 0:
            y = torch.clamp(y + torch.randn_like(y) * sigma, 0.0, 1.0)
        return y

    def _normalize_feat(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = f.mean(dim=(2, 3), keepdim=True)
        std = f.std(dim=(2, 3), keepdim=True)
        return (f - mean) / (std + eps)

    # ==========================================================
    # Training Loop
    # ==========================================================
    for epoch in range(num_epochs):
        model.train()

        epoch_loss_vis_total = 0.0
        epoch_loss_ir_total = 0.0
        epoch_loss_vis_main = 0.0
        epoch_loss_ir_main = 0.0

        for (v_low, v_gt, i_noisy, i_gt) in train_loader:
            v_low, v_gt = v_low.to(device), v_gt.to(device)
            i_noisy, i_gt = i_noisy.to(device), i_gt.to(device)

            # ----------------------------------------------------------
            # (1) VIS update: only touches model.vis parameters
            # ----------------------------------------------------------
            optimizer_vis.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(torch.cuda.is_available() and use_amp)):
                if lambda_cons_vis > 0:
                    v_hat, v_conf, v_feat = model.vis(v_low, ir_hint=i_noisy, return_conf=True, return_feats=True)
                    v_low_2 = _stochastic_degrade_vis(v_low, sigma=cons_sigma_vis)
                    _, _, v_feat2 = model.vis(v_low_2, ir_hint=i_noisy, return_conf=True, return_feats=True)
                else:
                    v_hat, v_conf = model.vis(v_low, ir_hint=i_noisy, return_conf=True)
                    v_feat = v_feat2 = None

            with torch.cuda.amp.autocast(enabled=False):
                v_hat_f = v_hat.float().clamp(0, 1)
                v_gt_f = v_gt.float()
                v_conf_f = v_conf.float()

                # main VIS loss (original CombinedLoss)
                loss_vis = vis_criterion(v_gt_f, v_hat_f)

                # confidence supervision (VIS only)
                t_vis = confidence_soft_target(v_hat_f, v_gt_f, alpha=conf_alpha)
                loss_conf_vis = F.l1_loss(v_conf_f, t_vis)

                # reliability-weighted auxiliary loss (VIS only)
                loss_w_vis = (
                    confidence_weighted_l1(v_hat_f, v_gt_f, v_conf_f.detach())
                    + confidence_weighted_grad_l1(v_hat_f, v_gt_f, v_conf_f.detach())
                )

                # consistency (VIS only)
                if lambda_cons_vis > 0:
                    v1 = _normalize_feat(v_feat.float())
                    v2 = _normalize_feat(v_feat2.float())
                    loss_cons_vis = 0.5 * (F.l1_loss(v1, v2.detach()) + F.l1_loss(v2, v1.detach()))
                else:
                    loss_cons_vis = v_hat_f.new_tensor(0.0)

                loss_vis_total = (
                    loss_vis
                    + lambda_conf_vis * loss_conf_vis
                    + lambda_w_vis * loss_w_vis
                    + lambda_cons_vis * loss_cons_vis
                )

            if use_amp and torch.cuda.is_available():
                scaler.scale(loss_vis_total).backward()
                scaler.step(optimizer_vis)
                scaler.update()
            else:
                loss_vis_total.backward()
                optimizer_vis.step()

            # ----------------------------------------------------------
            # (2) IR update: only touches model.ir parameters
            # ----------------------------------------------------------
            optimizer_ir.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(torch.cuda.is_available() and use_amp)):
                if lambda_cons_ir > 0:
                    i_hat, i_conf, i_feat = model.ir(i_noisy, return_conf=True, return_feats=True)
                    i_noisy_2 = _stochastic_degrade_ir(i_noisy, sigma=cons_sigma_ir)
                    _, _, i_feat2 = model.ir(i_noisy_2, return_conf=True, return_feats=True)
                else:
                    i_hat, i_conf = model.ir(i_noisy, return_conf=True)
                    i_feat = i_feat2 = None

            with torch.cuda.amp.autocast(enabled=False):
                i_hat_f = i_hat.float().clamp(0, 1)
                i_gt_f = i_gt.float()
                i_conf_f = i_conf.float()

                # main IR loss (L1 + grad L1)
                loss_ir = ir_criterion(i_hat_f, i_gt_f)

                # confidence supervision (IR only)
                t_ir = confidence_soft_target(i_hat_f, i_gt_f, alpha=conf_alpha)
                loss_conf_ir = F.l1_loss(i_conf_f, t_ir)

                # reliability-weighted auxiliary loss (IR only)
                loss_w_ir = (
                    confidence_weighted_l1(i_hat_f, i_gt_f, i_conf_f.detach())
                    + confidence_weighted_grad_l1(i_hat_f, i_gt_f, i_conf_f.detach())
                )

                # consistency (IR only)
                if lambda_cons_ir > 0:
                    i1 = _normalize_feat(i_feat.float())
                    i2 = _normalize_feat(i_feat2.float())
                    loss_cons_ir = 0.5 * (F.l1_loss(i1, i2.detach()) + F.l1_loss(i2, i1.detach()))
                else:
                    loss_cons_ir = i_hat_f.new_tensor(0.0)

                loss_ir_total = (
                    loss_ir
                    + lambda_conf_ir * loss_conf_ir
                    + lambda_w_ir * loss_w_ir
                    + lambda_cons_ir * loss_cons_ir
                )

            if use_amp and torch.cuda.is_available():
                scaler.scale(loss_ir_total).backward()
                scaler.step(optimizer_ir)
                scaler.update()
            else:
                loss_ir_total.backward()
                optimizer_ir.step()

            # logging
            epoch_loss_vis_total += float(loss_vis_total.detach().cpu().item())
            epoch_loss_ir_total += float(loss_ir_total.detach().cpu().item())
            epoch_loss_vis_main += float(loss_vis.detach().cpu().item())
            epoch_loss_ir_main += float(loss_ir.detach().cpu().item())

        # ----------------------
        # Validate & Save
        # ----------------------
        metrics = validate(
            model, val_loader, device, gt_mean_vis=GT_MEAN_VIS,
            save_dir=os.path.join(sample_dir, f"epoch_{epoch+1:04d}") if (epoch + 1) % 50 == 0 else None,
            save_first_n=4,
        )

        denom = max(len(train_loader), 1)
        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"vis_total={epoch_loss_vis_total/denom:.4f} (vis_main={epoch_loss_vis_main/denom:.4f}) | "
            f"ir_total={epoch_loss_ir_total/denom:.4f} (ir_main={epoch_loss_ir_main/denom:.4f}) | "
            f"PSNR(V)={metrics['psnr_vis']:.3f} SSIM(V)={metrics['ssim_vis']:.4f} | "
            f"PSNR(IR)={metrics['psnr_ir']:.3f}"
        )

        scheduler_vis.step()
        scheduler_ir.step()

        # Save last
        torch.save(model.state_dict(), ckpt_last)

        # Save best by VIS / IR independently (still store FULL model to be loadable by test.py)
        if metrics["psnr_vis"] > best_psnr_vis:
            best_psnr_vis = metrics["psnr_vis"]
            torch.save(model.state_dict(), ckpt_best_vis)
            print(f"[CKPT] best_vis saved -> {ckpt_best_vis} (PSNR(V)={best_psnr_vis:.3f})")

        if metrics["psnr_ir"] > best_psnr_ir:
            best_psnr_ir = metrics["psnr_ir"]
            torch.save(model.state_dict(), ckpt_best_ir)
            print(f"[CKPT] best_ir  saved -> {ckpt_best_ir} (PSNR(IR)={best_psnr_ir:.3f})")


if __name__ == "__main__":
    main()
