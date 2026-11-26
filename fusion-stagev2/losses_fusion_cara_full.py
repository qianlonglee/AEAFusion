"""Losses for Stage-2 Full CARA-v2 fusion."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def rgb_to_y(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.dim() != 4 or rgb.size(1) != 3:
        raise ValueError(f"Expected RGB BCHW, got {tuple(rgb.shape)}")
    return (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]).clamp(0.0, 1.0)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")
    c = x.size(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def fusion_loss_base(
    vi: torch.Tensor,
    ir: torch.Tensor,
    fu: torch.Tensor,
    w_int: float = 1.0,
    w_grad: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard unsupervised IVIF intensity + gradient loss."""
    if vi.size(1) != 3 or fu.size(1) != 3 or ir.size(1) != 1:
        raise ValueError(f"Shape mismatch: vi{tuple(vi.shape)} ir{tuple(ir.shape)} fu{tuple(fu.shape)}")
    vi_y = rgb_to_y(vi)
    fu_y = rgb_to_y(fu)
    target_int = torch.max(vi_y, ir)
    loss_int = F.l1_loss(fu_y, target_int)

    grad_vi = sobel_magnitude(vi_y)
    grad_ir = sobel_magnitude(ir)
    grad_target = torch.max(grad_vi, grad_ir)
    loss_grad = F.l1_loss(sobel_magnitude(fu_y), grad_target)
    return float(w_int) * loss_int + float(w_grad) * loss_grad, loss_int, loss_grad


Fusion_Loss_Base = fusion_loss_base


def max_evidence_feature_loss(
    vis_feat: torch.Tensor,
    ir_feat: torch.Tensor,
    fused_feat: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Feature-energy preservation loss.

    This is not a semantic loss unless the features come from a semantic model.
    Use the paper name 'max-evidence feature preservation' instead of
    'Max-Feature Semantic Loss'.
    """
    if vis_feat.dim() != 4 or ir_feat.dim() != 4 or fused_feat.dim() != 4:
        raise ValueError("Expected features in BCHW.")
    sv = torch.sqrt(torch.sum(vis_feat * vis_feat, dim=1, keepdim=True) + eps)
    si = torch.sqrt(torch.sum(ir_feat * ir_feat, dim=1, keepdim=True) + eps)
    sf = torch.sqrt(torch.sum(fused_feat * fused_feat, dim=1, keepdim=True) + eps)
    return F.l1_loss(sf, torch.max(sv, si).detach())


MaxFeatureSemanticLoss = max_evidence_feature_loss


def _tv_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] < 2 or x.shape[-2] < 2:
        return x.new_tensor(0.0)
    return (x[..., 1:] - x[..., :-1]).abs().mean() + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()


def low_frequency_consistency_loss(
    vi: torch.Tensor,
    ir: torch.Tensor,
    fu: torch.Tensor,
    kernel_size: int = 9,
) -> torch.Tensor:
    """Low-frequency consistency to stabilize brightness and naturalness.

    This is useful when w_rej suppresses high-frequency details and the fused
    image risks drifting in low-frequency luminance.
    """
    vi_y = rgb_to_y(vi)
    fu_y = rgb_to_y(fu)
    pad = kernel_size // 2
    vi_l = F.avg_pool2d(vi_y, kernel_size=kernel_size, stride=1, padding=pad)
    ir_l = F.avg_pool2d(ir, kernel_size=kernel_size, stride=1, padding=pad)
    fu_l = F.avg_pool2d(fu_y, kernel_size=kernel_size, stride=1, padding=pad)
    target_l = torch.max(vi_l, ir_l).detach()
    return F.l1_loss(fu_l, target_l)


def cara_full_regularization(
    maps: Dict[str, torch.Tensor],
    lambda_reject: float = 0.10,
    lambda_common: float = 0.05,
    lambda_smooth: float = 0.01,
    lambda_sparse: float = 0.01,
    residual_weight: float = 0.35,
    semantic_protect: float = 0.35,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Regularize Full CARA evidence allocation with weak pseudo-targets.

    Principles:
      - w_rej should increase mainly in conflict + unreliability regions:
            T_rej = D_conf * (1 - C_vis*C_ir) + lambda*N_res*max(1-C_vis,1-C_ir)
        then semantic/structure prior reduces over-rejection in salient regions.
      - w_com should increase when both modalities are reliable, non-conflicting,
        and residual degradation cues are low.
      - maps should be spatially smooth and w_rej should be sparse.
    """
    total = None
    log: Dict[str, torch.Tensor] = {}
    scale_count = 0

    semantic = maps.get("semantic_prior", None)
    residual = maps.get("residual_cue", None)
    for s in ["s1", "s2", "s3"]:
        w_rej = maps.get(f"{s}_w_rej")
        w_com = maps.get(f"{s}_w_com")
        conflict = maps.get(f"{s}_conflict")
        vrel = maps.get(f"{s}_vis_rel")
        irel = maps.get(f"{s}_ir_rel")
        if w_rej is None or w_com is None or conflict is None or vrel is None or irel is None:
            continue

        if semantic is None:
            sem = torch.zeros_like(w_rej)
        else:
            sem = F.interpolate(semantic, size=w_rej.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
        if residual is None:
            res = torch.zeros_like(w_rej)
        else:
            res = F.interpolate(residual, size=w_rej.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)

        risk = (1.0 - vrel * irel).clamp(0, 1)
        max_unrel = torch.maximum(1.0 - vrel, 1.0 - irel).clamp(0, 1)
        target_rej = (conflict * risk + float(residual_weight) * res * max_unrel).clamp(0, 1)
        target_rej = (target_rej * (1.0 - float(semantic_protect) * sem)).detach().clamp(0, 1)

        target_com = (torch.minimum(vrel, irel) * (1.0 - conflict) * (1.0 - 0.5 * res)).detach().clamp(0, 1)

        loss_rej = F.l1_loss(w_rej, target_rej)
        loss_com = F.l1_loss(w_com, target_com)
        loss_smooth = _tv_loss(w_rej) + _tv_loss(w_com) + _tv_loss(conflict)
        loss_sparse = w_rej.mean()

        loss = (
            float(lambda_reject) * loss_rej
            + float(lambda_common) * loss_com
            + float(lambda_smooth) * loss_smooth
            + float(lambda_sparse) * loss_sparse
        )
        total = loss if total is None else total + loss

        log[f"{s}_rej"] = loss_rej.detach()
        log[f"{s}_com"] = loss_com.detach()
        log[f"{s}_smooth"] = loss_smooth.detach()
        log[f"{s}_sparse"] = loss_sparse.detach()
        scale_count += 1

    if total is None:
        any_tensor = next(iter(maps.values()))
        total = any_tensor.new_tensor(0.0)
    else:
        total = total / max(scale_count, 1)
    log["cara_reg"] = total.detach()
    return total, log
