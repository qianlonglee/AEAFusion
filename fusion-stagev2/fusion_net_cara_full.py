"""Full CARA-v2 fusion network for Stage-2 low-light IR-VIS fusion.

CARA = Conflict-aware Abstention Reliability Allocation.

This version keeps the two-stage paradigm:
  Stage-1: restoration/confidence network outputs (vis_hat, ir_hat, C_vis, C_ir)
  Stage-2: this CARA fusion network allocates reliable evidence.

Compared with the earlier CARA version, this implementation makes three
changes that better fit low-light image fusion:
  1) w_rej is a *soft detail-suppression factor*, not a hard rejection switch.
     F_f = F_base + (1 - beta * w_rej) * F_detail.
  2) w_rej is driven by cross-modal conflict and residual unreliability:
     T_rej ~ D_conf * (1 - C_vis * C_ir), optionally assisted by a residual
     degradation cue.
  3) a lightweight residual-degradation cue is added for Stage-2 gating only.
     It does not turn the task into compound IR restoration; it only helps the
     fusion gate avoid injecting unreliable high-frequency details.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Color conversion utilities
# -----------------------------------------------------------------------------


def RGB2YCrCb(rgb_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert RGB in [0, 1] to luminance/chroma channels.

    Returns:
        Y:  (B,1,H,W)
        Cb: (B,1,H,W), detached
        Cr: (B,1,H,W), detached
    """
    if rgb_image.dim() != 4 or rgb_image.size(1) != 3:
        raise ValueError(f"RGB2YCrCb expects (B,3,H,W), got {tuple(rgb_image.shape)}")

    r = rgb_image[:, 0:1]
    g = rgb_image[:, 1:2]
    b = rgb_image[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5
    cb = (b - y) * 0.564 + 0.5
    return y.clamp(0.0, 1.0), cb.clamp(0.0, 1.0).detach(), cr.clamp(0.0, 1.0).detach()


def YCbCr2RGB(Y: torch.Tensor, Cb: torch.Tensor, Cr: torch.Tensor) -> torch.Tensor:
    """Convert Y, Cb, Cr in [0,1] back to RGB in [0,1]."""
    if not (Y.shape[-2:] == Cb.shape[-2:] == Cr.shape[-2:]):
        raise ValueError("Y/Cb/Cr spatial sizes must match")
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    bsz, _, h, w = ycrcb.shape

    im_flat = ycrcb.permute(0, 2, 3, 1).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0],
         [1.403, -0.714, 0.0],
         [0.0, -0.344, 1.773]],
        device=Y.device,
        dtype=Y.dtype,
    )
    bias = torch.tensor([0.0, -0.5, -0.5], device=Y.device, dtype=Y.dtype)
    out = (im_flat + bias).mm(mat)
    out = out.reshape(bsz, h, w, 3).permute(0, 3, 1, 2)
    return out.clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# Basic blocks
# -----------------------------------------------------------------------------


def _gn(ch: int) -> nn.GroupNorm:
    groups = 8
    while groups > 1 and ch % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, ch)


class DSConv(nn.Module):
    """Depthwise-separable convolution block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1, act: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class LiteEncoder(nn.Module):
    """Tiny 3-scale encoder."""

    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1, bias=False),
            _gn(base),
            nn.SiLU(inplace=True),
        )
        self.s1 = nn.Sequential(DSConv(base, base), DSConv(base, base))
        self.s2 = nn.Sequential(DSConv(base, base * 2, stride=2), DSConv(base * 2, base * 2))
        self.s3 = nn.Sequential(DSConv(base * 2, base * 4, stride=2), DSConv(base * 4, base * 4, dilation=2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        return f1, f2, f3


class LiteDecoder(nn.Module):
    """Small additive decoder."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 4, base * 2, kernel_size=1, bias=False),
            _gn(base * 2),
            nn.SiLU(inplace=True),
            DSConv(base * 2, base * 2),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 2, base, kernel_size=1, bias=False),
            _gn(base),
            nn.SiLU(inplace=True),
            DSConv(base, base),
        )
        self.head = nn.Conv2d(base, 1, kernel_size=3, padding=1)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        x = self.up2(f3) + f2
        x = self.up1(x) + f1
        return self.head(x)


# -----------------------------------------------------------------------------
# Priors and explicit cues
# -----------------------------------------------------------------------------


def _grad_xy(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")
    c = x.size(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return gx, gy


def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    gx, gy = _grad_xy(x)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _norm01_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.amin(dim=(2, 3), keepdim=True)
    mx = x.amax(dim=(2, 3), keepdim=True)
    return ((x - mn) / (mx - mn + eps)).clamp(0.0, 1.0)


class WeakSemanticReliabilityPrior(nn.Module):
    """Training-free structure/weak semantic prior.

    It does not use CLIP/SAM/VLM. It only estimates potentially important
    structure/target regions from restored VIS/IR images, preventing w_rej from
    over-suppressing useful details.
    """

    def __init__(self, w_vis_edge: float = 0.35, w_ir_sal: float = 0.45, w_ir_edge: float = 0.20):
        super().__init__()
        self.w_vis_edge = float(w_vis_edge)
        self.w_ir_sal = float(w_ir_sal)
        self.w_ir_edge = float(w_ir_edge)

    def forward(self, vis_y: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        vis_edge = _norm01_per_sample(_sobel_magnitude(vis_y))
        ir_edge = _norm01_per_sample(_sobel_magnitude(ir))
        local_mean = F.avg_pool2d(ir, kernel_size=15, stride=1, padding=7)
        ir_sal = _norm01_per_sample(F.relu(ir - local_mean))
        s = self.w_vis_edge * vis_edge + self.w_ir_sal * ir_sal + self.w_ir_edge * ir_edge
        return s.clamp(0.0, 1.0)


class ResidualDegradationCue(nn.Module):
    """Lightweight residual-degradation cue for Stage-2 gating only.

    This module is not an IR denoiser and does not claim to solve compound IR
    degradation. It gives CARA extra cues about unreliable high-frequency or
    structure-inconsistent details that should not be injected into fusion.
    """

    def __init__(self, w_hf: float = 0.40, w_stripe: float = 0.25, w_struct: float = 0.35):
        super().__init__()
        self.w_hf = float(w_hf)
        self.w_stripe = float(w_stripe)
        self.w_struct = float(w_struct)

    def forward(self, vis_y: torch.Tensor, ir: torch.Tensor) -> Dict[str, torch.Tensor]:
        # IR high-frequency residual-like cue.
        ir_smooth = F.avg_pool2d(ir, kernel_size=7, stride=1, padding=3)
        ir_hf = _norm01_per_sample((ir - ir_smooth).abs())

        # Directional response. Stripe/FPN-like artifacts tend to create coherent
        # directional gradients. This is a cue for the fusion gate, not a label.
        gx, gy = _grad_xy(ir)
        dir_resp = _norm01_per_sample(torch.maximum(gx.abs(), gy.abs()))
        dir_local = F.avg_pool2d(dir_resp, kernel_size=9, stride=1, padding=4)
        ir_stripe = _norm01_per_sample((dir_resp - dir_local).abs() + 0.5 * dir_resp)

        # VIS/IR structural inconsistency.
        vis_grad = _norm01_per_sample(_sobel_magnitude(vis_y))
        ir_grad = _norm01_per_sample(_sobel_magnitude(ir))
        struct_conflict = (vis_grad - ir_grad).abs().clamp(0.0, 1.0)

        residual = (self.w_hf * ir_hf + self.w_stripe * ir_stripe + self.w_struct * struct_conflict).clamp(0, 1)
        return {
            "residual_cue": residual,
            "ir_hf": ir_hf,
            "ir_stripe": ir_stripe,
            "struct_conflict": struct_conflict,
        }


class ConflictEstimator(nn.Module):
    """Estimate cross-modal conflict at one feature scale."""

    def __init__(self, channels: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1, bias=False),
            _gn(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        vis_feat: torch.Tensor,
        ir_feat: torch.Tensor,
        vis_conf: torch.Tensor,
        ir_conf: torch.Tensor,
        semantic_prior: torch.Tensor,
        residual_cue: torch.Tensor,
    ) -> torch.Tensor:
        h, w = vis_feat.shape[-2:]
        vc = F.interpolate(vis_conf, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        ic = F.interpolate(ir_conf, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        sp = F.interpolate(semantic_prior, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        rc = F.interpolate(residual_cue, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        vf = F.normalize(vis_feat, dim=1, eps=1e-6)
        tf = F.normalize(ir_feat, dim=1, eps=1e-6)
        cos_dist = (1.0 - (vf * tf).sum(dim=1, keepdim=True)).mul(0.5).clamp(0.0, 1.0)
        diff_energy = _norm01_per_sample((vis_feat - ir_feat).abs().mean(dim=1, keepdim=True))
        conf_disagree = (vc - ic).abs()
        risk = (1.0 - vc * ic).clamp(0.0, 1.0)

        cues = torch.cat([cos_dist, diff_energy, conf_disagree, risk, sp, rc], dim=1)
        explicit = (
            0.30 * cos_dist
            + 0.20 * diff_energy
            + 0.20 * conf_disagree
            + 0.15 * risk
            + 0.15 * rc
        ).clamp(0, 1)
        learned = torch.sigmoid(self.net(cues))
        return (0.5 * explicit + 0.5 * learned).clamp(0.0, 1.0)


class FusionAwareReliabilityRefiner(nn.Module):
    """Refine residual-calibrated confidence into fusion-aware reliability.

    The original confidence remains an anchor. The module only learns a small
    logit-space correction from conflict, semantic, residual and feature cues.
    """

    def __init__(self, channels: int, hidden: int = 16, max_delta: float = 1.0):
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Conv2d(8, hidden, kernel_size=3, padding=1, bias=False),
            _gn(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),
        )

    @staticmethod
    def _logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        x = x.clamp(eps, 1.0 - eps)
        return torch.log(x / (1.0 - x))

    def forward(
        self,
        vis_feat: torch.Tensor,
        ir_feat: torch.Tensor,
        vis_conf: torch.Tensor,
        ir_conf: torch.Tensor,
        conflict: torch.Tensor,
        semantic_prior: torch.Tensor,
        residual_cue: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = vis_feat.shape[-2:]
        vc = F.interpolate(vis_conf, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        ic = F.interpolate(ir_conf, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        cf = F.interpolate(conflict, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        sp = F.interpolate(semantic_prior, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        rc = F.interpolate(residual_cue, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        ev = _norm01_per_sample(vis_feat.abs().mean(dim=1, keepdim=True))
        ei = _norm01_per_sample(ir_feat.abs().mean(dim=1, keepdim=True))
        e_diff = (ev - ei).abs()
        cues = torch.cat([vc, ic, cf, sp, rc, ev, ei, e_diff], dim=1)
        delta = torch.tanh(self.net(cues)) * self.max_delta
        vc_ref = torch.sigmoid(self._logit(vc) + delta[:, 0:1])
        ic_ref = torch.sigmoid(self._logit(ic) + delta[:, 1:2])
        return vc_ref.clamp(0.0, 1.0), ic_ref.clamp(0.0, 1.0)


class EvidenceAllocator(nn.Module):
    """Allocate four evidence states at one feature scale."""

    def __init__(self, channels: int, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(10, hidden, kernel_size=3, padding=1, bias=False),
            _gn(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=3, padding=1),
        )
        last = self.net[-1]
        nn.init.constant_(last.bias, 0.0)
        with torch.no_grad():
            # Avoid early all-reject behavior.
            last.bias[3] = -1.25

    def forward(
        self,
        vis_feat: torch.Tensor,
        ir_feat: torch.Tensor,
        vis_rel: torch.Tensor,
        ir_rel: torch.Tensor,
        conflict: torch.Tensor,
        semantic_prior: torch.Tensor,
        residual_cue: torch.Tensor,
    ) -> torch.Tensor:
        h, w = vis_feat.shape[-2:]
        vc = F.interpolate(vis_rel, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        ic = F.interpolate(ir_rel, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        cf = F.interpolate(conflict, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        sp = F.interpolate(semantic_prior, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        rc = F.interpolate(residual_cue, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        ev = _norm01_per_sample(vis_feat.abs().mean(dim=1, keepdim=True))
        ei = _norm01_per_sample(ir_feat.abs().mean(dim=1, keepdim=True))
        both_rel = torch.minimum(vc, ic)
        risk = (1.0 - vc * ic).clamp(0, 1)
        conf_diff = (vc - ic).abs()
        cues = torch.cat([vc, ic, cf, sp, rc, ev, ei, both_rel, risk, conf_diff], dim=1)

        logits = self.net(cues)

        # Interpretable allocation priors.
        eps = 1e-4
        prior_vis = torch.log(vc + eps) + torch.log(1.0 - cf * ic + eps)
        prior_ir = torch.log(ic + eps) + torch.log(1.0 - cf * vc + eps) + torch.log(1.0 - 0.35 * rc + eps)
        prior_com = torch.log(torch.minimum(vc, ic) + eps) + torch.log(1.0 - cf + eps) + torch.log(1.0 - 0.35 * rc + eps)

        # Reject unreliable details when there is conflict + residual unreliability.
        reject_score = (cf * risk + 0.35 * rc * torch.maximum(1.0 - vc, 1.0 - ic)).clamp(0, 1)
        # Weak semantic prior protects target/structure regions from over-rejection.
        reject_score = (reject_score * (1.0 - 0.35 * sp)).clamp(0, 1)
        prior_rej = torch.log(reject_score + eps)

        prior = torch.cat([prior_vis, prior_ir, prior_com, prior_rej], dim=1)
        return torch.softmax(logits + prior, dim=1)


class CARAGate(nn.Module):
    """Conflict-aware Abstention Reliability Allocation gate."""

    def __init__(self, channels: int, hidden: int = 24, reject_strength: float = 0.6):
        super().__init__()
        self.reject_strength = float(reject_strength)
        self.conflict = ConflictEstimator(channels, hidden=hidden)
        self.refiner = FusionAwareReliabilityRefiner(channels, hidden=hidden)
        self.alloc = EvidenceAllocator(channels, hidden=hidden)

        self.base = DSConv(channels * 2, channels)
        self.vis_detail = DSConv(channels, channels)
        self.ir_detail = DSConv(channels, channels)
        self.common_detail = DSConv(channels * 3, channels)
        self.out = DSConv(channels, channels)

    def forward(
        self,
        vis_feat: torch.Tensor,
        ir_feat: torch.Tensor,
        vis_conf: torch.Tensor,
        ir_conf: torch.Tensor,
        semantic_prior: torch.Tensor,
        residual_cue: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        d_conf = self.conflict(vis_feat, ir_feat, vis_conf, ir_conf, semantic_prior, residual_cue)
        vc_ref, ic_ref = self.refiner(vis_feat, ir_feat, vis_conf, ir_conf, d_conf, semantic_prior, residual_cue)
        weights = self.alloc(vis_feat, ir_feat, vc_ref, ic_ref, d_conf, semantic_prior, residual_cue)
        wv, wi, wc, wr = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3], weights[:, 3:4]

        f_base = self.base(torch.cat([vis_feat, ir_feat], dim=1))
        d_vis = self.vis_detail(vis_feat)
        d_ir = self.ir_detail(ir_feat)
        d_com = self.common_detail(torch.cat([vis_feat, ir_feat, (vis_feat - ir_feat).abs()], dim=1))

        f_detail = wv * d_vis + wi * d_ir + wc * d_com
        # Soft detail suppression. w_rej does not reject the whole pixel/modalitiy;
        # it only reduces unreliable high-frequency detail injection.
        fused = f_base + (1.0 - self.reject_strength * wr) * f_detail
        fused = self.out(fused)
        maps = {
            "w_vis": wv,
            "w_ir": wi,
            "w_com": wc,
            "w_rej": wr,
            "conflict": d_conf,
            "vis_rel": vc_ref,
            "ir_rel": ic_ref,
        }
        return fused, maps


# -----------------------------------------------------------------------------
# Main network
# -----------------------------------------------------------------------------


class FusionNetCARAFull(nn.Module):
    """Stage-2 full CARA-v2 fusion network.

    Args:
        base_channels: width of the encoder/decoder.
        res_scale: residual scale for luminance output.
        output_base: "mean" uses 0.5*(Y_vis+IR) as conservative image base;
            "vis" keeps the old visible residual style.
        reject_strength: beta in F_f = F_base + (1 - beta*w_rej)*F_detail.
    """

    def __init__(
        self,
        base_channels: int = 32,
        res_scale: float = 0.45,
        output_base: str = "mean",
        reject_strength: float = 0.6,
    ):
        super().__init__()
        if output_base not in {"mean", "vis"}:
            raise ValueError("output_base must be 'mean' or 'vis'")
        self.base = int(base_channels)
        self.res_scale = float(res_scale)
        self.output_base = output_base
        self.reject_strength = float(reject_strength)

        self.vis_enc = LiteEncoder(in_ch=1, base=self.base)
        self.ir_enc = LiteEncoder(in_ch=1, base=self.base)
        self.semantic_prior = WeakSemanticReliabilityPrior()
        self.residual_cue = ResidualDegradationCue()

        self.gate1 = CARAGate(self.base, hidden=max(16, self.base // 2), reject_strength=self.reject_strength)
        self.gate2 = CARAGate(self.base * 2, hidden=max(16, self.base), reject_strength=self.reject_strength)
        self.gate3 = CARAGate(self.base * 4, hidden=max(16, self.base * 2), reject_strength=self.reject_strength)
        self.dec = LiteDecoder(base=self.base)

    def _default_conf(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        vis_conf: Optional[torch.Tensor] = None,
        ir_conf: Optional[torch.Tensor] = None,
        semantic_prior: Optional[torch.Tensor] = None,
        return_feats: bool = False,
        return_maps: bool = False,
    ):
        if rgb.dim() != 4 or rgb.size(1) != 3:
            raise ValueError(f"FusionNetCARAFull expects rgb (B,3,H,W), got {tuple(rgb.shape)}")
        if ir.dim() != 4:
            raise ValueError(f"FusionNetCARAFull expects ir BCHW, got {tuple(ir.shape)}")
        if ir.size(1) != 1:
            ir = ir[:, :1]

        vis_y, cb, cr = RGB2YCrCb(rgb)
        if ir.shape[-2:] != vis_y.shape[-2:]:
            ir = F.interpolate(ir, size=vis_y.shape[-2:], mode="bilinear", align_corners=False)
        if vis_conf is None:
            vis_conf = self._default_conf(vis_y)
        if ir_conf is None:
            ir_conf = self._default_conf(ir)
        if semantic_prior is None:
            semantic_prior = self.semantic_prior(vis_y, ir)
        residual_maps = self.residual_cue(vis_y, ir)
        residual_cue = residual_maps["residual_cue"]

        r1, r2, r3 = self.vis_enc(vis_y)
        t1, t2, t3 = self.ir_enc(ir)

        f1, m1 = self.gate1(r1, t1, vis_conf, ir_conf, semantic_prior, residual_cue)
        f2, m2 = self.gate2(r2, t2, vis_conf, ir_conf, semantic_prior, residual_cue)
        f3, m3 = self.gate3(r3, t3, vis_conf, ir_conf, semantic_prior, residual_cue)

        delta = torch.tanh(self.dec(f1, f2, f3)) * self.res_scale
        if self.output_base == "mean":
            y_base = 0.5 * (vis_y + ir)
        else:
            y_base = vis_y
        fused_y = (y_base + delta).clamp(0.0, 1.0)
        fused_rgb = YCbCr2RGB(fused_y, cb, cr)

        maps: Dict[str, torch.Tensor] = {
            "semantic_prior": semantic_prior,
            "residual_cue": residual_cue,
            "ir_hf": residual_maps["ir_hf"],
            "ir_stripe": residual_maps["ir_stripe"],
            "struct_conflict": residual_maps["struct_conflict"],
        }
        for prefix, mm in [("s1", m1), ("s2", m2), ("s3", m3)]:
            for k, v in mm.items():
                maps[f"{prefix}_{k}"] = v

        if return_feats:
            maps.update({"r1": r1, "r2": r2, "r3": r3, "t1": t1, "t2": t2, "t3": t3, "f1": f1, "f2": f2, "f3": f3})
        if return_maps or return_feats:
            return fused_y, fused_rgb, maps
        return fused_y, fused_rgb


# Backward-compatible short alias if you want to import FusionNetCARA.
FusionNetCARA = FusionNetCARAFull
