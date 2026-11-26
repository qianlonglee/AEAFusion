# ==============================================================
# eval_denoif_metrics.py
# - Replicate Deno-IF style metrics: SSIM / PSNR / FSIM / CC / BRISQUE
# - Full-reference metrics are computed against CLEAN source images.
# - Suggested aggregation for fusion:
#       score = 0.5 * metric(Fused_Y, VIS_clean_Y) + 0.5 * metric(Fused_Y, IR_clean)
# - Print per-image metrics and final mean±std
# ==============================================================

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_HAS_PIQ = False
try:
    import piq  # pip install piq
    _HAS_PIQ = True
except Exception:
    piq = None

# -----------------------
# User-editable paths
# -----------------------
FUSED_DIR = "/home/msia/Leeql/LYTFusion/Code_CARA/fusion-stagev2/runs_fusion_cara_w/full_108/results_LLVIP_100"
VIS_CLEAN_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/LLVIP200/VIS_HQ"
IR_CLEAN_DIR  = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/LLVIP200/Infrared_gt"

# FUSED_DIR = "/home/msia/Leeql/LYTFusion/Code_CARA/fusion-stagev2/runs_fusion_cara_w/full_108/results_MSRS_100"
# VIS_CLEAN_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/MSRS/VIS_HQ"
# IR_CLEAN_DIR  = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light/MSRS/Infrared_gt"

# FUSED_DIR = "/home/msia/Leeql/duibi/CrossFuse-main/1.27/M3FD"
# VIS_CLEAN_DIR = "/home/msia/Pon/DataSet/M3FD/test/vi"
# IR_CLEAN_DIR  = "/home/msia/Pon/DataSet/M3FD/test/ir"

# FUSED_DIR = "/home/msia/Leeql/LYTFusion/Code_CARA/fusion-stagev2/runs_fusion_cara_w/full_108_Exposure/runs_Over_exposure_freeze_ir/results_Exposure_freeze_ir_100"
# VIS_CLEAN_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Exposure/Visible_HQ"
# IR_CLEAN_DIR  = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Exposure/Infrared_HQ"

# FUSED_DIR = "/home/msia/Kevin/CPMFusion-master/results/vis_Exposure/Fusion/registrationfusion"
# VIS_CLEAN_DIR = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Exposure/Visible_HQ"
# IR_CLEAN_DIR  = "/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Exposure/Infrared_HQ"

FUSED_IS_GRAY = False

W_VIS = 0.5
W_IR  = 0.5

PRINT_EACH_IMAGE = True   # 是否打印每张图像指标
PRINT_DETAIL = True       # 是否打印 vis/ir 分别得分
SORT_BY_NAME = True       # 是否按文件名排序打印

tf = transforms.ToTensor()


def _norm01(t: torch.Tensor) -> torch.Tensor:
    t = t.float()
    mx = float(t.max())
    if mx > 1.5:
        denom = 65535.0 if mx > 300 else 255.0
        t = t / denom
    return t.clamp(0.0, 1.0)


def rgb_to_y(rgb: torch.Tensor) -> torch.Tensor:
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _match_hw(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    Ha, Wa = a.shape[-2:]
    Hb, Wb = b.shape[-2:]
    H = min(Ha, Hb)
    W = min(Wa, Wb)

    def crop(x):
        Hx, Wx = x.shape[-2:]
        top = max((Hx - H) // 2, 0)
        left = max((Wx - W) // 2, 0)
        return x[..., top:top + H, left:left + W]

    return crop(a), crop(b)


def _psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x, y = _match_hw(x, y)
    mse = torch.mean((x - y) ** 2).clamp_min(eps)
    return float(10.0 * torch.log10(1.0 / mse))


def _ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = _match_hw(x, y)
    if _HAS_PIQ:
        return float(piq.ssim(x, y, data_range=1.0))
    # fallback approx
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = ((x - mu_x) ** 2).mean()
    var_y = ((y - mu_y) ** 2).mean()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim)


def _fsim(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = _match_hw(x, y)
    if not _HAS_PIQ:
        return float("nan")
    return float(piq.fsim(x, y, data_range=1.0, chromatic=False))


def _cc(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x, y = _match_hw(x, y)
    x = x.flatten()
    y = y.flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x**2).sum() * (y**2).sum()).clamp_min(eps)
    return float((x * y).sum() / denom)


def _brisque(rgb01: torch.Tensor) -> float:
    if not _HAS_PIQ:
        return float("nan")
    return float(piq.brisque(rgb01, data_range=1.0))


def _list_files(folder: str) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts]
    return sorted(files) if SORT_BY_NAME else files


def _stem_map(folder: str) -> Dict[str, Path]:
    return {p.stem: p for p in _list_files(folder)}


def _load_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _norm01(tf(img)).unsqueeze(0)


def _load_gray(path: Path) -> torch.Tensor:
    img = Image.open(path)
    if img.mode not in ["L", "I;16", "I"]:
        img = img.convert("L")
    t = _norm01(tf(img)).unsqueeze(0)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def _fmt(x: float, ndigits: int = 4) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:.{ndigits}f}"


def _mean_std(x: List[float]) -> Tuple[float, float]:
    x = np.array(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=0))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    if not _HAS_PIQ:
        print("[WARN] piq is NOT installed -> FSIM/BRISQUE (and standard SSIM) will be missing/approx.")
        print("       Recommended: pip install piq")

    fused_map = _stem_map(FUSED_DIR)
    vis_map = _stem_map(VIS_CLEAN_DIR)
    ir_map  = _stem_map(IR_CLEAN_DIR)

    keys = sorted(set(fused_map) & set(vis_map) & set(ir_map))
    if len(keys) == 0:
        raise RuntimeError("No matched stems across FUSED_DIR / VIS_CLEAN_DIR / IR_CLEAN_DIR.")
    print(f"[Matched pairs] {len(keys)}")

    vals = {
        "PSNR": [], "SSIM": [], "FSIM": [], "CC": [], "BRISQUE": []
    }

    with torch.no_grad():
        for idx, k in enumerate(keys, 1):
            fused_rgb = _load_gray(fused_map[k]).to(device) if FUSED_IS_GRAY else _load_rgb(fused_map[k]).to(device)
            vis_rgb   = _load_rgb(vis_map[k]).to(device)
            ir_gray   = _load_gray(ir_map[k]).to(device)

            fused_y = fused_rgb if FUSED_IS_GRAY else rgb_to_y(fused_rgb)
            vis_y   = rgb_to_y(vis_rgb)

            # per-source metrics
            psnr_vis = _psnr(fused_y, vis_y)
            psnr_ir  = _psnr(fused_y, ir_gray)
            ssim_vis = _ssim(fused_y, vis_y)
            ssim_ir  = _ssim(fused_y, ir_gray)
            fsim_vis = _fsim(fused_y, vis_y)
            fsim_ir  = _fsim(fused_y, ir_gray)
            cc_vis   = _cc(fused_y, vis_y)
            cc_ir    = _cc(fused_y, ir_gray)

            # weighted fusion score
            psnr = W_VIS * psnr_vis + W_IR * psnr_ir
            ssim = W_VIS * ssim_vis + W_IR * ssim_ir
            fsim = W_VIS * fsim_vis + W_IR * fsim_ir
            cc   = W_VIS * cc_vis   + W_IR * cc_ir

            fused_for_brisque = fused_y.repeat(1, 3, 1, 1) if FUSED_IS_GRAY else fused_rgb
            brisque = _brisque(fused_for_brisque)

            vals["PSNR"].append(psnr)
            vals["SSIM"].append(ssim)
            vals["FSIM"].append(fsim)
            vals["CC"].append(cc)
            vals["BRISQUE"].append(brisque)

            if PRINT_EACH_IMAGE:
                print(f"\n[{idx:04d}/{len(keys):04d}] {k}")
                if PRINT_DETAIL:
                    print(
                        f"  PSNR    : vis={_fmt(psnr_vis)} | ir={_fmt(psnr_ir)} | fused={_fmt(psnr)}"
                    )
                    print(
                        f"  SSIM    : vis={_fmt(ssim_vis)} | ir={_fmt(ssim_ir)} | fused={_fmt(ssim)}"
                    )
                    print(
                        f"  FSIM    : vis={_fmt(fsim_vis)} | ir={_fmt(fsim_ir)} | fused={_fmt(fsim)}"
                    )
                    print(
                        f"  CC      : vis={_fmt(cc_vis)} | ir={_fmt(cc_ir)} | fused={_fmt(cc)}"
                    )
                else:
                    print(
                        f"  PSNR={_fmt(psnr)} | SSIM={_fmt(ssim)} | "
                        f"FSIM={_fmt(fsim)} | CC={_fmt(cc)} | BRISQUE={_fmt(brisque)}"
                    )

                print(f"  BRISQUE : {_fmt(brisque)}")

    print("\n===== Deno-IF style metrics (mean ± std) =====")
    for k in ["SSIM", "PSNR", "FSIM", "CC", "BRISQUE"]:
        m, s = _mean_std(vals[k])
        arrow = "↑" if k != "BRISQUE" else "↓"
        print(f"{k}{arrow}: {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()