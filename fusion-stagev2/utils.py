import os
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fully deterministic seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm01(t: torch.Tensor) -> torch.Tensor:
    """Robustly normalize a ToTensor() output to [0,1].

    Handles 8-bit inputs (0~1 already) and 16-bit-like inputs (0~65535).
    """
    t = t.float()
    mx = float(t.max()) if t.numel() > 0 else 1.0
    if mx > 1.5:
        denom = 65535.0 if mx > 300 else 255.0
        t = t / denom
    return t.clamp(0.0, 1.0)


def load_state_dict_safely(model: torch.nn.Module, ckpt_path: str, key: Optional[str] = None):
    """Load checkpoint saved as raw state_dict or dict with 'state_dict'.

    Also supports dict with custom key (e.g. {'fusion': state_dict}).
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    if key is not None:
        if not isinstance(state, dict) or key not in state:
            raise KeyError(f"Key '{key}' not found in checkpoint: {ckpt_path}")
        state = state[key]

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"[load_state_dict_safely] missing={len(missing)} unexpected={len(unexpected)}")
    print(f"Loaded ckpt: {ckpt_path}")
