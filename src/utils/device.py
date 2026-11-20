"""Device selection, mixed precision and reproducibility helpers."""
from __future__ import annotations

import contextlib
import os
import random

import numpy as np


def get_device(prefer: str | None = None):
    """CUDA if available (or `prefer`), else CPU."""
    import torch

    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device):
    """fp16 autocast on CUDA, no-op on CPU."""
    import torch

    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def free_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def is_oom(err: BaseException) -> bool:
    import torch

    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def gpu_label(name: str, total_bytes: int) -> str:
    """Marketing-style name: 'NVIDIA GeForce RTX 3050 Ti Laptop GPU', 3.7 GiB -> 'NVIDIA RTX 3050 Ti
    Laptop (4 GB)'. The driver reports usable memory in GiB, which understates the card's size."""
    n = " ".join(w for w in name.replace("GeForce", "").split() if w != "GPU")
    if not n.startswith("NVIDIA") and "RTX" in n:
        n = "NVIDIA " + n
    return f"{n} ({round(total_bytes / 1e9)} GB)"


def describe_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"GPU: {gpu_label(p.name, p.total_memory)}, torch {torch.__version__}"
        return f"cpu, torch {torch.__version__}"
    except ImportError:
        return "torch not installed"
