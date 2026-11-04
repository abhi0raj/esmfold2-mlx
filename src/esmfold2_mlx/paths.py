"""Where esmfold2-mlx keeps things that are not code."""

import os
from pathlib import Path


def cache_dir() -> Path:
    """Converted ESMC weights, server sockets and downloaded reference structures.

    Defaults to ~/.cache/esmfold2-mlx; set ESMFOLD2_MLX_CACHE to move it (the converted 8-bit
    ESMC-6B weights are 6.7 GB).
    """
    d = Path(os.environ.get("ESMFOLD2_MLX_CACHE", Path.home() / ".cache" / "esmfold2-mlx"))
    d.mkdir(parents=True, exist_ok=True)
    return d
