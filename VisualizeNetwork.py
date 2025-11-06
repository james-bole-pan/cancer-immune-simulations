import re
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from eval_f import Params

def visualizeNetwork(
    x: np.ndarray,                    # shape (N, 1)
    p: Params,                        # provides rows, cols
    title_prefix: str = "",
    figsize: Tuple[float, float] = (12.0, 4.0),
    cmap_C: str = "Reds",
    cmap_T: str = "Blues",
    cmap_A: str = "Greens",
    output_dir: str = "test_evalf_output_figures/",
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray, Path]:
    """
    Visualize spatial distributions of Cancer (C), T cells (T), and Drug (A)
    from a column vector x (N,1), save a single combined image, and return (fig, axes, save_path).
    """
    if not isinstance(p, Params):
        raise TypeError(f"'p' must be Params, got {type(p).__name__}")

    rows, cols = p.rows, p.cols
    expected_N = rows * cols * 3

    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"'x' must have shape (N, 1); got {x.shape}")
    if x.size != expected_N:
        raise ValueError(f"State length mismatch: expected N={expected_N}, got {x.size}")

    # reshape (N,1) -> (rows, cols, 3)
    x_rc3 = x.ravel(order="C").reshape(rows, cols, 3, order="C")
    C, T, A = x_rc3[:, :, 0], x_rc3[:, :, 1], x_rc3[:, :, 2]

    # plot (single combined figure)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = (
        (C, cmap_C, f"{title_prefix}Cancer (C)"),
        (T, cmap_T, f"{title_prefix}T cells (T)"),
        (A, cmap_A, f"{title_prefix}Drug (A)"),
    )
    for ax, (data, cmap, ttl) in zip(axes, panels):
        im = ax.imshow(np.asarray(data), cmap=cmap, origin="lower")
        ax.set_title(ttl, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    # save ONE image
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _sanitize(s: str) -> str:
        s = s.strip()
        if not s:
            return "viz"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    base = _sanitize(title_prefix) or "viz"
    save_path = outdir / f"{base}_CTA_grid.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes, save_path