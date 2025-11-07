import re
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from eval_f import Params
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

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
    visualize: bool = True,
    save: bool = True,
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

    if visualize:
        plt.show()

    if not save:
        return fig, axes, None

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

def create_network_evolution_gif(
    X_an: np.ndarray,
    p: Params,
    output_dir: str = "test_evalf_output_figures/",
    title_prefix: str = "evolution_",
    fps: int = 20,
    dpi: int = 150,
    save: bool = True,
    show: bool = False,
):
    """
    Create a smooth animated GIF showing spatial network evolution over iterations,
    with colorbars and a combined RGB overlay.

    Parameters
    ----------
    X_an : np.ndarray
        Array of shape (N, num_frames) from newtonNd.
    p : Params
        Contains rows and cols for reshaping.
    output_dir : str
        Output directory for the resulting GIF.
    title_prefix : str
        Prefix for saved GIF file.
    fps : int
        Frames per second.
    dpi : int
        Resolution for output GIF.
    save : bool
        Whether to save the GIF to file.
    show : bool
        Whether to display the animation interactively.
    """

    rows, cols = p.rows, p.cols
    n_frames = X_an.shape[1]
    N_expected = rows * cols * 3
    assert X_an.shape[0] == N_expected, f"Shape mismatch: expected {N_expected} per frame"

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    gif_path = outdir / f"{title_prefix.rstrip('_')}.gif"

    print(f"[INFO] Creating GIF with {n_frames} frames → {gif_path}")

    # --- colormaps ---
    cancer_cmap = LinearSegmentedColormap.from_list("cancer", ["white", "red", "darkred"])
    immune_cmap = LinearSegmentedColormap.from_list("immune", ["white", "blue", "darkblue"])
    drug_cmap   = LinearSegmentedColormap.from_list("drug",   ["white", "green", "darkgreen"])

    # --- reshape all frames for fast access ---
    all_frames = np.empty((n_frames, rows, cols, 3), dtype=float)
    for k in range(n_frames):
        all_frames[k] = X_an[:, k].reshape(rows, cols, 3)

    cmax = np.max(all_frames[..., 0])
    imax = np.max(all_frames[..., 1])
    dmax = np.max(all_frames[..., 2])

    # --- figure layout ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axC, axT, axA, axComb = axes

    # --- initial frame ---
    f0 = all_frames[0]
    imgC = axC.imshow(f0[:, :, 0], cmap=cancer_cmap, vmin=0, vmax=cmax, origin="lower")
    imgT = axT.imshow(f0[:, :, 1], cmap=immune_cmap, vmin=0, vmax=imax, origin="lower")
    imgA = axA.imshow(f0[:, :, 2], cmap=drug_cmap, vmin=0, vmax=dmax, origin="lower")

    # combined overlay
    combined = np.zeros_like(f0)
    if cmax > 0: combined[:, :, 0] = f0[:, :, 0] / cmax
    if imax > 0: combined[:, :, 2] = f0[:, :, 1] / imax
    imgComb = axComb.imshow(combined, interpolation="nearest", origin="lower")

    # --- colorbars ---
    cbarC = fig.colorbar(imgC, ax=axC, fraction=0.046, pad=0.04)
    cbarT = fig.colorbar(imgT, ax=axT, fraction=0.046, pad=0.04)
    cbarA = fig.colorbar(imgA, ax=axA, fraction=0.046, pad=0.04)
    cbarC.set_label("Cancer cell density")
    cbarT.set_label("T cell density")
    cbarA.set_label("Drug concentration")

    for ax, title in zip(
        axes,
        ["Cancer (C)", "T cells (T)", "Drug (A)", "Combined (R=C, B=T)"]
    ):
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    # --- dynamic time label ---
    time_text = fig.suptitle("Iteration 0")

    # --- update function ---
    def animate(k):
        frame = all_frames[k]
        imgC.set_data(frame[:, :, 0])
        imgT.set_data(frame[:, :, 1])
        imgA.set_data(frame[:, :, 2])

        comb = np.zeros_like(frame)
        if cmax > 0: comb[:, :, 0] = frame[:, :, 0] / cmax
        if imax > 0: comb[:, :, 2] = frame[:, :, 1] / imax
        imgComb.set_data(comb)

        time_text.set_text(f"Iteration {k}")
        return [imgC, imgT, imgA, imgComb, time_text]

    # --- animate ---
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=500, blit=False, repeat=False
    )

    if save:
        anim.save(gif_path, writer="pillow", fps=fps, dpi=dpi)
        print(f"[SUCCESS] GIF saved at {gif_path}")

    if show:
        plt.show()

    plt.close(fig)
    return gif_path if save else None