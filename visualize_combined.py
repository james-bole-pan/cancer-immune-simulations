import matplotlib.pyplot as plt
import numpy as np
import os

def DoubleVis(
    t_simple, X_simple,
    t_newton, X_newton,
    grid_size=(3,3),
    cell_index=1,
    show_together=True,
    save_path=None,
    w=None,                   # Step size used in SimpleSolver
    newton_converged=None,    # Boolean indicating Newton convergence
    newton_iterations=None    # Number of Newton iterations
):
    """
    Compare time dynamics (SimpleSolver) and steady-state progression (Newton)
    for a specified grid cell, and optionally save the figure to file.

    Parameters
    ----------
    t_simple : array
        Time vector from SimpleSolver.
    X_simple : 2D array
        States from SimpleSolver (num_states x time_steps).
    t_newton : array
        Iteration indices from Newton solver.
    X_newton : 2D array
        States from Newton solver (num_states x iterations).
    grid_size : tuple, optional
        Spatial grid size (rows, cols). Default (3,3).
    cell_index : int, optional
        Index (1-based) of which grid cell to visualize.
    show_together : bool, optional
        If True, show both plots on one figure; if False, separate windows.
    save_path : str, optional
        Full file path to save the figure (e.g. "results/Grid1_comparison.png").
    w : float, optional
        Step size used in SimpleSolver.
    newton_converged : bool, optional
        Whether the Newton solver converged.
    newton_iterations : int, optional
        Number of iterations used by the Newton solver.
    """

    num_cells = grid_size[0] * grid_size[1]
    num_states_total = X_simple.shape[0]
    num_states_per_cell = num_states_total // num_cells

    if cell_index < 1 or cell_index > num_cells:
        raise ValueError(f"cell_index must be between 1 and {num_cells}")

    # Indices for selected cell
    idx_start = (cell_index - 1) * num_states_per_cell
    idx = np.arange(idx_start, idx_start + num_states_per_cell)
    labels = ["Cancer cells", "Immune cells", "Drug conc"]

    # --- Create figure(s) ---
    if show_together:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Grid Cell {cell_index} Dynamics and Steady-State Convergence",
                     fontsize=18, weight='bold')
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig2, ax2 = plt.subplots(figsize=(10, 6))

    # --- Plot 1: SimpleSolver Time Dynamics ---
    for i in range(num_states_per_cell):
        ax1.plot(t_simple, X_simple[idx[i], :], linewidth=2, label=labels[i])
    ax1.set_title("SimpleSolver Time Dynamics", fontsize=15)
    ax1.set_xlabel("Time", fontsize=13)
    ax1.set_ylabel("State Value", fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=11)

    # Annotate the step size (w)
    if w is not None:
        ax1.text(
            0.05, 1.05, f"Step size (w) = {w:g}",
            transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8)
        )

    # --- Plot 2: Newton Steady-State Progression ---
    for i in range(num_states_per_cell):
        ax2.plot(t_newton, X_newton[idx[i], :], '.-', linewidth=2, markersize=8, label=labels[i])
    ax2.set_title("Newton Steady-State Progression", fontsize=15)
    ax2.set_xlabel("Iteration Index", fontsize=13)
    ax2.set_ylabel("State Value", fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=11)

    # Annotate Newton convergence info
    if newton_converged is not None or newton_iterations is not None:
        status = "Converged" if newton_converged else "Did NOT converge"
        it_text = f"Iterations: {newton_iterations}" if newton_iterations is not None else ""
        ax2.text(
            0.05, 1.10, f"{status}\n{it_text}",
            transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8)
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Save figure if requested ---
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()
