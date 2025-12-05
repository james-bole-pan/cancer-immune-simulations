import numpy as np
import time
import matplotlib.pyplot as plt
from SimpleSolver import SimpleSolver
from trapezoidal import trapezoidal
from trapezoidal_adaptive import trapezoidal_adapt
from eval_f import eval_f
from eval_u import constant_input
from save_params_npz import load_params_npz

# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------
grid_sizes = [1, 2, 4, 8]  
dt = 0.01                         # same dt for all fixed-step solvers
t_start = 0
t_stop = 10

# store runtimes
euler_times = []
trap_times = []
trap_adapt_times = []

for N in grid_sizes:
    print(f"\n====== Running grid size = {N} x {N} ======")

    # Load params
    p = load_params_npz("scenarios/sim_one_grid.npz")
    p.rows = N
    p.cols = N

    # build initial condition for all grid points
    C0, T0, A0 = 10, 0, 0
    x0_cell = np.array([C0, T0, A0], float)
    x0 = np.tile(x0_cell, N*N)

    evalf = eval_f
    evalu = constant_input(0)

    num_iter = int((t_stop - t_start) / dt)

    # ------------------------------------------------------------
    # Forward Euler
    # ------------------------------------------------------------
    start = time.time()
    Xe, te = SimpleSolver(evalf, x0.copy(), p, evalu, num_iter, dt, visualize=False)
    euler_runtime = time.time() - start
    euler_times.append(euler_runtime)

    print(f"Euler runtime: {euler_runtime:.4f} s")

    # ------------------------------------------------------------
    # Trapezoidal (fixed dt)
    # ------------------------------------------------------------
    start = time.time()
    Xt, tt = trapezoidal(evalf, x0.copy(), p, evalu, t_start, t_stop, dt)
    trap_runtime = time.time() - start
    trap_times.append(trap_runtime)

    print(f"Trapezoidal runtime: {trap_runtime:.4f} s")

    # ------------------------------------------------------------
    # Adaptive Trapezoidal
    # ------------------------------------------------------------
    start = time.time()
    Xta, tta = trapezoidal_adapt(
        evalf, x0.copy(), p, evalu,
        t_start, t_stop, dt,
        tol_f=2,
        dt_min=0.1 * dt,
        dt_max=5 * dt,
    )
    trap_adapt_runtime = time.time() - start
    trap_adapt_times.append(trap_adapt_runtime)

    print(f"Adaptive Trap runtime: {trap_adapt_runtime:.4f} s")


# ------------------------------------------------------------
# Plot runtime vs grid size
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.loglog(grid_sizes, euler_times, 'o-', label="Forward Euler")
plt.loglog(grid_sizes, trap_times, 's-', label="Trapezoidal")
plt.loglog(grid_sizes, trap_adapt_times, '^-', label="Adaptive Trapezoidal")

plt.xlabel("Grid Size (N × N)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Grid Size")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
