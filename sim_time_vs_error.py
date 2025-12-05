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
# Load reference solution 
# ------------------------------------------------------------
ref = np.load("test_reference.npz")
ref_solution = ref["ReferenceSolution"]
ref_time     = ref["ReferenceTime"]
ref_conf     = ref["ReferenceConfidence"]


# ------------------------------------------------------------
# Simulation setup
# ------------------------------------------------------------
p = load_params_npz("scenarios/sim_one_grid.npz")

rows = cols = 1
p.rows = rows
p.cols = cols

# initial condition
C0, T0, A0 = 10, 0, 0
x0 = np.array([C0, T0, A0], dtype=float).reshape(-1)

evalf = eval_f
evalu = constant_input(0)
t_start = 0
t_stop = 10

# ------------------------------------------------------------
# Range of timesteps to test
# ------------------------------------------------------------
dt_list = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

euler_times = []
trap_times  = []
trap_adapt_times = []

euler_errors = []
trap_errors  = []
trap_adapt_errors = []

# ------------------------------------------------------------
# Main experiment loop
# ------------------------------------------------------------
for dt in dt_list:
    print(f"\n=== Testing dt = {dt} ===")

    # ----------------------------------------
    # Forward Euler
    # ----------------------------------------
    start = time.time()
    num_iter = int((t_stop - t_start) / dt)

    print("Running Forward Euler...")
    Xe, te = SimpleSolver(evalf, x0.copy(), p, evalu, num_iter, dt, visualize=False)
    euler_runtime = time.time() - start

    euler_error = np.max(np.abs(Xe[:, -1] - ref_solution))

    euler_times.append(euler_runtime)
    euler_errors.append(euler_error)

    print(f"Euler: runtime={euler_runtime:.4f}, error={euler_error:.3e}")


    # ----------------------------------------
    # Trapezoidal
    # ----------------------------------------
    start = time.time()

    print("Running Trapezoidal...")
    Xt, tt = trapezoidal(evalf, x0.copy(), p, evalu, t_start, t_stop, dt)

    trap_runtime = time.time() - start

    trap_error = np.max(np.abs(Xt[:, -1] - ref_solution))

    trap_times.append(trap_runtime)
    trap_errors.append(trap_error)

    print(f"Trapezoidal: runtime={trap_runtime:.4f}, error={trap_error:.3e}")

    # ----------------------------------------
    # Adaptive Trapezoidal
    # ----------------------------------------
    start = time.time()

    print("Running Adaptive Trapezoidal...")
    Xta, tta = trapezoidal_adapt(evalf, x0.copy(), p, evalu, t_start, t_stop, dt, tol_f=2, dt_min=dt*0.1)

    trap_adapt_runtime = time.time() - start

    trap_adapt_error = np.max(np.abs(Xta[:, -1] - ref_solution))

    trap_adapt_times.append(trap_adapt_runtime)
    trap_adapt_errors.append(trap_adapt_error)

    print(f"Adaptive Trapezoidal: runtime={trap_adapt_runtime:.4f}, error={trap_adapt_error:.3e}")


# ------------------------------------------------------------
# Plot Error vs Runtime 
# ------------------------------------------------------------
plt.figure(figsize=(8,6))

plt.loglog(euler_times, euler_errors, 'o-', label='Forward Euler')
plt.loglog(trap_times, trap_errors, 's-', label='Trapezoidal')
plt.loglog(trap_adapt_times, trap_adapt_errors, '^-', label='Adaptive Trapezoidal')

plt.xlabel("Runtime (seconds)")
plt.ylabel("Terminal Error")
plt.title("Error vs Runtime")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()
