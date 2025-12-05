import numpy as np
from forward_euler import forward_euler
from transient_reference_euler import transient_reference
from generate_reference import transient_ref_simplesolve
from save_params_npz import load_params_npz
from eval_u import constant_input, actual_drug_input
from visualize_combined import EulerVis
from SimpleSolver import SimpleSolver
from eval_f import eval_f
from trapezoidal import trapezoidal
from trapezoidal_adaptive import trapezoidal_adapt



# -----------------------------
# Trapezoidal comparison
# -----------------------------

# Use default parameters to simulate one grid square
evalf = eval_f
evalu = constant_input(0.0)

# Load in p for single grid with default parameters
p = load_params_npz("scenarios/sim_one_grid.npz")

# --- grid setup ---
rows, cols = 1, 1
p.rows = rows
p.cols = cols
n_cells = rows * cols

# Initial conditions
C0 = 10
T0 = 0
A0 = 0
x0 = np.zeros((n_cells * 3, 1))
x0[0::3, 0] = C0   # set C for each cell
x0[1::3, 0] = T0  # T cells
x0[2::3, 0] = A0  # Drug

# Total time
t_start = 0
t_stop = 10
n_min = 3
n_max = 5

# Load reference solution from test_reference.npz
ref = np.load("test_reference.npz")

ref_solution = ref["ReferenceSolution"]
ref_time     = ref["ReferenceTime"]
ref_error    = ref["ReferenceConfidence"]
ref_n        = ref["ReferenceExponent"]

###############################################################################################

# Test the same dt values for Euler:
dt_list = [1e-3, 1e-2, 1e-1, 7.64e-1, 1.0]

for dt in dt_list:
    print(f"\n--- Comparing dt={dt} ---")

    # Euler
    print("Running Forward Euler...")
    num_iter = int(t_stop / dt)
    Xe, te = SimpleSolver(evalf, x0, p, evalu, num_iter, dt, visualize=False)

    # Trapezoidal
    print("Running trapezoidal...")
    Xt, tt = trapezoidal(evalf, x0.ravel(), p, evalu, t_start, t_stop, dt)

    # Adaptive Trapezoidal
    print("Running trapezoidal adapt...")
    Xta, tta = trapezoidal_adapt(evalf, x0, p, evalu, t_start, t_stop, dt, tol_f=0.4, dt_min=dt)

    # Visualize both
    print("Euler plot:")
    EulerVis(te, Xe, grid_size=(rows, cols), cell_index=1)

    print("Trapezoidal plot:")
    EulerVis(tt, Xt, grid_size=(rows, cols), cell_index=1)

    print("Trapezoidal adaptive plot:")
    EulerVis(tta, Xta, grid_size=(rows, cols), cell_index=1)

    # Error vs reference
    euler_err = np.max(np.abs(Xe[:, -1] - ref_solution))
    trap_err  = np.max(np.abs(Xt[:, -1] - ref_solution))
    trap_adapt_err = np.max(np.abs(Xta[:, -1] - ref_solution))

    print("Euler error:", euler_err)
    print("Trapezoidal error:", trap_err)
    print("Trapezoidal adaptive error:", trap_adapt_err)
