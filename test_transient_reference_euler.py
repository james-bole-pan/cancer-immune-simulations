import numpy as np
from forward_euler import forward_euler
from transient_reference_euler import transient_reference
from generate_reference import transient_ref_simplesolve
from save_params_npz import load_params_npz
from eval_u import constant_input, actual_drug_input
from visualize_combined import EulerVis
from SimpleSolver import SimpleSolver
from eval_f import eval_f

'''
# Test getting the simple solver to run and visualize
# Use logistic growth scenario parameters
evalf = eval_f
evalu = constant_input(0.0)
p = load_params_npz("scenarios/logistic_growth.npz")

# --- grid setup ---
rows, cols = 1, 1
p.rows = rows
p.cols = cols
n_cells = rows * cols

# Initial conditions and time span
C0 = 0.1
x0 = np.zeros((n_cells * 3, 1))
x0[0::3, 0] = C0   # set C for each cell
x0[1::3, 0] = 0.0  # T cells
x0[2::3, 0] = 0.0  # Drug

num_iter = 100
n_min = 1
n_max = 7
timestep = 0.1

#X, t = SimpleSolver(evalf, x0, p, evalu, num_iter, timestep, visualize=False)

#EulerVis(t, X, grid_size=(rows, cols), cell_index=1)
'''
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
t_stop = 10
n_min = 3
n_max = 5

# Now generate the reference solution with transient_reference
# w/ n=5, e-5 err
ref_sol, ref_time, ref_err = transient_ref_simplesolve(evalf, x0, p, evalu, t_stop, 
                                                       n_min, n_max, save_path="test_reference.npz")

# Manually test different dt until it becomes unstable
# Unstable at dt=1 when C0 = 10 and T0 = 0
# Expanding oscillations at dt=7.64e-1



# Start with n=3 (1e-3)
# e-4 error vs ref
print("\nTesting dt = 1e-3")
timestep = 1e-3
num_iter = int(t_stop / timestep)
X, t = SimpleSolver(evalf, x0, p, evalu, num_iter, timestep, visualize=False)
EulerVis(t, X, grid_size=(rows, cols), cell_index=1)
dt_error = abs(ref_sol - X[:, -1]).max()
print("Max error vs reference for timestep ", timestep, ": ", dt_error)

# e-3 error vs ref
print("\nTesting dt = 1e-2")
timestep = 1e-2
num_iter = int(t_stop / timestep)
X, t = SimpleSolver(evalf, x0, p, evalu, num_iter, timestep, visualize=False)
EulerVis(t, X, grid_size=(rows, cols), cell_index=1)
dt_error = abs(ref_sol - X[:, -1]).max()
print("Max error vs reference for timestep ", timestep, ": ", dt_error)

# error ~ 1
print("\nTesting dt = 1e-1")
timestep = 1e-1
num_iter = int(t_stop / timestep)
X, t = SimpleSolver(evalf, x0, p, evalu, num_iter, timestep, visualize=False)
EulerVis(t, X, grid_size=(rows, cols), cell_index=1)
dt_error = abs(ref_sol - X[:, -1]).max()
print("Max error vs reference for timestep ", timestep, ": ", dt_error)

# error ~ 1
print("\nTesting dt = 7.64e-1")
timestep = 7.64e-1
num_iter = int(t_stop / timestep)
X, t = SimpleSolver(evalf, x0, p, evalu, num_iter, timestep, visualize=False)
EulerVis(t, X, grid_size=(rows, cols), cell_index=1)
dt_error = abs(ref_sol - X[:, -1]).max()
print("Max error vs reference for timestep ", timestep, ": ", dt_error)

# error ~ 1.9
print("\nTesting dt = 1")
timestep = 1
num_iter = int(t_stop / timestep)
X, t = SimpleSolver(evalf, x0, p, evalu, num_iter, timestep, visualize=False)
EulerVis(t, X, grid_size=(rows, cols), cell_index=1)
dt_error = abs(ref_sol - X[:, -1]).max()
print("Max error vs reference for timestep ", timestep, ": ", dt_error)

# Next steps: run trapezoidal for the same parameters and compare plots
# Add in dynamic timesteps to trapezoidal
# Compare for larger grid sizes the speed and performace
# Final results include:
# Plots of euler becoming unstable/oscillating at sufficiently large timestep
# Plots of trapezoidal remaining stable for same timestep
# Timing comparisons for large grids (e.g., 10x10) between euler and trapezoidal (standardize to same error or time)
# Show how dynamic timestep increases speed of trapezoidal without losing accuracy

