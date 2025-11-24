import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Import the dosing function and solver components
# ---------------------------------------------------

from SimpleSolver import SimpleSolver   # if in same directory adjust accordingly
from eval_u_keytruda_input import eval_u_keytruda_input   # <-- modify import as necessary

# --------------------------
# 2. Define a dummy eval_f
# --------------------------
def eval_f_dummy(x, p, u):
    """
    Simple test ODE:
        dx/dt = u
    So the solution should be cumulative dosing input.
    """
    return np.array([u])

# --------------------------
# 3. Test parameters
# --------------------------
dose = 200.0
interval = 21.0
NumIter = 8400      # short test
w = 0.01           # time-step

# Create the dosing input function
u_fun = eval_u_keytruda_input(dose=dose, interval=interval)

# -----------------------------------------
# 4. Visual test: plot the dosing schedule
# -----------------------------------------
times = np.arange(0, NumIter * w, w)
inputs = np.array([u_fun(t) for t in times])

plt.figure(figsize=(8,4))
plt.plot(times, inputs, marker='.', label='Keytruda Input')
plt.xlabel("Time")
plt.ylabel("Dose Input")
plt.title("Check Keytruda Dosing Pulses")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("test_evalf_output_figures/keytruda_input_test_plot.png")
plt.show()
plt.close()

print("✓ Saved dosing pulse plot as test_evalf_output_figures/keytruda_input_test_plot.png")

# ---------------------------------------------------------
# 5. Solver test: does the system accumulate inputs properly?
# ---------------------------------------------------------

x_start = np.array([0.0])   # initial condition
p = None                    # no parameters needed

X, t = SimpleSolver(
    eval_f=eval_f_dummy,
    x_start=x_start,
    p=p,
    eval_u=u_fun,
    NumIter=NumIter,
    w=w,
    visualize=False
)

# ----------------------------------------
# 6. Plot accumulated state over time
# ----------------------------------------

plt.figure(figsize=(8,4))
plt.plot(t, X[0], label="State (cumulative dosing)")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title("SimpleSolver State with Keytruda Pulses")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("test_evalf_output_figures/keytruda_solver_test_plot.png")
plt.close()

print("✓ Saved solver test result as test_evalf_output_figures/keytruda_solver_test_plot.png")

# ----------------------------------------
# 7. Print final sanity checks
# ----------------------------------------
print("\nFinal state value:", X[0, -1])
print("Number of pulses detected:", np.count_nonzero(inputs > 0))
print("Pulse times:", times[inputs > 0])
