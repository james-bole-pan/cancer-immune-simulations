# test_jacobian_two_methods.py
import numpy as np
import matplotlib.pyplot as plt

from eval_f import evalf_autograd, Params
from eval_Jf_autograd import eval_Jf_autograd
from eval_Jf_FiniteDifference import eval_Jf_FiniteDifference

# --- Setup: initial state ---
rows, cols, d = 2, 2, 3
N = rows * cols * d

x0_flat = np.zeros(N)
x0_flat[0::3] = 15.0   # tumor cells
x0_flat[1::3] = 2.0    # T cells
x0_flat[2::3] = 0.0    # drug
x0 = x0_flat.reshape((-1, 1))   # (N,1) column vector

p = Params(
    lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
    lambda_T=3.0, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
    d_A=0.0315, rows=rows, cols=cols
)
u = 0.0

# --- Autograd Jacobian ---
J_ad = eval_Jf_autograd(evalf_autograd, x0, p, u)

# --- FD Jacobians for different dx, and error vs dx ---
dx_values = 10.0 ** np.arange(-16, 1, 0.25)  # from 1e-16 to 1
errors = []

for dx in dx_values:
    p.dxFD = dx
    J_fd, _ = eval_Jf_FiniteDifference(evalf_autograd, x0.copy(), p, u)
    err = np.linalg.norm(J_fd - J_ad, ord='fro')
    errors.append(err)

errors = np.asarray(errors)

# --- Reference dx values ---
eps = np.finfo(float).eps
dx_machine = np.sqrt(eps)

norm_inf = np.linalg.norm(x0.reshape(-1), np.inf)
dx_nitsol = 2.0 * np.sqrt(eps) * max(1.0, norm_inf)

# --- Optimal dx (smallest error) ---
idx_optimal = np.argmin(errors)
dx_optimal = dx_values[idx_optimal]

# --- Plot ---
plt.figure(figsize=(7,5))
plt.loglog(dx_values, errors, linewidth=2)
plt.xlabel(r"$dx$")
plt.ylabel(r"$\|J_{\mathrm{FD}}(dx)-J_{\mathrm{AD}}\|_F$")
plt.title("Jacobian comparison: Finite Difference vs Autograd")
plt.axvline(dx_machine, linestyle="--", alpha=0.7, label=r"$\sqrt{\varepsilon}$")
plt.axvline(dx_nitsol, linestyle="-.", alpha=0.7, color="green",
            label=r"$2\sqrt{\varepsilon}\max(1,\|x\|_{\infty})$")
plt.axvline(dx_optimal, linestyle=":", alpha=0.7, color="red",
            label=f"Optimal dx = {dx_optimal:.2e}")
plt.legend()
plt.grid(True, which="both", alpha=0.5)
plt.tight_layout()
plt.show()