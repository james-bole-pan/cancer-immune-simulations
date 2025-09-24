from autograd import jacobian
import autograd.numpy as np
import matplotlib.pyplot as plt
from evalf_autograd import evalf_autograd, Params
import eval_Jf_FiniteDifference_flatten as Jf

# Flatten / unflatten helpers for Jacobians
def flatten_x(x):         return x.reshape(-1)
def unflatten_x(x_flat, rows, cols): return x_flat.reshape(rows, cols, 5)

def f_flat(x_flat, p: Params, u):
    X = unflatten_x(x_flat, p.rows, p.cols)
    F = evalf_autograd(X, p, u)
    return flatten_x(F)

# initial state (2x2 grid, 5 vars)
x0 = np.array([
    [[1.0e7, 1.0e7, 0.0029, 0.02, 0.015], [1.0e7, 1.0e7, 0.0029, 0.02, 0.015]],
    [[1.0e7, 1.0e7, 0.0029, 0.02, 0.015], [1.0e7, 1.0e7, 0.0029, 0.02, 0.015]],
], dtype=float)

p = Params(
    lc=0.5, tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10.0, dc=0.18, D_c=0.01,
    lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
    ligt8=2.5e-8, dig=18.0, D_ig=0.01,
    mu_a=0.03, da=0.05, D_a=0.01,
    rows=2, cols=2
)
u = 0.015

x0_flat = flatten_x(x0)

# --- Autograd Jacobian (analytic via AD) ---
J_ad_fun = jacobian(lambda z: f_flat(z, p, u))
J_ad = J_ad_fun(x0_flat)        # shape (5N, 5N)

# --- FD Jacobians for different dx, and error vs dx ---
dx_values = 10.0 ** np.arange(-16, 1, 0.1)   # from 1e-16 to 1e0
errors = []

for dx in dx_values:
    # Set the finite difference step size
    p.dxFD = dx
    # Use the imported finite difference Jacobian function
    J_fd, _ = Jf.eval_Jf_FiniteDifference(evalf_autograd, x0, p, u)
    err = np.linalg.norm(J_fd - J_ad, ord='fro')   # ||J_FD - J_AD||_F
    errors.append(err)

errors = np.asarray(errors)

# Calculate reference dx values
eps = np.finfo(float).eps
dx_machine = np.sqrt(eps)  # sqrt(machine epsilon)

norm_inf = np.linalg.norm(x0_flat, np.inf)
dx_nitsol  = 2.0 * np.sqrt(eps) * max(1.0, norm_inf)  # 2√eps max(1, ||x||_inf)

# Find optimal dx (smallest error), no constraints
idx_optimal = np.argmin(errors)
dx_optimal = dx_values[idx_optimal]

# --- Plot (line plot, no markers) ---
plt.figure()
plt.loglog(dx_values, errors, linewidth=2)
plt.grid(False, which='both')
plt.xlabel(r'$dx$')
plt.ylabel(r'$\|J_{\mathrm{FD}}(dx)-J_{\mathrm{AD}}\|_F$')
plt.title('Jacobian comparison: finite difference vs autograd')
plt.axvline(dx_machine, linestyle='--', alpha=0.7, label=r'$\sqrt{\varepsilon_{machine}}$')
plt.axvline(dx_nitsol, linestyle='-.', alpha=0.7, color='green', label=r'$2\sqrt{\varepsilon}\max(1,\|x\|_{\infty})$')
plt.axvline(dx_optimal, linestyle=':', alpha=0.7, color='red', label=f'Optimal dx = {dx_optimal:.2e}')
plt.legend()
plt.tight_layout()
plt.show()