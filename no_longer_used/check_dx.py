import numpy as np
import matplotlib.pyplot as plt
import no_longer_used.evalf as f
import eval_Jf_FiniteDifference_flatten as Jf

# Testing with paper params
x0_acc = np.array([
    [[10e6,10e6,0.0029,0.02,0.015], [10e6,10e6,0.0029,0.02,0.015]],
    [[10e6,10e6,0.0029,0.02,0.015], [10e6,10e6,0.0029,0.02,0.015]]
])

p_acc = f.Params(
    lc=0.5,        # lambda_c
    tc=5e7,        # theta_c
    nc=2,          # n_c
    k8=3e-7,       # kappa_8
    ng=0.1,        # n_g
    ki=10,         # K_i
    dc=0.18,       # d_c
    D_c=0.01,      # D_c
    lt8=0.03,      # lambda_t8
    rl=3e-7,       # rho_l
    kq=12.6,       # K_q
    dt8=0.1,       # d_t8
    D_t8=0.01,     # D_t8
    ligt8=2.5e-8,  # lambda_igt8
    dig=18,        # d_ig
    D_ig=0.01,     # D_ig
    mu_a=0.03,     # mu_a
    da=0.05,       # d_a
    D_a=0.01,      # D_a
    rows=2,        # rows in grid
    cols=2         # cols in grid
)

u_acc = 0.015

# --- Sweep dx for FD Jacobian and measure self-consistency ---
dx_values = 10.0 ** np.arange(-17, 1, 0.1, dtype=float)
errors = []

for dx in dx_values:
    # J_FD(dx)
    p_acc.dxFD = dx
    J1, _ = Jf.eval_Jf_FiniteDifference(f.evalf, x0_acc, p_acc, u_acc)
    # J_FD(2dx)
    p_acc.dxFD = 2.0 * dx
    J2, _ = Jf.eval_Jf_FiniteDifference(f.evalf, x0_acc, p_acc, u_acc)

    errors.append(np.linalg.norm(J1 - J2, ord='fro'))

errors = np.asarray(errors)

# dx_machine would be square-root of machine precision
dx_machine = np.sqrt(np.finfo(float).eps)

# Find dx corresponding to minimum error, but constrain to be >= sqrt(machine precision)
# Only consider dx values that are >= dx_machine
valid_indices = dx_values >= dx_machine
valid_dx_values = dx_values[valid_indices]
valid_errors = errors[valid_indices]

if len(valid_errors) > 0:
    min_error_idx_valid = np.argmin(valid_errors)
    dx_optimal = valid_dx_values[min_error_idx_valid]
else:
    # Fallback: if no dx values are >= dx_machine, use dx_machine itself
    dx_optimal = dx_machine

# --- Plot ---
plt.figure()
plt.loglog(dx_values, errors, linewidth=2)
plt.grid(True, which='both')
plt.xlabel(r'$dx$')
plt.ylabel(r'$\|J_{\mathrm{FD}}(dx)-J_{\mathrm{FD}}(2dx)\|_F$')
plt.title('Finite-Difference Jacobian Errors')
plt.axvline(dx_machine, linestyle='--', alpha=0.7, label=r'$\sqrt{\varepsilon_{machine}}$')
plt.axvline(dx_optimal, linestyle=':', alpha=0.7, color='red', label=f'Optimal dx = {dx_optimal:.2e}')
plt.legend()
plt.tight_layout()
plt.show()