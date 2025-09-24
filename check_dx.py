# %%
import numpy as np
import matplotlib.pyplot as plt
import evalf as f
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

# plot to study error on Finite Difference Jacobian for different dx
k = 0
dx = []
error = []
dxFD = 0.1

# Evaluating Jf(dx) - Jf(2dx)
for n in np.arange(-17, 5, 0.11):
    dx.append(10 ** (n))
    p_acc.dxFD = dx[k]
    Jf_FiniteDifference1,_ = Jf.eval_Jf_FiniteDifference(f.evalf, x0_acc, p_acc, u_acc)
    p_acc.dxFD = p_acc.dxFD * 2  # 2dx
    Jf_FiniteDifference2,_ = Jf.eval_Jf_FiniteDifference(f.evalf, x0_acc, p_acc, u_acc)
    error.append(np.max(np.abs(Jf_FiniteDifference1 - Jf_FiniteDifference2)))
    k += 1

plt.loglog(dx, error)
plt.grid(True)
plt.xlabel('dxFD')
plt.axvline(x=dxFD, color='g', linestyle='--', label='dxFDnitsol')
plt.legend(['|| J_{FD}-J_{an} ||', 'dxFD'])
plt.title('Difference between Analytic & Finite Difference Jacobians')
plt.show()

