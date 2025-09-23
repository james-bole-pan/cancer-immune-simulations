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
p_acc = f.Params(0.5, 5e7, 2, 3e-7, 0.1, 10, 0.18, 0.01, 0.03, 3e-7, 12.6, 0.1, 0.01, 2.5e-8, 
                 18, 0.01, 0.03, 0.05, 0.01, 1, 2, 2)
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

