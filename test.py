# %%
import numpy as np
import evalf as f
import eval_Jf_FiniteDifference_flatten as Jf
np.set_printoptions(linewidth=np.inf, precision=2, suppress=True)

# Test parameters where everything is 1
## Note-- need a large dx to have any results with these parameters
x0_t1 = np.array([
    [[1,1,1,1,1], [1,1,1,1,1]],
    [[1,1,1,1,1], [1,1,1,1,1]]
])

p_t1 = f.Params(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2)

u_t1 = 1

J_t1, dxdF = Jf.eval_Jf_FiniteDifference(f.evalf, x0_t1, p_t1, u_t1)
#print(J_t1.shape)
#print(J_t1)


# Parameters somewhat taken from the paper
x0_acc = np.array([
    [[10e6,10e6,0.0029,0,0], [10e6,10e6,0.0029,0,0]],
    [[10e6,10e6,0.0029,0,0], [10e6,10e6,0.0029,0,0]]
])
p_acc = f.Params(0.5, 5e7, 2, 3e-7, 0.1, 10, 0.18, 0.01, 0.03, 3e-7, 12.6, 0.1, 0.01, 2.5e-8, 18, 0.01, 0.03, 0.05, 0.01, 1, 2, 2)
u_acc = 0.015

J_acc, dxdF = Jf.eval_Jf_FiniteDifference(f.evalf, x0_acc, p_acc, u_acc)
print(J_acc)