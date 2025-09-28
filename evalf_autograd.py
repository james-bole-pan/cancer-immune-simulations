# %%
# evalf function, just using autograd to allow easy Jacobian computation
import numpy as np
import matplotlib.pyplot as plt

# --- autograd imports ---
import autograd
import autograd.numpy as anp
from autograd import jacobian

# ------------------------------
# Original Params (unchanged)
# ------------------------------
class Params:
    def __init__(self, lc=0, tc=0, nc=0, k8=0, ng=0, ki=0, dc=0, D_c=0, lt8=0, rl=0, kq=0, dt8=0,
                 D_t8=0, ligt8=0, dig=0, D_ig=0, mu_a=0, da=0, D_a=0, dxFD=10e-6, rows=1, cols=1):
        self.lc = lc; self.tc = tc; self.nc = nc; self.k8 = k8; self.ng = ng; self.ki = ki
        self.dc = dc; self.D_c = D_c; self.lt8 = lt8; self.rl = rl; self.kq = kq; self.dt8 = dt8
        self.D_t8 = D_t8; self.ligt8 = ligt8; self.dig = dig; self.D_ig = D_ig
        self.mu_a = mu_a; self.da = da; self.D_a = D_a
        self.dxFD = dxFD; self.rows = rows; self.cols = cols

    def tuple(self):
        return (self.lc, self.tc, self.nc, self.k8, self.ng, self.ki, self.dc, self.D_c,
                self.lt8, self.rl, self.kq, self.dt8, self.D_t8, self.ligt8, self.dig,
                self.D_ig, self.mu_a, self.da, self.D_a, self.dxFD, self.rows, self.cols)

# ------------------------------
# Autograd-compatible evalf
# ------------------------------
def evalf_autograd(x, p: Params, u):
    """
    x: anp.ndarray with shape (rows, cols, 5)
    returns: same shape of time-derivatives
    """
    (lc, tc, nc, k8, ng, ki, dc, D_c, lt8, rl, kq, dt8, D_t8,
     ligt8, dig, D_ig, mu_a, da, D_a, dxFD, rows, cols) = p.tuple()
    ra = u

    rows, cols, d = x.shape
    assert d == 5
    f_val = []

    eps_div = 1e-12  # numerical safety for division by T8

    def neigh_sum(X, i, j, comp):
        s = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ii, jj = i + di, j + dj
                if 0 <= ii < rows and 0 <= jj < cols:
                    s = s + X[ii, jj, comp]
        return s

    for i in range(rows):
        row_vals = []
        for j in range(cols):
            c, t8, ig, p8, a = x[i, j, :]

            cn  = neigh_sum(x, i, j, 0)
            t8n = neigh_sum(x, i, j, 1)
            ign = neigh_sum(x, i, j, 2)
            an  = neigh_sum(x, i, j, 4)

            del_c  = (lc / (1 + (c / tc)**nc) - (k8 * t8 + ng * ig / (ig + ki)) - dc) * c + D_c * cn
            del_t8 = (lt8 / (1 + rl * p8 * c / kq) - dt8) * t8 + D_t8 * t8n
            del_ig = ligt8 * t8 - dig * ig + D_ig * ign
            del_p8 = (p8 / (t8 + eps_div)) * del_t8 - mu_a * p8 * a
            del_a  = ra - mu_a * p8 * a - da * a + D_a * an

            row_vals.append(anp.stack([del_c, del_t8, del_ig, del_p8, del_a]))
        f_val.append(anp.stack(row_vals))
    return anp.stack(f_val)
