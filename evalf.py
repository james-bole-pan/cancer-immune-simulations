import numpy as np

class Params:
    def __init__(self, lc=0, tc=0, nc=0, k8=0, ng=0, ki=0, dc=0, D_c=0, lt8=0, rl=0, kq=0, dt8=0,
                 D_t8=0, ligt8=0, dig=0, D_ig=0, mu_a=0, da=0, D_a=0, dxFD=10e-6, rows=1, cols=1):
        self.lc = lc # lambda_c -            0.5
        self.tc = tc # theta_c -             5e7
        self.nc = nc # n_c -                 2
        self.k8 = k8 # kappa_8 -             3e-7
        self.ng = ng # n_g -                 0.1
        self.ki = ki # K_i                   couldn't find
        self.dc = dc # d_c -                 0.18
        self.D_c = D_c # D_c                 0.01
        self.lt8 = lt8 # lambda_t8 -         0.03
        self.rl = rl # rho_l -               3e-7
        self.kq = kq # K_q -                 12.6
        self.dt8 = dt8 # d_t8 -              0.1
        self.D_t8 = D_t8 # D_t8 -            0.01
        self.ligt8 = ligt8 # lambda_igt8 -   2.5e-8
        self.dig = dig # d_ig -              18
        self.D_ig = D_ig # D_ig -            0.01
        self.mu_a = mu_a # mu_a -            0.03
        self.da = da # d_a -                 0.05
        self.D_a = D_a # D_a -               0.01
        self.dxFD = dxFD # Step size
        self.rows = rows # rows in grid
        self.cols = cols # cols in grid

    def tuple(self):
        return (self.lc, self.tc, self.nc, self.k8, self.ng, self.ki, self.dc, self.D_c,
                self.lt8, self.rl, self.kq, self.dt8, self.D_t8, self.ligt8, self.dig,
                self.D_ig, self.mu_a, self.da, self.D_a)

def evalf(x, p, u):
    shape = np.shape(x)
    grid_x = shape[0]
    grid_y = shape[1]
    assert shape[2] == 5, "x must have five components in the last dimension"
    (lc, tc, nc, k8, ng, ki, dc, D_c, lt8, rl, kq, dt8, D_t8, ligt8, dig, D_ig, mu_a, da, D_a) = p.tuple()
    ra = u # 0.015
    f_val = np.zeros(shape)

    for i in range(grid_x):
        for j in range(grid_y):
            c, t8, ig, p8, a = x[i, j, :]
            neighbors = sum(x[i + di, j + dj, :] for di in [-1, 0, 1] for dj in [-1, 0, 1]
                            if di != dj and 0 <= i + di < grid_x and 0 <= j + dj < grid_y)
            cn, t8n, ign, _, an = neighbors
            del_c = (lc / (1 + (c / tc)**nc) - (k8 * t8 + ng * ig / (ig + ki)) - dc) * c + D_c * cn
            del_t8 = (lt8 / (1 + rl * p8 * c / kq) - dt8) * t8 + D_t8 * t8n
            del_ig = ligt8 * t8 - dig * ig + D_ig * ign
            del_p8 = p8 / t8 * del_t8 - mu_a * p8 * a
            del_a = ra - (mu_a * p8 - da) * a + D_a * an
            f_val[i, j, :] = (del_c, del_t8, del_ig, del_p8, del_a)

    return f_val
