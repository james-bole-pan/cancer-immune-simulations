import numpy as np

class Params:
    def __init__(self, lc=0, tc=0, nc=0, kd=0, ng=0, ki=0, dc=0, mc=0, lt8=0, rl=0, kq=0, dt8=0,
                 mt8=0, ligt8=0, dig=0, mg=0, mua=0, da=0, ma=0):
        self.lc = lc
        self.tc = tc
        self.nc = nc
        self.kd = kd
        self.ng = ng
        self.ki = ki
        self.dc = dc
        self.mc = mc
        self.lt8 = lt8
        self.rl = rl
        self.kq = kq
        self.dt8 = dt8
        self.mt8 = mt8
        self.ligt8 = ligt8
        self.dig = dig
        self.mg = mg
        self.mua = mua
        self.da = da
        self.ma = ma
    
    def tuple(self):
        return (self.lc, self.tc, self.nc, self.kd, self.ng, self.ki, self.dc, self.mc,
                self.lt8, self.rl, self.kq, self.dt8, self.mt8, self.ligt8, self.dig,
                self.mg, self.mua, self.da, self.ma)

def evalf(x, p, u):
    shape = np.shape(x)
    grid_x = shape[0]
    grid_y = shape[1]
    assert shape[2] == 5, "x must have five components in the last dimension"
    (lc, tc, nc, kd, ng, ki, dc, mc, lt8, rl, kq, dt8, mt8, ligt8, dig, mg, mua, da, ma) = p.tuple()
    ga = u
    f_val = np.zeros(shape)

    for i in range(grid_x):
        for j in range(grid_y):
            c, t8, ig, p8, a = x[i, j, :]
            neighbors = sum(x[i + di, j + dj, :] for di in [-1, 0, 1] for dj in [-1, 0, 1]
                            if di != dj and 0 <= i + di < grid_x and 0 <= j + dj < grid_y)
            cn, t8n, ign, _, an = neighbors
            del_c = (lc / (1 + (c / tc)**nc) - (kd * t8 + ng * ig / (ig + ki)) - dc) * c + mc * cn
            del_t8 = (lt8 / (1 + rl * p8 * c / kq) - dt8) * t8 + mt8 * t8n
            del_ig = ligt8 * t8 - dig * ig + mg * ign
            del_p8 = p8 / t8 * del_t8 - mua * p8 * a
            del_a = ga - (mua * p8 - da) * a + ma * an
            f_val[i, j, :] = (del_c, del_t8, del_ig, del_p8, del_a)

    return f_val
