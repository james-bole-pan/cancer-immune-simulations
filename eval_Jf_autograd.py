from autograd import jacobian
from evalf_autograd import Params 

def eval_Jf_autograd(eval_f, x0, p, u):
    x0_flat = x0.reshape(-1)

    def _f_flat(x_flat, p: Params, u):
        X = x_flat.reshape(p.rows, p.cols, 5)
        F = eval_f(X, p, u)
        return F.reshape(-1)

    J_ad = jacobian(_f_flat)
    J = J_ad(x0_flat, p, u)

    return J

