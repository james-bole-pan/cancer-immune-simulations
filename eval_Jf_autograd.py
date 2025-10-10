from autograd import jacobian
import autograd.numpy as anp

def eval_Jf_autograd(eval_f, x0_col, p, u):
    """
    Compute Jacobian of eval_f at state x0 for the (N,1) column-vector interface.

    Parameters
    ----------
    eval_f : function(x_col, p, u) -> (N,1) column vector
    x0_col : ndarray, shape (N,1)
        Current state as a column vector.
    p : Params
    u : float

    Returns
    -------
    J : ndarray, shape (N, N)
        Jacobian matrix d f / d x at x0.
    """
    x0_vec = anp.ravel(x0_col)  # (N,)

    def _f_vec(x_vec, p, u):
        # Convert 1D -> (N,1) for eval_f, then back to 1D for autograd
        f_col = eval_f(x_vec.reshape((-1, 1)), p, u)  # (N,1)
        return anp.ravel(f_col)                       # (N,)

    J_fun = jacobian(_f_vec)
    J = J_fun(x0_vec, p, u)  # (N,N)
    return J
