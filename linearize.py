from eval_Jf_FiniteDifference import eval_Jf_FiniteDifference
from eval_Jf_autograd import eval_Jf_autograd
import numpy as np

def linearize_FiniteDifference(eval_f, x0, u0, p):
    """
    Linearize the nonlinear system dx/dt = f(x, p, u) where u = u(t).
    Computes the Jacobians of f using the finite difference method.
    
    :param eval_f: A function which evaluates f(x, p, u) given x, p, u as arguments
    :param x0: The value of x at which to linearize
    :param u0: The value of u at which to linearize
    :param p: The parameter values for the system (to be plugged into eval_f)
    """

    def swapped_eval_f(u, p, x):
        if not np.isscalar(u) and np.prod(u.shape) == 1:
            u = u.item()

        return eval_f(x, p, u)
    
    u0_array = np.array([[u0]]) if np.isscalar(u0) else u0
    Jx, _ = eval_Jf_FiniteDifference(eval_f, x0, p, u0)
    Ju, _ = eval_Jf_FiniteDifference(swapped_eval_f, u0_array, p, x0)

    K = eval_f(x0, p, u0) - Jx @ x0 - Ju @ u0_array

    A = Jx
    B = np.hstack((K.reshape(-1, 1), Ju))
    return A, B

def linearize_autograd(eval_f, x0, u0, p):
    """
    Linearize the nonlinear system dx/dt = f(x, p, u) where u = u(t).
    Computes the Jacobians of f using automatic differentiation.
    
    :param eval_f: A function which evaluates f(x, p, u) given x, p, u as arguments
    :param x0: The value of x at which to linearize
    :param u0: The value of u at which to linearize
    :param p: The parameter values for the system (to be plugged into eval_f)
    """

    def swapped_eval_f(u, p, x):
        if hasattr(u, 'shape') and np.prod(u.shape) == 1:
            u = u[(0,)*len(u.shape)]

        return eval_f(x, p, u)
    
    u0_array = np.array([[u0]]) if np.isscalar(u0) else u0

    Jx = eval_Jf_autograd(eval_f, x0, p, u0)
    Ju = eval_Jf_autograd(swapped_eval_f, u0_array, p, x0)
    K = eval_f(x0, p, u0) - Jx @ x0 - Ju @ u0_array

    A = Jx
    B = np.hstack((K.reshape(-1, 1), Ju))
    return A, B
