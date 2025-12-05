from eval_Jf_FiniteDifference import eval_Jf_FiniteDifference
from eval_Jf_autograd import eval_Jf_autograd
import numpy as np

def linearize_FiniteDifference(eval_f, eval_u, x0, u0, p):
    """
    Linearize the nonlinear system dx/dt = f(x, p, u) where u = u(t).
    Computes the Jacobians of f using the finite difference method.
    
    :param eval_f: A function which evaluates f(x, p, u) given x, p, u as arguments
    :param eval_u: A function which evaluates u(t) given t as an argument
    :param x0: The value of x at which to linearize
    :param u0: The value of u at which to linearize
    :param p: The parameter values for the system (to be plugged into eval_f)
    """

    A = eval_Jf_FiniteDifference(eval_f, x0, p, u0)

    def swapped_eval_f(u, p, x):
        return eval_f(x, p, u)

    B = eval_Jf_FiniteDifference(swapped_eval_f, u0, p, x0)
    return A, B

def linearize_autograd(eval_f, eval_u, x0, u0, p):
    """
    Linearize the nonlinear system dx/dt = f(x, p, u) where u = u(t).
    Computes the Jacobians of f using automatic differentiation.
    
    :param eval_f: A function which evaluates f(x, p, u) given x, p, u as arguments
    :param eval_u: A function which evaluates u(t) given t as an argument
    :param x0: The value of x at which to linearize
    :param u0: The value of u at which to linearize
    :param p: The parameter values for the system (to be plugged into eval_f)
    """

    def swapped_eval_f(u, p, x):
        return eval_f(x, p, u)

    Jx = eval_Jf_autograd(eval_f, x0, p, u0)
    Ju = eval_Jf_autograd(swapped_eval_f, u0, p, x0)
    K = eval_f(x0, p, u0) - Jx @ x0 - Ju @ u0

    A = Jx
    print( "shape of A:", A.shape )
    print( "shape of K:", K.shape )
    print( "shape of Ju:", Ju.shape )
    B = np.hstack((K.reshape(-1, 1), Ju))
    print( "shape of B:", B.shape )
    return A, B
