import numpy as np

def eval_Jf_FiniteDifference(eval_f, x0, p, u):
    """
    evaluates the Jacobian of the vector field f() at state x0
    p is a structure containing all model parameters
    u is the value of the input at the current time
    uses a finite difference approach computing one column k at the time
    as difference of function evaluations perturbed by scalar p.dxFD
    Jf[:, k] = (f(x0 + p.dxFD) - f(x0)) / p.dxFD
    If p.dxFD is NOT specified, uses NITSOL value p.dxFD = 2 * sqrt(eps) * max(1, norm(x)).

    EXAMPLES:
    Jf        = eval_Jf_FiniteDifference(eval_f,x0,p,u);
    [Jf,dxFD] = eval_Jf_FiniteDifference(eval_f,x0,p,u);
    """
    f_x0 = eval_f(x0, p, u)
    f_x0 = f_x0.flatten() # Flatten from 3d array into 1d array
    x0 = x0.flatten() # Flatten state matrix into state vector
    N = len(x0)  

    ## Edited this to work with class instead of dictionary
    if hasattr(p, 'dxFD'):
        dxFD = p.dxFD  # If user specified it, use that
    else:
        # dxFD = np.sqrt(np.finfo(float).eps) # works ok in general if ||x0|| not huge
        # dxFD = 2 * np.sqrt(np.finfo(float).eps) * (1 + np.linalg.norm(x0, np.inf)) # correction for ||x0|| very large (works best)
        # dxFD = 2 * np.sqrt(np.finfo(float).eps) * max(1, np.linalg.norm(x0, np.inf)) # similar correction for large ||x0||
        dxFD = 2 * np.sqrt(np.finfo(float).eps) * np.sqrt(1 + np.linalg.norm(x0, np.inf)) # used in the NITSOL solver
        # dxFD = 2 * np.sqrt(np.finfo(float).eps) * np.sqrt(max(1, np.linalg.norm(x0, np.inf))) # similar to NITSOL
        print(f'dxFD not specified: using 2*sqrt(eps)*sqrt(1+||x||) = {dxFD}')

    Jf = np.zeros((len(f_x0), N))

    for k in range(N):
        xk = x0.copy()
        xk[k] = x0[k] + dxFD
        xk = xk.reshape(p.rows, p.cols, 5) # Transform vector into 3d array for evalf
        f_xk = eval_f(xk, p, u)
        f_xk = f_xk.flatten() # Flatten from 3d array into 1d array
        Jf[:, k] = np.reshape((f_xk - f_x0) / dxFD,[-1])

    return Jf, dxFD
