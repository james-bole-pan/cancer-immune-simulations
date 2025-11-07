import numpy as np
import matplotlib.pyplot as plt
from visualize_state import visualize_state 
from eval_Jf_FiniteDifference import eval_Jf_FiniteDifference 

def newtonNd(fhand, x0, p, u,errf,errDeltax,relDeltax,MaxIter,visualize, FiniteDifference, Jfhand):
    """
    # uses Newton Method to solve the VECTOR nonlinear system f(x)=0
    # x0        is the initial guess for Newton iteration
    # p         is a structure containing all parameters needed to evaluate f( )
    # u         contains values of inputs 
    # eval_f    is a text string with name of function evaluating f for a given x 
    # eval_Jf   is a text string with name of function evaluating Jacobian of f at x (i.e. derivative in 1D)
    # FiniteDifference = 1 forces the use of Finite Difference Jacobian instead of given eval_Jf
    # errF      = absolute equation error: how close do you want f to zero?
    # errDeltax = absolute output error:   how close do you want x?
    # relDeltax = relative output error:   how close do you want x in perentage?
    # note: 		declares convergence if ALL three criteria are satisfied 
    # MaxIter   = maximum number of iterations allowed
    # visualize = 1 shows intermediate results
    #
    # OUTPUTS:
    # converged   1 if converged, 0 if not converged
    # errf_k      ||f(x)||
    # errDeltax_k ||X(end) -X(end-1)||
    # relDeltax_k ||X(end) -X(end-1)|| / ||X(end)||
    # iterations  number of Newton iterations k to get to convergence
    #
    # EXAMPLE:
    # x,converged,errf_k,errDeltax_k,relDeltax_k,iterations = newtonNd(eval_f,x0,p,u,errf,errDeltax,relDeltax,MaxIter,visualize,FiniteDifference,eval_Jf)
    """

    k = 0                        # Newton iteration index
    X = np.zeros((len(x0), MaxIter+1))
    X[:,k] = x0                        # X stores intermetiade solutions as columns

    f = fhand(X[:,k],p,u)
    errf_k  = np.linalg.norm(f, np.inf)

    errDeltax_k = np.float32('inf')
    relDeltax_k = np.float32('inf')

    # Initialize visualization
    if visualize:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1)
        fig.show()


    while k < MaxIter and (errf_k > errf or errDeltax_k > errDeltax or relDeltax_k > relDeltax):

        if FiniteDifference:
            x_col = X[:,k].reshape(-1,1)
            Jf,_ = eval_Jf_FiniteDifference(fhand,x_col,p,u)
        else: 
            Jf = Jfhand(fhand, X[:,k],p,u)

        detJ = np.linalg.det(Jf)
        rankJ = np.linalg.matrix_rank(Jf)
        condJ = np.linalg.cond(Jf)

        print(f"[DEBUG] det(Jf)={detJ:.2e}, rank={rankJ}/{Jf.shape[0]}, cond={condJ:.2e}")

        if rankJ < Jf.shape[0] or not np.isfinite(detJ) or abs(detJ) < 1e-12 or condJ > 1e12:
            print(f"[WARNING] Singular or ill-conditioned Jacobian detected at iteration {k}.")
            Jf = Jf + 1e-6 * np.eye(Jf.shape[0])

        Deltax = np.linalg.solve(Jf, -f)       #NOTE this is the only difference from 1D to multiD
        Deltax = Deltax.flatten()
        X[:, k+1] = X[:,k] + Deltax
        k = k+1
        f = fhand(X[:,k],p,u)
        errf_k = np.linalg.norm(f, np.inf)
        errDeltax_k = np.linalg.norm(Deltax, np.inf)
        relDeltax_k = np.linalg.norm(Deltax, np.inf)/max(abs(X[:,k]))
        
        # Update plot
        if visualize:
            ax_top, ax_bottom = visualize_state(range(1, k + 2), X, k, '.b', ax_top, ax_bottom)
            plt.pause(0.001)

    x = X[:, k]    # extracting the very last solution

    # returning the number of iterations with ACTUAL computation
    # i.e. exclusing the given initial guess
    iterations = k 


    if errf_k <=errf and errDeltax_k<=errDeltax and relDeltax_k<=relDeltax:
        converged = True
        if visualize:
            print('Newton converged in {} iterations'.format(iterations))
    else:
        converged = False
        print('Newton did NOT converge! Maximum Number of Iterations reached')
    
    return x, converged, errf_k, errDeltax_k, relDeltax_k, iterations, X[:, 0:k+1]