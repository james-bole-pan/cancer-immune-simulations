import numpy as np
import os
import matplotlib.pyplot as plt
from time import time
from implicit import implicit

def transient_reference(err_tol, filename, eval_f, p, eval_u, x_start, t_start, t_stop, eval_Jf=None):
    """
    Generates a golden reference by running a convergence test.

    INPUTS:
    err_tol     - ideal desired confidence level for the generated reference solution (e.g., 1e-6)
    filename    - file to save/load the reference values
    eval_f      - function to evaluate f(x, p, u)
    p           - dictionary containing parameters for the model
    eval_u      - function returning the input at a given time t
    x_start     - initial state vector
    t_start     - initial time for the ODE integrator
    t_stop      - final time for the ODE integrator
    eval_Jf     - [optional] function to evaluate the Jacobian; if not provided, finite difference is used

    OUTPUTS:
    X_ref       - reference solution
    t_ref       - times of the reference solution
    dt_ref      - timestep of the reference solution
    err_ref     - actual confidence level of the generated reference solution
    
    EXAMPLE:
    X_ref, t_ref, dt_ref, err_ref = transient_reference(err_tol, filename, eval_f, p, eval_u, x_start, t_start, t_stop, eval_Jf)
    X_ref, t_ref, dt_ref, err_ref = transient_reference(err_tol, filename, eval_f, p, eval_u, x_start, t_start, t_stop)
    """

    if os.path.exists(filename):
        print("\nLoading scalar dt_ref err_ref and vectors t_ref and X_ref from saved file:")
        print(filename)
        data = np.load(filename)
        X_ref, t_ref, dt_ref, err_ref = data['X_ref'], data['t_ref'], data['dt_ref'], data['err_ref']

        if err_ref < err_tol:
            return X_ref, t_ref, dt_ref, err_ref
        else:
            print("The loaded reference is not accurate enough.")

    print("\nRunning a convergence test with decreasing dt until desired err_tol is reached")
    visualize = False
    FiniteDifference = True if eval_Jf is None else False
    dt = []
    X, t, X_diff, comp_times = [], [], [], []
    k = 1
    n = -1
    err_ref = np.inf

    while err_ref > err_tol:
        current_dt = 10**n
        dt.append(current_dt)

        start_time = time()
        X_k, t_k, _ = implicit('Trapezoidal', eval_f, x_start, p, eval_u, t_start, t_stop, current_dt, visualize, FiniteDifference, eval_Jf)
        elapsed_time = time() - start_time
        comp_times.append(elapsed_time)

        if k > 1:
            X_diff.append(np.abs(X_k[-1, -1] - X[-1][-1, -1]))

            # Plot error
            plt.loglog(dt[1:], X_diff, 'b.')
            plt.pause(0.01)

            # Save intermediate reference
            t_ref = t_k
            X_ref = X_k
            dt_ref = current_dt
            err_ref = X_diff[-1]
            print(f"Saved reference with confidence: {err_ref:.2e}. Solved in {elapsed_time:.3f} seconds with dt = {current_dt:.2e}")

            np.savez(filename, X_ref=X_ref, t_ref=t_ref, dt_ref=dt_ref, err_ref=err_ref)

        X.append(X_k)
        t.append(t_k)

        k += 1
        n -= 0.5

    print("\nConvergence test completed.")

    return X_ref, t_ref, dt_ref, err_ref
