import numpy as np
import matplotlib.pyplot as plt
from visualize_state import visualize_state
from newtonNd import newtonNd

# Newton params (fhand, x0, p, u,errf,errDeltax,relDeltax,MaxIter,visualize, FiniteDifference, Jfhand)

# Function to set up F_trap-- the function used for eval_f in the Newton solver
# Uses params to pass in values from main trapezoidal function while preserving f(x, p, u) structure
def f_trap(x, params, u):
    # Coerce inputs to 1-D arrays (shape (n,))
    x   = np.asarray(x).reshape(-1)
    x_n = np.asarray(params["x_n"]).reshape(-1)

    # f_l might be (n,1) or (n,) so flatten it too
    f_l = np.asarray(params["f_l"]).reshape(-1)

    dt    = params["dt"]
    p     = params["p"]
    u_np1 = params["u_np1"]
    eval_f = params["eval_f"]

    # Evaluate eval_f and force it to be 1-D
    f_eval = np.asarray(eval_f(x, p, u_np1)).reshape(-1)

    # Build trapezoidal residual as 1-D vector
    res = x - x_n - 0.5 * dt * (f_l + f_eval)

    # Ensure result is 1-D (shape (n,))
    return res.reshape(-1)

# Function to perform integration using trapezoidal method
def trapezoidal(eval_f, x_start, p, eval_u, t_start, t_stop, timestep,
                           errf=1e-9, errDeltax=1e-9, relDeltax=1e-9,
                           MaxIter=20):

    # Number of time steps
    num_steps = int(np.ceil((t_stop - t_start) / timestep)) + 1

    # Solution matrix and time vector
    X = np.zeros((len(x_start), num_steps))
    t = np.zeros(num_steps)

    # Start with inital conditions
    X[:, 0] = x_start
    t[0] = t_start

    # Integrate over each timestep
    for l in range(num_steps - 1):

        # If the timestep doesn't go into the time range evenly, then change dt for the last timestep to avoid errors
        dt = min(timestep, t_stop - t[l])
        t[l+1] = t[l] + dt

        # Inputs and state at t_l and t_{l+1}
        u_n   = eval_u(t[l])
        u_np1 = eval_u(t[l+1])

        # Evaluate f(x_n)
        f_l = eval_f(X[:, l], p, u_n).ravel()

        # Build param package for residual function
        params = {
            "x_n"   : X[:, l],
            "f_l"   : f_l,
            "dt"    : dt,
            "p"     : p,
            "u_np1" : u_np1,
            "eval_f": eval_f
        }

        # Initial guess
        x0 = X[:, l]

        # Solve g(x)=0 using Newton
        x_next, converged, _, _, _, _, _ = newtonNd(
            f_trap,   
            x0,
            params,                 
            u_np1,                  
            errf, errDeltax, relDeltax,
            MaxIter,
            visualize=False,
            FiniteDifference=1,   
            Jfhand=None
        )

        if not converged:
            print("Newton failed to converge at step: ", l)

        X[:, l+1] = x_next.ravel()

    return X, t
