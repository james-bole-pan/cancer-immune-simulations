import numpy as np
import matplotlib.pyplot as plt
from visualize_state import visualize_state

def forward_euler(eval_f, x_start, p, eval_u, t_start, t_stop, timestep, visualize=False):
    """
    Uses the Forward Euler algorithm to simulate the state model dx/dt = f(x, p, u)
    starting from state vector x_start at time t_start until time t_stop with time intervals of timestep.
    
    Parameters:
    eval_f     - function to evaluate f(x, p, u)
    x_start    - initial state vector
    p          - parameters needed for the function
    eval_u     - function to evaluate u(t)
    t_start    - start time
    t_stop     - stop time
    timestep   - time interval
    visualize  - if True, generates intermediate plots of the state
    
    Returns:
    X          - array of state vectors over time
    t          - array of time points
    """
    
    # Initialize arrays
    num_steps = int(np.ceil((t_stop - t_start) / timestep)) + 1
    X = np.zeros((len(x_start), num_steps))
    t = np.zeros(num_steps)
    
    # Set initial values
    X[:, 0] = x_start
    t[0] = t_start
    
    # Initialize visualization with two subplots
    if visualize:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1)
        fig.show()
    
    # Forward Euler loop
    for n in range(num_steps - 1):
        dt = min(timestep, t_stop - t[n])
        t[n + 1] = t[n] + dt
        u, _ = eval_u(t[n]) if isinstance(eval_u(t[n]), tuple) else (eval_u(t[n]), None)
        f = eval_f(X[:, n], p, u)
        X[:, n + 1] = X[:, n] + dt * f
        
        # Update visualization
        if visualize:
            ax_top, ax_bottom = visualize_state(t[:n+2], X[:, :n+2], n + 1, '.b', ax_top, ax_bottom)
            plt.pause(0.001)

    if visualize:
        plt.show()
    
    return X, t

