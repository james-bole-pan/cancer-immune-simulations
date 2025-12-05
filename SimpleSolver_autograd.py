import autograd.numpy as anp
import numpy as np
from VisualizeState import VisualizeState
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

def SimpleSolver_autograd(eval_f, x_start, p, eval_u,
                          NumIter, w=1.0,
                          visualize=True,
                          gif_file_name="State_visualization_autograd.gif"):
    """
    Autograd-compatible version of SimpleSolver.

    Key difference:
    - We DO NOT preallocate X and write into X[:, n].
      (This breaks autograd because X would be a NumPy ndarray)
    - Instead, we store each step as a separate anp array in a Python list.

    Args
    ----
    eval_f: function returning dx/dt
    x_start: column vector initial condition
    p: parameter object
    eval_u: input function u(t)
    NumIter: number of Euler steps
    w: step size
    visualize: whether to generate gif
    """

    #("running SimpleSolver_autograd (autograd-safe)...")

    NumIter = int(NumIter)

    # INITIAL CONDITION
    x0 = x_start.reshape(-1, 1)
    states = [x0]                       # autograd-safe storage (list of anp arrays)
    times  = [0.0]

    # FORWARD EULER LOOP (autograd-safe)
    x = x0
    t = 0.0
    for n in range(NumIter):
        u = eval_u(t)
        f = eval_f(x, p, u)             # f is an autograd array
        x = x + w * f                   # autograd-safe (no in-place writes)
        t = t + w

        states.append(x)
        times.append(t)

    # Convert list-of-arrays → stacked result AFTER computation
    # This is now safe because no autograd ops require backprop through stacking.
    X = anp.hstack(states)
    t_vec = anp.array(times)

    # -----------------------------
    # Visualization (optional)
    # -----------------------------
    if visualize:
        X_np = np.array(X, dtype=float)  # safe: only for visualization
        t_np = np.array(t_vec, dtype=float)

        if X_np.shape[0] > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax = (ax1, ax2)
        else:
            fig, ax = plt.subplots(1, 1)
            ax = (ax,)

        plt.tight_layout(pad=3.0)
        ani = animation.FuncAnimation(
            fig,
            partial(VisualizeState, t=t_np, X=X_np, ax=ax),
            frames=X_np.shape[1],
            repeat=False,
            interval=100
        )
        ani.save(gif_file_name, writer="pillow")

    return X, t_vec
