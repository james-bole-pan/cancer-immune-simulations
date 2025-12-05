import autograd.numpy as anp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from VisualizeState import VisualizeState

def SimpleSolver_autograd(eval_f, x_start, p, eval_u, NumIter, w=1, visualize=True, gif_file_name="State_visualization.gif"):

    NumIter = int(NumIter)

    x0 = anp.reshape(anp.array(x_start), (-1,))
    N = x0.shape[0]

    X = anp.zeros((N, NumIter + 1))
    t = anp.zeros(NumIter + 1)

    X[:, 0] = x0
    t = anp.array(t)

    for n in range(NumIter):
        t[n+1] = t[n] + w
        u = eval_u(t[n])
        f = eval_f(anp.reshape(X[:, n], (-1, 1)), p, u)
        X[:, n+1] = X[:, n] + w * anp.reshape(f, X[:, n].shape)

    if visualize:
        if X.shape[0] > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax = (ax1, ax2)
        else:
            fig, ax = plt.subplots(1, 1)
            ax = (ax,)

        plt.tight_layout(pad=3.0)
        ani = animation.FuncAnimation(
            fig,
            partial(VisualizeState, t=anp.array(t), X=anp.array(X), ax=ax),
            frames=NumIter + 1,
            repeat=False,
            interval=100
        )

        ani.save(gif_file_name, writer="pillow")
        plt.close(fig)

    return X, t
