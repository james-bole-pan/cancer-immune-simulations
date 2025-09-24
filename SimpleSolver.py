import numpy as np
from VisualizeState import VisualizeState
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from functools import partial

def SimpleSolver(eval_f, x_start, p, eval_u, NumIter, w=1, visualize=True, gif_file_name="State_visualization.gif"):
    """
    an extremely basic iterative algorithm 
    that can surprisingly potentially solve very different problems
    (although sometimes not reliably or efficiently):
    e.g. nonlinear problems, i.e. finding vector x such that f(x,p,u)=0
    e.g. steady state step response x of nonlinear ODEs dx/dt = f(x,p,u)%     
        and of some Differential Algebraic Equations (ADEs)
            0 = f_a(x,p,u)
        dx/dt = f_d(x,p,u)
        simply setting f(x,p,u) = [f_a(x,p,u) f_d(x,p,u)]'
    e.g. can also produce a possibly rough approximation
        of the dynamics leading to steady state
    e.g. and it can even solve optimization problems 
        it is basically steepest descent when f is grad(Loss)

    x_start        is the initial guess for the iteration
    p         is a structure containing all parameters needed to evaluate f( )
    eval_u    contains values of the input sources 
    eval_f    is a text string with the name of the function that evaluates f(x,p,u) 
    NumIter   is the desired number of iterations 
    visualize = True (optional) shows intermediate results
    gif_file_name = "State_visualization.gif" (optional) name of the gif file to save the visualization

    w  is is an optional (yet important) scalar parameter 
    w=1 is used as default when w is not given
    but order of magnitude smaller values (or negative) 
    may be necessary to prevent wild numerical instabilities,
    at the expense of progressively slower computation times. 
    Intepretations can be given for w when solving:
        nonlinear systems, w controls the convergence of the sequence
        ODEs and ADEs, w can be interpreted as "timestep size"
        optimization problems, w can be interpreted as "learning rate"

    EXAMPLE
    [X,t] = SimpleSolver(eval_f, x_start, p, eval_u, NumIter, w, visualize)
    """

    print("started running a simple solver, please wait...")

    NumIter = int(NumIter)
    X = np.zeros((len(x_start), NumIter + 1))
    t = np.zeros(NumIter + 1)

    X[:, 0] = np.reshape(x_start,[-1])
    t[0] = 0

    for n in range(NumIter):
        t[n+1] = t[n] + w  # this is usefull only when solving ODEs or ADEs to help relating interation indeces to times
        u = eval_u(t[n])
        f = eval_f(np.reshape(X[:,n],[-1,1]), p, u)
        X[:, n+1] = X[:, n] + w * f.reshape(X[:, n].shape)

    if visualize:
        if X.shape[0] > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax = (ax1, ax2)
        else:
            fig, ax = plt.subplots(1, 1)
            ax = (ax,)  # Ensure ax is always a tuple for consistency

        plt.tight_layout(pad=3.0)
        ani = animation.FuncAnimation(
            fig,
            partial(VisualizeState, t=t, X=X, ax=ax),
            frames=NumIter + 1,
            repeat=False,
            interval=100  # Adjust speed of animation here
        )

        ani.save(gif_file_name, writer="pillow")
        plt.show()
        
    return X, t