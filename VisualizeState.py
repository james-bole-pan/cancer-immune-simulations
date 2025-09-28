import matplotlib.pyplot as plt
import numpy as np

def VisualizeState(n, t, X, ax):
    """
    Visualizes state components. It can be used to visualize:
    1) the time domain evolution of the state during ODE integration
    2) or the intermediate progress of a Newton iteration

    INPUTS:
    n	     index of the current data to be added to the plot
    t 	     vector containing time stamps
             (or iteration indeces when used to visualize itermediate Newton iterations)
    X	     contains intermediate solutions as columns
    ax       A tuple containing the axes of figures

    EXAMPLES:
    VisualizeState(t,X,n,plottype);       % to visualize time domain evolution
    VisualizeState([1:1:k,X,k,plottype);  % to visualize Newton iteration progress
    """
    N = X.shape[0]  # number of components in the solution/state

    # Clear the axes for animation
    ax[0].cla()  # Clear the top plot
    ax[1].cla()  # Clear the bottom plot (if N > 1)

    # Top figure shows the intermediate progress of all solution components vs iteration index
    ax[0].plot(np.reshape(t[:n+1], [n+1]), np.reshape(X[:, :n+1], [N, n+1]).T, ".b")
    ax[0].set_xlabel("Time or Iteration Index")
    ax[0].set_ylabel("x")

    if N > 1:
        # Bottom part shows all component values of the current solution
        ax[1].plot(X[:, n], ".b")
        minX = np.min(X)
        maxX = np.max(X)
        if maxX == minX:
            if maxX == 0:
                minX = -1
                maxX = 1
            else:
                minX = min(minX * 0.9, minX * 1.1)
                maxX = max(minX * 0.9, minX * 1.1)

        maxh = X.shape[0]

        if maxh == 1:
            maxh = 2

        ax[1].set_xlim(0, maxh)
        ax[1].set_ylim(minX, maxX)
        ax[1].set_xlabel("State Components Index")
        ax[1].set_ylabel("x")
