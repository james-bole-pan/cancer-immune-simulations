import matplotlib.pyplot as plt
import numpy as np

def visualize_state(t, X, n, plottype, ax_top=None, ax_bottom=None):
    """
    Visualizes state components. If ax_top is provided, it updates the existing plots.
    """

    # Initialize the subplots only once
    if ax_top is None or ax_bottom is None:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1)
        fig.show()
    else:
        ax_bottom.clear()
    
    N = X.shape[0]  # Number of components in the solution/state

    # Top part shows the intermediate progress of all solution components vs iteration index
    ax_top.plot([t[n] for i in range(X.shape[0])], X[:, n], plottype, markersize=2)
    ax_top.set_xlabel('time or (iteration index)')
    ax_top.set_ylabel('x')
    
    if N > 1:
        # Bottom part shows all component values of the last solution
        ax_bottom.plot(range(1, N + 1), X[:, n], plottype, markersize=5)
        minX = np.min(X)
        maxX = np.max(X)
        if maxX == minX:
            if maxX == 0:
                minX, maxX = -1, 1
            else:
                minX, maxX = minX * 0.9, maxX * 1.1
    
        maxh = X.shape[0]
        if maxh == 1:
            maxh = 2
            
        ax_bottom.set_xlim(1, maxh)
        ax_bottom.set_ylim(minX, maxX)
        ax_bottom.set_xlabel('state components index')
        ax_bottom.set_ylabel('x')
    
    plt.draw()
    plt.pause(0.001)

    return ax_top, ax_bottom
