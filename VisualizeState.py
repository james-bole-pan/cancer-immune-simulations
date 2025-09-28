import matplotlib.pyplot as plt
import numpy as np

def VisualizeState(frame, t, X, ax):
    """
    Simple visualization function for SimpleSolver
    
    Parameters:
    -----------
    frame : int
        Current frame index
    t : array
        Time points
    X : array
        State trajectories (variables x time)
    ax : tuple of axes
        Matplotlib axes for plotting
    """
    
    # Clear previous plots
    for a in ax:
        a.clear()
    
    # Plot state evolution
    if len(ax) >= 2:
        # Plot first few variables
        ax[0].plot(t[:frame+1], X[0, :frame+1], 'r-', label='Cancer cells', linewidth=2)
        if X.shape[0] > 1:
            ax[0].plot(t[:frame+1], X[1, :frame+1], 'b-', label='T8 cells', linewidth=2)
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Cell Count')
        ax[0].set_title(f'Cell Populations (t = {t[frame]:.3f})')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Plot remaining variables
        if X.shape[0] > 2:
            for i in range(2, min(5, X.shape[0])):
                ax[1].plot(t[:frame+1], X[i, :frame+1], label=f'Variable {i}', linewidth=2)
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Concentration')
        ax[1].set_title('Other Variables')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
    else:
        # Single plot for all variables
        for i in range(min(5, X.shape[0])):
            ax[0].plot(t[:frame+1], X[i, :frame+1], label=f'Variable {i}', linewidth=2)
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Value')
        ax[0].set_title(f'All Variables (t = {t[frame]:.3f})')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
    
    return ax