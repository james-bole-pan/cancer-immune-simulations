# %%
# 1D wrapper for evalf_autograd function
import numpy as np
try:
    import autograd.numpy as anp
except ImportError:
    import numpy as anp
import evalf_autograd as f

def evalf_autograd_1dwrapper(x_1d, p: f.Params, u):
    """
    1D wrapper for evalf_autograd function.
    
    Parameters:
    -----------
    x_1d : anp.ndarray with shape (rows * cols * 5,)
        Flattened state vector where each group of 5 consecutive elements
        represents [c, t8, ig, p8, a] for one grid cell
    p : f.Params
        Parameter object containing model parameters including rows and cols
    u : float
        Drug input concentration
    
    Returns:
    --------
    dx_dt_1d : anp.ndarray with shape (rows * cols * 5,)
        Flattened derivatives vector with same structure as input
    
    Notes:
    ------
    The 1D array is expected to be structured as:
    [c_00, t8_00, ig_00, p8_00, a_00, c_01, t8_01, ig_01, p8_01, a_01, ...]
    where the first index is row, second is column, and the 5 variables are:
    c (cancer cells), t8 (T8 cells), ig (interferon-gamma), p8 (P8 cells), a (antigen)
    """
    
    # Get grid dimensions from parameters
    rows = p.rows
    cols = p.cols
    
    # Validate input dimensions
    expected_length = rows * cols * 5
    if len(x_1d) != expected_length:
        raise ValueError(f"Input x_1d must have length {expected_length} for a {rows}x{cols} grid, "
                        f"but got length {len(x_1d)}")
    
    # Reshape 1D array to 3D format (rows, cols, 5)
    x_3d = x_1d.reshape((rows, cols, 5))
    
    # Call the original evalf_autograd function
    dx_dt_3d = f.evalf_autograd(x_3d, p, u)
    
    # Flatten the result back to 1D
    dx_dt_1d = dx_dt_3d.flatten()
    
    return dx_dt_1d

def reshape_1d_to_3d(x_1d, rows, cols):
    """
    Helper function to reshape 1D state vector to 3D format.
    
    Parameters:
    -----------
    x_1d : array_like
        1D state vector
    rows, cols : int
        Grid dimensions
        
    Returns:
    --------
    x_3d : ndarray with shape (rows, cols, 5)
        3D state array
    """
    return x_1d.reshape((rows, cols, 5))

def reshape_3d_to_1d(x_3d):
    """
    Helper function to reshape 3D state array to 1D format.
    
    Parameters:
    -----------
    x_3d : array_like with shape (rows, cols, 5)
        3D state array
        
    Returns:
    --------
    x_1d : ndarray
        1D state vector
    """
    return x_3d.flatten()

# Example usage and testing
if __name__ == "__main__":
    print("Testing evalf_autograd_1dwrapper...")
    
    # Create test parameters for a 2x2 grid
    p_test = f.Params(
        lc=0.5, tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10, dc=0.18, D_c=0.01,
        lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
        ligt8=2.5e-8, dig=18, D_ig=0.01, mu_a=0.03, da=0.05, D_a=0.01,
        rows=2, cols=2
    )
    
    # Create test 1D input (2x2 grid = 20 elements total)
    x_1d_test = np.array([
        # Cell (0,0): [c, t8, ig, p8, a]
        1.0e7, 1.0e7, 0.0029, 0.02, 0.015,
        # Cell (0,1)
        1.0e7, 1.0e7, 0.0029, 0.02, 0.015,
        # Cell (1,0)
        1.0e7, 1.0e7, 0.0029, 0.02, 0.015,
        # Cell (1,1)
        1.0e7, 1.0e7, 0.0029, 0.02, 0.015
    ])
    
    u_test = 0.015
    
    try:
        # Test the wrapper
        result_1d = evalf_autograd_1dwrapper(x_1d_test, p_test, u_test)
        
        print(f"✓ Input shape: {x_1d_test.shape}")
        print(f"✓ Output shape: {result_1d.shape}")
        print(f"✓ All derivatives finite: {np.all(np.isfinite(result_1d))}")
        
        # Compare with direct 3D version
        x_3d_test = x_1d_test.reshape((2, 2, 5))
        result_3d_direct = f.evalf_autograd(x_3d_test, p_test, u_test)
        result_3d_flattened = result_3d_direct.flatten()
        
        # Check if results match
        if np.allclose(result_1d, result_3d_flattened):
            print("✓ Results match direct 3D computation!")
        else:
            print("✗ Results don't match direct 3D computation")
            print(f"Max difference: {np.max(np.abs(result_1d - result_3d_flattened))}")
        
        print(f"Sample derivatives (first 5): {result_1d[:5]}")
        
        # Test single cell case
        print("\nTesting single cell (1x1 grid)...")
        p_single = f.Params(
            lc=0.5, tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10, dc=0.18, D_c=0.01,
            lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
            ligt8=2.5e-8, dig=18, D_ig=0.01, mu_a=0.03, da=0.05, D_a=0.01,
            rows=1, cols=1
        )
        
        x_1d_single = np.array([1.0e7, 1.0e7, 0.0029, 0.02, 0.015])
        result_1d_single = evalf_autograd_1dwrapper(x_1d_single, p_single, u_test)
        
        print(f"✓ Single cell input shape: {x_1d_single.shape}")
        print(f"✓ Single cell output shape: {result_1d_single.shape}")
        print(f"✓ Single cell derivatives finite: {np.all(np.isfinite(result_1d_single))}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()