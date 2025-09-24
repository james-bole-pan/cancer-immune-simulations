import numpy as np
import pytest
from autograd import jacobian


# --- Helper Class and Functions for Testing ---

# A placeholder for the 'p' object to make the test runnable.
# You will replace this with your actual 'p' class.
class Parameters:
    def __init__(self, values):
        assert len(values) == 19, "Parameters tuple must have 19 values."
        self._values = values

    def tuple(self):
        return tuple(self._values)

def evalf(x, p, u):
    """
    Evaluates the system of equations for a given state x.
    This version handles both 1D and 2D grid inputs and uses autograd.numpy.
    """
    shape = np.shape(x)
    assert shape[-1] == 5, "x must have five components in the last dimension"
    
    if len(shape) == 2:
        grid_x = shape[0]
        grid_y = 1
        x = np.reshape(x, (grid_x, grid_y, shape[-1]))
    elif len(shape) == 3:
        grid_x = shape[0]
        grid_y = shape[1]
    else:
        raise ValueError("Input 'x' must be a 2D or 3D array.")

    (lc, tc, nc, k8, ng, ki, dc, D_c, lt8, rl, kq, dt8, D_t8, ligt8, dig, D_ig, mu_a, da, D_a) = p.tuple()
    ra = u
    f_val = np.zeros_like(x)

    for i in range(grid_x):
        for j in range(grid_y):
            c, t8, ig, p8, a = x[i, j, :]
            
            neighbors = np.zeros(5)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if not (di == 0 and dj == 0):
                        ni, nj = i + di, j + dj
                        
                        if 0 <= ni < grid_x and 0 <= nj < grid_y:
                            neighbors += x[ni, nj, :]

            cn, t8n, ign, _, an = neighbors
            
            del_c = (lc / (1 + (c / tc)**nc) - (k8 * t8 + ng * ig / (ig + ki)) - dc) * c + D_c * cn
            del_t8 = (lt8 / (1 + rl * p8 * c / kq) - dt8) * t8 + D_t8 * t8n
            del_ig = ligt8 * t8 - dig * ig + D_ig * ign
            del_p8 = p8 / t8 * del_t8 - mu_a * p8 * a
            del_a = ra - (mu_a * p8 - da) * a + D_a * an
            
            f_val[i, j, :] = np.array([del_c, del_t8, del_ig, del_p8, del_a])

    return f_val

def numerical_jacobian(eval_func, x, p, u, h=1e-6):
    """
    Computes a numerical approximation of the Jacobian of the eval_func
    using the finite difference method.
    """
    # Flatten the input array for easier indexing
    x_flat = x.flatten()
    n_states = x_flat.size
    
    # Initialize the Jacobian matrix
    jacobian = np.zeros((n_states, n_states))
    
    # Evaluate the function at the unperturbed point
    f0 = eval_func(x, p, u).flatten()
    
    for i in range(n_states):
        # Perturb a single state variable
        x_perturbed = x_flat.copy()
        x_perturbed[i] += h
        
        # Reshape to the original array shape for eval_func
        x_perturbed_reshaped = np.reshape(x_perturbed, x.shape)
        
        # Evaluate the function at the perturbed point
        f1 = eval_func(x_perturbed_reshaped, p, u).flatten()
        
        # Compute the i-th column of the Jacobian
        jacobian[:, i] = (f1 - f0) / h
        
    return jacobian

def analytical_jacobian(x, p, u):
    """
    Computes the analytical Jacobian using autograd for a multi-cell grid.
    The input 'x' can be a 1D or 2D array.
    """
    # Define a new function that takes a flattened array and returns a flattened array,
    # as required by autograd's jacobian function.
    original_shape = x.shape
    def evalf_for_autograd(x_flat_autograd):
        x_reshaped = np.reshape(x_flat_autograd, original_shape)
        result_reshaped = evalf(x_reshaped, p, u)
        return np.flatten(result_reshaped)

    # Compute the Jacobian using autograd
    # The result will be a function. Call it with the current state 'x'.
    return jacobian(evalf_for_autograd)(np.flatten(x))

# --- Main Test Function ---

def test_jacobian_with_numerical_approximation():
    """
    Main test function that compares the analytical Jacobian against a
    numerical approximation for a simple 1D grid.
    """
    # Define a set of random but consistent parameters for the test
    p_values = np.random.rand(19) * 10 
    p = Parameters(p_values)
    u = 0.015

    # Define a simple 1D grid input for testing (shape: 1x1x5)
    # The Jacobian for this case will be a 5x5 matrix
    x_test = np.array([[[1.0, 0.5, 0.2, 0.8, 0.1]]])

    # 1. Compute the numerical Jacobian
    numerical_J = numerical_jacobian(evalf, x_test, p, u, h=1e-6)
    
    # 2. Compute the analytical Jacobian
    analytical_J = analytical_jacobian(x_test, p, u)
    
    # 3. Compare the two matrices
    
    # The assertion uses numpy's allclose to account for floating-point errors.
    # The tolerance (atol) may need to be adjusted depending on the function's sensitivity.
    assert np.allclose(numerical_J, analytical_J, atol=1e-5), "Analytical and numerical Jacobians do not match."

    # --- Test for a multi-cell grid ---
    # The Jacobian for this case will be a 20x20 matrix
    x_test_multi = np.array([
        [[1.0, 0.5, 0.2, 0.8, 0.1], [1.0, 0.5, 0.2, 0.8, 0.1]],
        [[1.0, 0.5, 0.2, 0.8, 0.1], [1.0, 0.5, 0.2, 0.8, 0.1]]
    ])
    
    numerical_J_multi = numerical_jacobian(evalf, x_test_multi, p, u, h=1e-6)
    analytical_J_multi = analytical_jacobian(x_test_multi, p, u)
    
    assert np.allclose(numerical_J_multi, analytical_J_multi, atol=1e-5), "Analytical and numerical Jacobians for multi-cell grid do not match."
