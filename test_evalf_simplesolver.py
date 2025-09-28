import numpy as np
import matplotlib.pyplot as plt
import evalf_autograd as f
import evalf_autograd_1dwrapper as wrapper
from SimpleSolver import SimpleSolver
import copy

class TestSimpleSolver:
    """Test evalf_autograd_1dwrapper with SimpleSolver for analytically verifiable cases"""
    
    def __init__(self):
        self.setup_test_cases()
    
    def setup_test_cases(self):
        """Setup simple test cases with known analytical behavior"""
        
        # Test Case 1: Single cell with no interactions (pure decay)
        self.p_pure_decay = f.Params(
            lc=0.0,      # No cancer growth
            tc=1e10,     # Very high carrying capacity (no effect)
            nc=1,        # Linear
            k8=0.0,      # No T8 killing
            ng=0.0,      # No IFN-γ effect
            ki=1e10,     # Very high half-saturation (no effect)
            dc=0.1,      # Cancer decay rate
            D_c=0.0,     # No diffusion
            lt8=0.0,     # No T8 growth
            rl=0.0,      # No P8 inhibition
            kq=1e10,     # Very high half-saturation (no effect)
            dt8=0.1,     # T8 decay rate
            D_t8=0.0,    # No diffusion
            ligt8=0.0,   # No IFN-γ production
            dig=0.1,     # IFN-γ decay rate
            D_ig=0.0,    # No diffusion
            mu_a=0.0,    # No drug effect
            da=0.1,      # drug decay rate
            D_a=0.0,     # No diffusion
            rows=1, 
            cols=1
        )
        
    def constant_input(self, u_val):
        """Create constant input function"""
        def input_func(t):
            return u_val
        return input_func

    def test_pure_decay(self, w=1.0, num_iter=50):
        """Test pure exponential decay - should follow x(t) = x0 * exp(-decay_rate * t)"""
        
        print("="*60)
        print("TEST 1: Pure Exponential Decay")
        print("="*60)
        print("Expected behavior: All variables should decay exponentially")
        print("Analytical solution: x(t) = x0 * exp(-decay_rate * t)")
        
        # Initial condition: single cell with some values
        x0_1d = np.array([100, 90, 80, 70, 60])  # [c, t8, ig, p8, a]
        
        # No input
        u_func = self.constant_input(0.0)
        
        # Create wrapper function for SimpleSolver
        def eval_f_wrapper(x, p, u):
            return wrapper.evalf_autograd_1dwrapper(x.flatten(), p, u)
        
        X, t = SimpleSolver(
            eval_f_wrapper, 
            x0_1d, 
            self.p_pure_decay, 
            u_func, 
            num_iter, 
            w=w, 
            visualize=True,
            gif_file_name=f"test_evalf_output_figures/pure_decay_w_{w}.gif"
        )

        # Analytical verification
        print("\n" + "-"*50)
        print("ANALYTICAL VERIFICATION:")
        print("-"*50)
        
        # Calculate analytical solution at final time
        final_time = t[-1]
        decay_rates = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # dc, dt8, dig, dt8 (for p8), da
        analytical_final = x0_1d * np.exp(-decay_rates * final_time)
        
        # Get numerical solution at final time
        numerical_final = X[:, -1]
        
        print(f"Final simulation time: {final_time:.3f}")
        print(f"Initial state: {x0_1d}")
        print(f"Analytical final: {analytical_final}")
        print(f"Numerical final:  {numerical_final}")
        
        # Calculate errors
        absolute_error = np.abs(numerical_final - analytical_final)
        relative_error = absolute_error / np.abs(analytical_final)
        max_relative_error = np.max(relative_error)
        
        print(f"\nAbsolute errors: {absolute_error}")
        print(f"Relative errors: {relative_error}")
        print(f"Max relative error: {max_relative_error:.6f} ({max_relative_error*100:.4f}%)")
        
        # Assert the solutions are close (within 1% relative error)
        tolerance = 0.01  # 1% tolerance
        try:
            assert max_relative_error < tolerance, f"Maximum relative error {max_relative_error:.6f} exceeds tolerance {tolerance}"
            print(f"\n✅ ANALYTICAL TEST PASSED: All variables within {tolerance*100}% of analytical solution")
        except AssertionError as e:
            print(f"\n❌ ANALYTICAL TEST FAILED: {e}")
            print(f"Consider reducing omega (w) for higher accuracy or increasing tolerance")
            raise
        
        return X, t, analytical_final, numerical_final

    def run_all_tests(self, total_simulation_time=15, omega=0.01):
        """Run all test cases"""
        print("Starting comprehensive SimpleSolver tests with evalf_autograd_1dwrapper")
        print("="*80)

        num_iterations = int(total_simulation_time / omega)
        
        print(f"Using omega = {omega} for {num_iterations} iterations over {total_simulation_time} time units")
        
        # Run tests
        self.test_pure_decay(w=omega, num_iter=num_iterations)

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        
if __name__ == "__main__":
    tester = TestSimpleSolver()

    total_simulation_time = 15.0
    omega = 0.5

    tester.run_all_tests(total_simulation_time = total_simulation_time, omega = omega)