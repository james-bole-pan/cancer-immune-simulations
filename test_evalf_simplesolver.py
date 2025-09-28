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
            da=0.1,      # Antigen decay rate
            D_a=0.0,     # No diffusion
            rows=1, cols=1
        )
        
        # Test Case 2: Single cell with simple growth (logistic cancer growth only)
        self.p_logistic_growth = f.Params(
            lc=0.5,      # Cancer growth rate
            tc=1e8,      # Carrying capacity
            nc=1,        # Linear (no Hill effect)
            k8=0.0,      # No T8 killing
            ng=0.0,      # No IFN-γ effect
            ki=1e10,     # Very high half-saturation (no effect)
            dc=0.0,      # No additional decay
            D_c=0.0,     # No diffusion
            lt8=0.0,     # No T8 growth
            rl=0.0,      # No P8 inhibition
            kq=1e10,     # Very high half-saturation (no effect)
            dt8=0.1,     # T8 decay
            D_t8=0.0,    # No diffusion
            ligt8=0.0,   # No IFN-γ production
            dig=0.1,     # IFN-γ decay
            D_ig=0.0,    # No diffusion
            mu_a=0.0,    # No drug effect
            da=0.1,      # Antigen decay
            D_a=0.0,     # No diffusion
            rows=1, cols=1
        )
        
        # Test Case 3: Drug input only (constant antigen source)
        self.p_drug_only = f.Params(
            lc=0.0,      # No cancer growth
            tc=1e10,     # Very high carrying capacity
            nc=1,        # Linear
            k8=0.0,      # No T8 killing
            ng=0.0,      # No IFN-γ effect
            ki=1e10,     # Very high half-saturation
            dc=0.1,      # Cancer decay
            D_c=0.0,     # No diffusion
            lt8=0.0,     # No T8 growth
            rl=0.0,      # No P8 inhibition
            kq=1e10,     # Very high half-saturation
            dt8=0.1,     # T8 decay
            D_t8=0.0,    # No diffusion
            ligt8=0.0,   # No IFN-γ production
            dig=0.1,     # IFN-γ decay
            D_ig=0.0,    # No diffusion
            mu_a=0.0,    # No drug effect initially
            da=0.05,     # Antigen decay rate
            D_a=0.0,     # No diffusion
            rows=1, cols=1
        )
    
    def constant_input(self, u_val):
        """Create constant input function"""
        def input_func(t):
            return u_val
        return input_func
    
    def test_pure_decay(self, w_values=[1.0, 0.1, 0.01], num_iter=50):
        """Test pure exponential decay - should follow x(t) = x0 * exp(-decay_rate * t)"""
        
        print("="*60)
        print("TEST 1: Pure Exponential Decay")
        print("="*60)
        print("Expected behavior: All variables should decay exponentially")
        print("Analytical solution: x(t) = x0 * exp(-decay_rate * t)")
        
        # Initial condition: single cell with some values
        x0_1d = np.array([1e6, 1e5, 0.1, 0.01, 0.05])  # [c, t8, ig, p8, a]
        
        # No input
        u_func = self.constant_input(0.0)
        
        # Create wrapper function for SimpleSolver
        def eval_f_wrapper(x, p, u):
            return wrapper.evalf_autograd_1dwrapper(x.flatten(), p, u)
        
        results = {}
        
        for w in w_values:
            print(f"\nTesting with w = {w}")
            
            try:
                X, t = SimpleSolver(
                    eval_f_wrapper, 
                    x0_1d, 
                    self.p_pure_decay, 
                    u_func, 
                    num_iter, 
                    w=w, 
                    visualize=False,
                    gif_file_name=f"pure_decay_w_{w}.gif"
                )
                
                results[w] = {'X': X, 't': t}
                
                # Check final values
                final_values = X[:, -1]
                decay_rates = [0.1, 0.1, 0.1, 0.1, 0.1]  # dc, dt8, dig, implicit, da
                
                print(f"  Initial values: {x0_1d}")
                print(f"  Final values:   {final_values}")
                
                # Analytical solution at final time
                t_final = t[-1]
                analytical = x0_1d * np.exp(-np.array(decay_rates) * t_final)
                print(f"  Analytical:     {analytical}")
                print(f"  Relative error: {np.abs((final_values - analytical) / analytical)}")
                
            except Exception as e:
                print(f"  ERROR with w={w}: {e}")
        
        return results
    
    def test_logistic_growth(self, w_values=[0.01, 0.001], num_iter=100):
        """Test logistic growth - cancer should grow to carrying capacity"""
        
        print("="*60)
        print("TEST 2: Logistic Cancer Growth")
        print("="*60)
        print("Expected behavior: Cancer grows toward carrying capacity")
        print("Other variables should decay to zero")
        
        # Start with small cancer population
        x0_1d = np.array([1e5, 1e5, 0.1, 0.01, 0.05])  # [c, t8, ig, p8, a]
        
        # No input
        u_func = self.constant_input(0.0)
        
        def eval_f_wrapper(x, p, u):
            return wrapper.evalf_autograd_1dwrapper(x.flatten(), p, u)
        
        results = {}
        
        for w in w_values:
            print(f"\nTesting with w = {w}")
            
            try:
                X, t = SimpleSolver(
                    eval_f_wrapper, 
                    x0_1d, 
                    self.p_logistic_growth, 
                    u_func, 
                    num_iter, 
                    w=w, 
                    visualize=False,
                    gif_file_name=f"logistic_growth_w_{w}.gif"
                )
                
                results[w] = {'X': X, 't': t}
                
                # Check behavior
                cancer_trajectory = X[0, :]
                carrying_capacity = self.p_logistic_growth.tc
                
                print(f"  Initial cancer: {cancer_trajectory[0]:.2e}")
                print(f"  Final cancer:   {cancer_trajectory[-1]:.2e}")
                print(f"  Carrying cap:   {carrying_capacity:.2e}")
                print(f"  Growth achieved: {(cancer_trajectory[-1] / carrying_capacity * 100):.1f}%")
                
                # Check if growth is monotonic initially
                growth_phase = cancer_trajectory[:min(20, len(cancer_trajectory))]
                is_growing = np.all(np.diff(growth_phase) > 0)
                print(f"  Initially growing: {is_growing}")
                
            except Exception as e:
                print(f"  ERROR with w={w}: {e}")
        
        return results
    
    def test_drug_input_steady_state(self, w_values=[0.1, 0.01], num_iter=100):
        """Test constant drug input - antigen should reach steady state"""
        
        print("="*60)
        print("TEST 3: Constant Drug Input")
        print("="*60)
        print("Expected behavior: Antigen reaches steady state = r_a / d_a")
        print("Other variables decay to zero")
        
        # Start with small values
        x0_1d = np.array([1e5, 1e5, 0.01, 0.001, 0.001])  # [c, t8, ig, p8, a]
        
        # Constant drug input
        r_a = 0.1
        u_func = self.constant_input(r_a)
        
        def eval_f_wrapper(x, p, u):
            return wrapper.evalf_autograd_1dwrapper(x.flatten(), p, u)
        
        results = {}
        
        for w in w_values:
            print(f"\nTesting with w = {w}")
            
            try:
                X, t = SimpleSolver(
                    eval_f_wrapper, 
                    x0_1d, 
                    self.p_drug_only, 
                    u_func, 
                    num_iter, 
                    w=w, 
                    visualize=False,
                    gif_file_name=f"drug_input_w_{w}.gif"
                )
                
                results[w] = {'X': X, 't': t}
                
                # Check antigen steady state
                antigen_trajectory = X[4, :]  # Antigen is index 4
                da = self.p_drug_only.da
                expected_steady_state = r_a / da
                
                print(f"  Initial antigen: {antigen_trajectory[0]:.4f}")
                print(f"  Final antigen:   {antigen_trajectory[-1]:.4f}")
                print(f"  Expected steady: {expected_steady_state:.4f}")
                print(f"  Relative error:  {abs((antigen_trajectory[-1] - expected_steady_state) / expected_steady_state):.4f}")
                
                # Check if approaching steady state
                final_values = antigen_trajectory[-10:]
                is_converging = np.std(final_values) < 0.01 * np.mean(final_values)
                print(f"  Converging: {is_converging}")
                
            except Exception as e:
                print(f"  ERROR with w={w}: {e}")
        
        return results
    
    def test_omega_convergence(self):
        """Test convergence as omega decreases"""
        
        print("="*60)
        print("TEST 4: Omega Convergence Study")
        print("="*60)
        print("Testing how solutions change as w decreases")
        
        # Use the pure decay case
        x0_1d = np.array([1e6, 1e5, 0.1, 0.01, 0.05])
        u_func = self.constant_input(0.0)
        
        def eval_f_wrapper(x, p, u):
            return wrapper.evalf_autograd_1dwrapper(x.flatten(), p, u)
        
        w_values = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
        solutions = {}
        
        for w in w_values:
            print(f"Testing w = {w}")
            try:
                X, t = SimpleSolver(
                    eval_f_wrapper, 
                    x0_1d, 
                    self.p_pure_decay, 
                    u_func, 
                    50, 
                    w=w, 
                    visualize=False
                )
                solutions[w] = X[:, -1]  # Final solution
                print(f"  Final solution: {X[:, -1]}")
            except Exception as e:
                print(f"  ERROR: {e}")
        
        # Compare consecutive solutions
        print("\nConvergence analysis:")
        w_list = sorted(solutions.keys(), reverse=True)
        for i in range(len(w_list)-1):
            w1, w2 = w_list[i], w_list[i+1]
            if w1 in solutions and w2 in solutions:
                diff = np.linalg.norm(solutions[w1] - solutions[w2])
                rel_diff = diff / np.linalg.norm(solutions[w2])
                print(f"  ||x(w={w1}) - x(w={w2})|| = {diff:.6f} (relative: {rel_diff:.6f})")
        
        return solutions
    
    def run_all_tests(self):
        """Run all test cases"""
        print("Starting comprehensive SimpleSolver tests with evalf_autograd_1dwrapper")
        print("="*80)
        
        # Run tests
        results = {}
        results['pure_decay'] = self.test_pure_decay()
        results['logistic_growth'] = self.test_logistic_growth()
        results['drug_input'] = self.test_drug_input_steady_state()
        results['convergence'] = self.test_omega_convergence()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        
        return results

if __name__ == "__main__":
    tester = TestSimpleSolver()
    results = tester.run_all_tests()