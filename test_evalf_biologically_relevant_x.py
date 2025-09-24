import numpy as np
import pytest
import evalf_autograd as f

class Test_evalf_autograd:
    """Regression tests for the evalf_autograd function"""
    
    def setup_method(self):
        """Setup common test parameters and initial conditions"""

        # read in biologically relevant initial state from file
        self.x0 = np.load("data/xenium5k_luad_5ch_grid50um.npy")
        assert self.x0.shape[2] == 5, "Expected 5 variables per grid cell"
        
        # Standard parameters from paper
        self.p_standard = f.Params(
            lc=0.5,        # lambda_c
            tc=5e7,        # theta_c
            nc=2,          # n_c
            k8=3e-7,       # kappa_8
            ng=0.1,        # n_g
            ki=10,         # K_i
            dc=0.18,       # d_c
            D_c=0.01,      # D_c
            lt8=0.03,      # lambda_t8
            rl=3e-7,       # rho_l
            kq=12.6,       # K_q
            dt8=0.1,       # d_t8
            D_t8=0.01,     # D_t8
            ligt8=2.5e-8,  # lambda_igt8
            dig=18,        # d_ig
            D_ig=0.01,     # D_ig
            mu_a=0.03,     # mu_a
            da=0.05,       # d_a
            D_a=0.01,      # D_a
            rows=2,        # rows in grid
            cols=2         # cols in grid
        )
        
        self.u_standard = 0.015

if __name__ == "__main__":
    # Run tests if script is executed directly
    test_instance = Test_evalf_autograd()
    test_instance.setup_method()
    
    print("Running evalf_autograd regression tests...")
    
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    x0 = test_instance.x0
    print(x0.shape)
    
    print("Tests completed!")