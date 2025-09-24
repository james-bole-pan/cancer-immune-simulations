import numpy as np
import pytest
import evalf_autograd as f
import eval_Jf_FiniteDifference_flatten as j

class Test_jacobian:
    """Regression tests for the evalf_autograd function"""
    
    def setup_method(self):
        """Setup common test parameters and initial conditions"""
        # Basic 2x2 grid initial state
        self.x0_basic = np.array([
            [[1.0e7, 1.0e7, 0.0029, 0.02, 0.015], [1.0e7, 1.0e7, 0.0029, 0.02, 0.015]],
            [[1.0e7, 1.0e7, 0.0029, 0.02, 0.015], [1.0e7, 1.0e7, 0.0029, 0.02, 0.015]]
        ])
        
        # Standard parameters from paper
        self.p_standard = f.Params(
            lc=0.5,        # lambda_c
            tc=5e7,        # theta_c
            nc=2,          # n_c
            k8=3,          # kappa_8
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

    def test_plot_jacobian(self):
        result = j.eval_Jf_FiniteDifference(f.evalf_autograd, self.x0_basic, self.p_standard, self.u_standard)
        import matplotlib.pyplot as plt
        plt.imshow(np.log10(np.abs(result[0]) + 1e-12), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Log10 Jacobian Heatmap')
        plt.show()

        assert True

    def test_high_drug_decreases_P_jacobian(self):
        result = j.eval_Jf_FiniteDifference(f.evalf_autograd, self.x0_basic, self.p_standard, self.u_standard)

        # test that the 5th variable has a negative effect on the 4th variable and zero effect on the first three
        Jf = result[0]
        dynamics = Jf[3, 4]
        print(dynamics)
        assert np.all(dynamics < 0), "Increasing drug amount should decrease the response to receptor changes"
        cross_effects = Jf[0:3, 4]
        assert np.all(cross_effects == 0), "Drug amount should not affect cancer, T8, or immune cells directly"

    def test_output_nonsingularity(self):
        """Test that output is nonsingular for most input values"""

        perturbations = [np.exp(10 * (np.random.rand(len(self.p_standard.tuple())) - 0.5)) for _ in range(1)]

        def perturb_p(perturbation):
            params = [self.p_standard.tuple()[i] * perturbation[i] for i in range(len(perturbation))]
            # Round the last three parameters to the nearest integer
            for i in range(-3, 0):
                params[i] = self.p_standard.tuple()[i]
            return f.Params(*params)

        print("TEST TEST")

        results = []
        for perturbation in perturbations:
            perturbed_params = perturb_p(perturbation)
            print(1)
            jacobian = j.eval_Jf_FiniteDifference(f.evalf_autograd, self.x0_basic, perturbed_params, self.u_standard)
            print(2)
            det_jacobian = np.linalg.det(jacobian[0])
            print(3)
            is_nonsingular = abs(det_jacobian) > 0.001
            print(4)
            results.append(is_nonsingular)
        
        is_good = np.mean(results)

        assert is_good > 0.95, "Most Jacobians should be nonsingular"

    def test_output_sparsity(self):
        """Test that at most 13 entries per row/column are nonzero"""

        result = j.eval_Jf_FiniteDifference(f.evalf_autograd, self.x0_basic, self.p_standard, self.u_standard)
        Jf = result[0]
        
        max_nonzero_per_row = np.max(np.sum(Jf != 0, axis=1))
        max_nonzero_per_col = np.max(np.sum(Jf != 0, axis=0))
        
        assert max_nonzero_per_row <= 13, "Each row should have at most 13 non-zero entries"
        assert max_nonzero_per_col <= 13, "Each column should have at most 13 non-zero entries"

    def test_output_shape_consistency(self):
        """Test that output shape always matches input shape"""
        # Test different grid sizes
        test_shapes = [
            (1, 1, 5),  # Single cell
            (2, 2, 5),  # 2x2 grid
            (3, 3, 5),  # 3x3 grid
        ]
        
        for rows, cols, vars in test_shapes:
            x_test = np.ones((rows, cols, vars)) * 1e6  # Initialize with reasonable values
            x_test[:, :, 2:] *= 1e-3  # Scale down the last 3 variables
            
            p_test = f.Params(
                lc=0.5, tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10, dc=0.18, D_c=0.01,
                lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
                ligt8=2.5e-8, dig=18, D_ig=0.01,
                mu_a=0.03, da=0.05, D_a=0.01, rows=rows, cols=cols
            )
            
            result = j.eval_Jf_FiniteDifference(f.evalf_autograd, x_test, p_test, self.u_standard)
            
            expected_dim = rows * cols * vars
            assert result[0].shape == (expected_dim, expected_dim), f"Shape mismatch for {rows}x{cols} grid"
            assert np.all(np.isfinite(result[0])), "All outputs should be finite"

if __name__ == "__main__":
    # Run tests if script is executed directly
    test_instance = Test_jacobian()
    test_instance.setup_method()
    
    print("Running evalf_autograd regression tests...")
    
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    for test_method in test_methods:
        try:
            getattr(test_instance, test_method)()
            print(f"✓ {test_method}")
        except Exception as e:
            print(f"✗ {test_method}: {e}")
    
    print("Tests completed!")
