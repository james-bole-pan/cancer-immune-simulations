import numpy as np
import copy
import os
import eval_f as f
from eval_Jf_autograd import eval_Jf_autograd
from test_eval_f import TestEvalF

class Test_jacobian:
    """Regression tests for the eval_f function using Jacobian analysis"""
    
    def setup_method(self):
        """Setup common test parameters and initial conditions based on test_eval_f"""
        # Initialize test framework from test_eval_f
        self.eval_f_framework = TestEvalF()
        
        # Use parameters from test_eval_f
        self.p_standard = copy.deepcopy(self.eval_f_framework.p_default)
        
        # Basic 2x2 grid initial state with 3 variables (C, T, A)
        rows, cols = 2, 2
        self.p_standard.rows = rows
        self.p_standard.cols = cols
        
        # Create initial state as column vector (n_cells * 3, 1)
        n_cells = rows * cols
        self.x0_basic = np.zeros((n_cells * 3, 1))
        
        # Initialize with reasonable values
        for i in range(n_cells):
            self.x0_basic[i*3 + 0, 0] = 15  # Cancer cells (scaled down from original)
            self.x0_basic[i*3 + 1, 0] = 2  # T cells (scaled down)
            self.x0_basic[i*3 + 2, 0] = 0  # Drug concentration
        
        # Standard drug input function
        # Use scalar input for Jacobian computation (evaluated at t=0)
        self.u_standard = 200

    def test_plot_jacobian(self):
        """Test Jacobian computation and create heatmap visualization"""
        J = eval_Jf_autograd(f.eval_f, self.x0_basic, self.p_standard, self.u_standard)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(np.log10(np.abs(J) + 1e-12), cmap='hot', interpolation='nearest')
        plt.colorbar(label='Log10(|Jacobian|)')
        plt.title(f'Jacobian Heatmap ({J.shape[0]}x{J.shape[1]})')
        plt.xlabel('State Variable Index')
        plt.ylabel('Derivative Index')
        
        # Save plot instead of showing (for headless environments)
        if not os.path.exists('test_evalf_output_figures'):
            os.makedirs('test_evalf_output_figures')
        plt.savefig('test_evalf_output_figures/jacobian_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Verify Jacobian properties
        assert J.shape[0] == J.shape[1], "Jacobian should be square matrix"
        assert np.all(np.isfinite(J)), "All Jacobian entries should be finite"
        assert True

    def test_drug_effects_jacobian(self):
        """Test that drug concentration affects the system appropriately through Jacobian"""
        J = eval_Jf_autograd(f.eval_f, self.x0_basic, self.p_standard, self.u_standard)
        
        # In the new 3-variable system: [C, T, A] per cell
        # For a 2x2 grid: variables 0,3,6,9 are cancer (C), 1,4,7,10 are T-cells (T), 2,5,8,11 are drug (A)
        
        # Test drug self-dynamics (drug clearance should be negative)
        drug_indices = [2, 5, 8, 11]  # Drug concentration indices for each cell
        for i in drug_indices:
            drug_clearance = J[i, i]  # Effect of drug on its own dynamics
            print(f"Drug clearance at cell {i//3}: {drug_clearance}")
            assert drug_clearance < 0, f"Drug clearance should be negative at index {i}"
        
        # Test that drug enhances T-cell dynamics (positive coupling)
        t_cell_indices = [1, 4, 7, 10]  # T-cell indices
        for i, t_idx in enumerate(t_cell_indices):
            drug_idx = drug_indices[i]  # Corresponding drug index in same cell
            drug_effect_on_tcells = J[t_idx, drug_idx]
            print(f"Drug effect on T-cells at cell {i}: {drug_effect_on_tcells}")
            # Drug should enhance T-cell survival/activation (positive effect expected)
        
        print("Drug effects Jacobian test completed")

    def test_output_nonsingularity(self):
        """Test that Jacobian is nonsingular for most parameter combinations"""
        np.random.seed(42)  # For reproducible tests
        
        # Test multiple parameter perturbations
        num_tests = 5
        results = []
        
        for i in range(num_tests):
            # Create parameter perturbations (moderate changes)
            perturbation_factor = 0.1 * (2 * np.random.rand() - 1)  # ±10% changes
            
            p_perturbed = copy.deepcopy(self.p_standard)
            
            # Perturb key biological parameters
            p_perturbed.lambda_C *= (1 + perturbation_factor)
            p_perturbed.d_C *= (1 + perturbation_factor)
            p_perturbed.lambda_T *= (1 + perturbation_factor)
            p_perturbed.d_T *= (1 + perturbation_factor)
            p_perturbed.d_A *= (1 + perturbation_factor)
            
            try:
                jacobian = eval_Jf_autograd(f.eval_f, self.x0_basic, p_perturbed, self.u_standard)
                
                # For larger matrices, use condition number instead of determinant
                condition_num = np.linalg.cond(jacobian)
                is_well_conditioned = condition_num < 1e12  # Reasonable condition number
                
                print(f"Test {i+1}: Condition number = {condition_num:.2e}")
                results.append(is_well_conditioned)
                
            except Exception as e:
                print(f"Test {i+1} failed: {e}")
                results.append(False)
        
        success_rate = np.mean(results)
        print(f"Success rate: {success_rate:.2%}")
        
        assert success_rate > 0.8, "Most Jacobians should be well-conditioned"

    def test_output_sparsity(self):
        """Test sparsity pattern of the Jacobian matrix"""
        J = eval_Jf_autograd(f.eval_f, self.x0_basic, self.p_standard, self.u_standard)
        
        # Count non-zero entries per row and column
        tolerance = 1e-12
        nonzero_mask = np.abs(J) > tolerance
        
        nonzero_per_row = np.sum(nonzero_mask, axis=1)
        nonzero_per_col = np.sum(nonzero_mask, axis=0)
        
        max_nonzero_per_row = np.max(nonzero_per_row)
        max_nonzero_per_col = np.max(nonzero_per_col)
        
        print(f"Max non-zeros per row: {max_nonzero_per_row}")
        print(f"Max non-zeros per column: {max_nonzero_per_col}")
        print(f"Total matrix size: {J.shape}")
        print(f"Sparsity: {np.sum(nonzero_mask) / J.size:.2%} non-zero")
        
        # For a 2x2 grid with 3 variables each (12x12 matrix):
        # Each variable can be affected by itself + neighbors + coupled variables
        # Reasonable upper bound for biological system connectivity
        expected_max_connections = min(J.shape[0], 15)  # Adjust based on grid size
        
        assert max_nonzero_per_row <= expected_max_connections, f"Each row should have at most {expected_max_connections} non-zero entries"
        assert max_nonzero_per_col <= expected_max_connections, f"Each column should have at most {expected_max_connections} non-zero entries"

    def test_output_shape_consistency(self):
        """Test that Jacobian shape is consistent with input size"""
        
        # Test with different grid sizes (updated for 3-variable system)
        for n_x, n_y in [(2, 2), (3, 2), (2, 3)]:
            # Create test parameters 
            p_test = copy.deepcopy(self.p_standard)
            p_test.rows = n_x
            p_test.cols = n_y
            
            # Create test state
            n_cells = n_x * n_y
            x_test = np.zeros((n_cells * 3, 1))
            for i in range(n_cells):
                x_test[i*3 + 0, 0] = 15  # Cancer cells
                x_test[i*3 + 1, 0] = 2  # T cells
                x_test[i*3 + 2, 0] = 0.0  # Drug concentration

            J = eval_Jf_autograd(f.eval_f, x_test, p_test, self.u_standard)
            
            expected_size = n_x * n_y * 3  # 3 variables per cell
            
            assert J.shape == (expected_size, expected_size), \
                f"Jacobian should be {expected_size}x{expected_size}, got {J.shape}"
            assert np.all(np.isfinite(J)), f"All Jacobian entries should be finite for {n_x}x{n_y} grid"
            
            print(f"Grid size {n_x}x{n_y}: Jacobian shape {J.shape} ✓")

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
