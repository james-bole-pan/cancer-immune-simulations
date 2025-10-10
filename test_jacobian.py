import numpy as np
import copy
import os
import eval_f as f
from eval_Jf_autograd import eval_Jf_autograd
from test_eval_f import TestEvalF
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

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

    def test_plot_jacobian_heatmap(self):
        """Compute Jacobian and save (1) sign heatmap and (2) log|J| heatmap, plus a combined figure."""
        
        print("Running test_plot_jacobian_heatmap...")
        J = eval_Jf_autograd(f.eval_f, self.x0_basic, self.p_standard, self.u_standard)

        assert J.shape[0] == J.shape[1], "Jacobian should be a square matrix."
        assert np.all(np.isfinite(J)), "All Jacobian entries should be finite."

        # --- Sign matrix (negative / ~zero / positive) ---
        tol = 1e-12
        sign_mat = np.zeros_like(J, dtype=int)
        sign_mat[J >  tol] =  1
        sign_mat[J < -tol] = -1

        # 3-color categorical map: negative / zero / positive
        cmap_sign = ListedColormap(["#2b6cb0", "#e2e8f0", "#e53e3e"])   # blue / light gray / red
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm_sign = BoundaryNorm(bounds, cmap_sign.N)

        # --- Log magnitude (add epsilon to avoid log(0)) ---
        eps = 1e-12
        logmag = np.log10(np.abs(J) + eps)
        cmap_mag = "hot"  # or "viridis", etc.

        # Output directory
        outdir = "test_evalf_output_figures"
        os.makedirs(outdir, exist_ok=True)

        # (A) Sign heatmap
        plt.figure(figsize=(9, 8))
        im0 = plt.imshow(sign_mat, cmap=cmap_sign, norm=norm_sign, interpolation="nearest")
        plt.title(f"Jacobian Sign Heatmap ({J.shape[0]}×{J.shape[1]})")
        plt.xlabel("State Variable Index (column: cause)")
        plt.ylabel("Derivative Index (row: effect)")
        legend_patches = [
            Patch(facecolor="#2b6cb0", edgecolor="none", label="Negative (< -tol)"),
            Patch(facecolor="#e2e8f0", edgecolor="none", label=f"Zero (|J| ≤ {tol:g})"),
            Patch(facecolor="#e53e3e", edgecolor="none", label="Positive (> tol)"),
        ]
        plt.legend(handles=legend_patches, loc="upper right", frameon=False)
        #plt.savefig(os.path.join(outdir, "jacobian_heatmap_signs.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # (B) Log-magnitude heatmap
        plt.figure(figsize=(9, 8))
        im1 = plt.imshow(logmag, cmap=cmap_mag, interpolation="nearest")
        cbar = plt.colorbar(im1)
        cbar.set_label("log10(|J| + ε)")
        plt.title(f"Jacobian Log-Magnitude Heatmap ({J.shape[0]}×{J.shape[1]})")
        plt.xlabel("State Variable Index (column: cause)")
        plt.ylabel("Derivative Index (row: effect)")
        #plt.savefig(os.path.join(outdir, "jacobian_heatmap_logmag.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # (C) Combined two-panel figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        imA = axes[0].imshow(sign_mat, cmap=cmap_sign, norm=norm_sign, interpolation="nearest")
        axes[0].set_title("Sign")
        axes[0].set_xlabel("Column (cause)")
        axes[0].set_ylabel("Row (effect)")
        axes[0].legend(handles=legend_patches, loc="upper right", frameon=False)

        imB = axes[1].imshow(logmag, cmap=cmap_mag, interpolation="nearest")
        axes[1].set_title("log10(|J| + ε)")
        axes[1].set_xlabel("Column (cause)")
        axes[1].set_ylabel("Row (effect)")
        fig.colorbar(imB, ax=axes[1])

        fig.suptitle(f"Jacobian Heatmaps ({J.shape[0]}×{J.shape[1]})", y=1.02, fontsize=12)
        fig.savefig(os.path.join(outdir, "jacobian_heatmap_combined.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def test_jacobian_value_direction(self):
        """Test that the Jacobian reflects the system appropriately"""
        print("Running test_jacobian_value_direction...")
        J = eval_Jf_autograd(f.eval_f, self.x0_basic, self.p_standard, self.u_standard)
  
        # --- Index helpers (interleaved [C,T,A] per cell) ---
        n = J.shape[0]
        assert n % 3 == 0, "State must be interleaved [C,T,A] per cell."
        N = n // 3
        C_idx = [3*k + 0 for k in range(N)]
        T_idx = [3*k + 1 for k in range(N)]
        A_idx = [3*k + 2 for k in range(N)]

        # Tolerances
        tol_neg  = 1e-12   # strictly negative (allow tiny numerical noise)
        tol_zero = 1e-12   # approximately zero
        tol_nonneg = 1e-12 # >= 0 with tolerance

        # 1) df_A/dA < 0  (drug self-clearance is negative)
        for a in A_idx:
            val = J[a, a]
            assert val < -tol_neg, f"Drug clearance should be negative: J[{a},{a}]={val:.3e}"

        # 2) df_A/dT ≈ 0  (no direct T → A coupling)
        for t in T_idx:
            for a in A_idx:
                val = J[a, t]
                assert np.isclose(val, 0.0, atol=tol_zero), (
                    f"Drug dynamics should not depend on T: J[{a},{t}]={val:.3e}"
                )

        # 3) df_A/dC ≈ 0  (no direct C → A coupling)
        for c in C_idx:
            for a in A_idx:
                val = J[a, c]
                assert np.isclose(val, 0.0, atol=tol_zero), (
                    f"Drug dynamics should not depend on C: J[{a},{c}]={val:.3e}"
                )

        # 4) df_T/dA ≥ 0 locally (drug enhances or does not harm T); same-cell pairing
        for k in range(N):
            t, a = T_idx[k], A_idx[k]
            val = J[t, a]
            assert val >= -tol_nonneg, (
                f"Local A→T coupling should be ≥0: J[{t},{a}]={val:.3e} (cell {k})"
            )

        # 5) df_T/dC ≥ 0 locally (cancer stimulates T activation); same-cell pairing
        for k in range(N):
            t, c = T_idx[k], C_idx[k]
            val = J[t, c]
            assert val >= -tol_nonneg, (
                f"Local C→T coupling should be ≥0: J[{t},{c}]={val:.3e} (cell {k})"
            )
            
        # 6) df_C/dT ≤ 0 locally (T cells kill cancer); same-cell pairing
        for k in range(N):
            c, t = C_idx[k], T_idx[k]
            val = J[c, t]
            assert val <= tol_nonneg, (
                f"Local T→C coupling should be ≤0: J[{c},{t}]={val:.3e} (cell {k})"
            )

        # 7) df_C/dA ≈ 0  (no direct A → C coupling)
        for k in range(N):
            c, a = C_idx[k], A_idx[k]
            val = J[c, a]
            assert np.isclose(val, 0.0, atol=tol_zero), (
                f"Cancer dynamics should not directly depend on A: J[{c},{a}]={val:.3e} (cell {k})"
            )

    def test_output_nonsingularity(self):
        """Test that Jacobian is nonsingular for most parameter combinations"""
        print("Running test_output_nonsingularity...")
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
                is_well_conditioned = condition_num < 1e6  # Reasonable condition number
                
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
        print("Running test_output_sparsity...")
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

        expected_max_connections = 7
        
        assert max_nonzero_per_row <= expected_max_connections, f"Each row should have at most {expected_max_connections} non-zero entries"
        assert max_nonzero_per_col <= expected_max_connections, f"Each column should have at most {expected_max_connections} non-zero entries"

if __name__ == "__main__":
    # Run tests if script is executed directly
    test_instance = Test_jacobian()
    test_instance.setup_method()
    
    print("Running evalf_autograd regression tests...")
    
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]

    total_tests = len(test_methods)
    failed_tests = 0
    
    for test_method in test_methods:
        try:
            getattr(test_instance, test_method)()
            print(f"✓ {test_method}")
        except Exception as e:
            print(f"✗ {test_method}: {e}")
            failed_tests += 1

    print("Tests completed!")
    print(f"Successful tests: {total_tests - failed_tests}, Failed tests: {failed_tests}; Success rate: {(total_tests - failed_tests) / total_tests:.2%}")
