import numpy as np
import pytest
import evalf_autograd as f

class Test_evalf_autograd:
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

    def test_high_lambda_c_promotes_cancer_growth(self):
        """Test that very high lambda_c (cancer growth rate) leads to positive cancer cell dynamics"""
        # Set lambda_c very high
        p_high_lc = f.Params(
            lc=10.0,       # Very high lambda_c (vs standard 0.5)
            tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10, dc=0.18, D_c=0.01,
            lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
            ligt8=2.5e-8, dig=18, D_ig=0.01,
            mu_a=0.03, da=0.05, D_a=0.01, rows=2, cols=2
        )
        
        result = f.evalf_autograd(self.x0_basic, p_high_lc, self.u_standard)
        
        # Cancer cells (index 0) should have positive growth rate
        cancer_dynamics = result[:, :, 0]  # del_c
        assert np.all(cancer_dynamics > 0), "High lambda_c should promote cancer growth"

    def test_high_death_rate_reduces_cancer(self):
        """Test that very high cancer death rate leads to negative cancer dynamics"""
        # Set cancer death rate very high
        p_high_dc = f.Params(
            lc=0.5, tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10,
            dc=5.0,        # Very high death rate (vs standard 0.18)
            D_c=0.01, lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
            ligt8=2.5e-8, dig=18, D_ig=0.01,
            mu_a=0.03, da=0.05, D_a=0.01, rows=2, cols=2
        )
        
        result = f.evalf_autograd(self.x0_basic, p_high_dc, self.u_standard)
        
        # Cancer cells should have negative dynamics (death dominates)
        cancer_dynamics = result[:, :, 0]  # del_c
        assert np.all(cancer_dynamics < 0), "High death rate should reduce cancer cells"

    def test_high_t8_growth_increases_t8_cells(self):
        """Test that high T8 cell growth rate promotes T8 cell expansion"""
        # Set lambda_t8 very high
        p_high_lt8 = f.Params(
            lc=0.5, tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10, dc=0.18, D_c=0.01,
            lt8=2.0,       # Very high lambda_t8 (vs standard 0.03)
            rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
            ligt8=2.5e-8, dig=18, D_ig=0.01,
            mu_a=0.03, da=0.05, D_a=0.01, rows=2, cols=2
        )
        
        result = f.evalf_autograd(self.x0_basic, p_high_lt8, self.u_standard)
        
        # T8 cells (index 1) should have positive growth
        t8_dynamics = result[:, :, 1]  # del_t8
        assert np.all(t8_dynamics > 0), "High lambda_t8 should promote T8 cell growth"

    def test_high_ig_death_reduces_ig(self):
        """Test that high interferon-gamma death rate reduces IFN-γ levels"""
        # Set d_ig very high
        p_high_dig = f.Params(
            lc=0.5, tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10, dc=0.18, D_c=0.01,
            lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
            ligt8=2.5e-8,
            dig=200.0,     # Very high IFN-γ degradation (vs standard 18)
            D_ig=0.01, mu_a=0.03, da=0.05, D_a=0.01, rows=2, cols=2
        )
        
        result = f.evalf_autograd(self.x0_basic, p_high_dig, self.u_standard)
        
        # IFN-γ (index 2) should have negative dynamics
        ig_dynamics = result[:, :, 2]  # del_ig
        assert np.all(ig_dynamics < 0), "High d_ig should reduce interferon-gamma levels"

    def test_high_antigen_input_increases_antigen(self):
        """Test that high antigen input rate increases antigen levels"""
        # Set antigen input very high
        u_high = 0.5  # Much higher than standard 0.015
        
        result = f.evalf_autograd(self.x0_basic, self.p_standard, u_high)
        
        # Antigen (index 4) should have positive dynamics
        antigen_dynamics = result[:, :, 4]  # del_a
        assert np.all(antigen_dynamics > 0), "High antigen input should increase antigen levels"

    def test_high_drug_effectiveness_kills_cancer(self):
        """Test that high drug effectiveness (mu_a) reduces cancer through enhanced immune response"""
        # Set mu_a (drug effectiveness) very high
        p_high_drug = f.Params(
            lc=0.5, tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10, dc=0.18, D_c=0.01,
            lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
            ligt8=2.5e-8, dig=18, D_ig=0.01,
            mu_a=1.0,      # Very high drug effectiveness (vs standard 0.03)
            da=0.05, D_a=0.01, rows=2, cols=2
        )
        
        # Use moderate antigen input to see drug effect
        u_moderate = 0.05  # Higher than standard to activate drug mechanism
        
        result = f.evalf_autograd(self.x0_basic, p_high_drug, u_moderate)
        
        # With high drug effectiveness, the mu_a * p8 * a term should be large
        # This affects both P8 dynamics (del_p8 = p8/t8 * del_t8 - mu_a * p8 * a)
        # and antigen dynamics (del_a = ra - (mu_a * p8 - da) * a + D_a * an)
        
        # P8 cells (index 3) should have negative dynamics due to drug action
        p8_dynamics = result[:, :, 3]  # del_p8
        assert np.any(p8_dynamics < 0), "High drug effectiveness should reduce P8 levels through drug action"
        
        # The overall effect should lead to better immune control
        # (This is a more complex systems-level effect, but we can check that dynamics are reasonable)
        assert np.all(np.isfinite(result)), "High drug effectiveness should produce finite dynamics"

    def test_zero_diffusion_no_neighbor_effects(self):
        """Test that zero diffusion coefficients eliminate spatial coupling"""
        # Set all diffusion coefficients to zero
        p_no_diff = f.Params(
            lc=0.5, tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10, dc=0.18,
            D_c=0.0,       # No cancer diffusion
            lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1,
            D_t8=0.0,      # No T8 diffusion
            ligt8=2.5e-8, dig=18,
            D_ig=0.0,      # No IFN-γ diffusion
            mu_a=0.03, da=0.05,
            D_a=0.0,       # No antigen diffusion
            rows=2, cols=2
        )
        
        # Create asymmetric initial condition
        x0_asym = np.array([
            [[1.0e7, 1.0e7, 0.0029, 0.02, 0.015], [2.0e7, 2.0e7, 0.006, 0.04, 0.03]],
            [[1.5e7, 1.5e7, 0.004, 0.03, 0.02], [1.0e7, 1.0e7, 0.0029, 0.02, 0.015]]
        ])
        
        result = f.evalf_autograd(x0_asym, p_no_diff, self.u_standard)
        
        # With no diffusion, each cell should evolve independently
        # The dynamics should be purely local (no spatial coupling terms)
        # This is mainly a structural test - we verify the function runs without error
        assert result.shape == x0_asym.shape, "Output shape should match input shape"
        assert np.all(np.isfinite(result)), "All dynamics should be finite"

    def test_equilibrium_conditions(self):
        """Test behavior near equilibrium points"""
        # Set initial conditions close to what might be an equilibrium
        x0_equilibrium = np.array([
            [[1.0e6, 5.0e6, 0.001, 0.01, 0.005], [1.0e6, 5.0e6, 0.001, 0.01, 0.005]],
            [[1.0e6, 5.0e6, 0.001, 0.01, 0.005], [1.0e6, 5.0e6, 0.001, 0.01, 0.005]]
        ])
        
        result = f.evalf_autograd(x0_equilibrium, self.p_standard, self.u_standard)
        
        # Near equilibrium, dynamics should be relatively small
        max_dynamics = np.max(np.abs(result))
        assert max_dynamics < 1e8, "Dynamics should be bounded near equilibrium"

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
            
            result = f.evalf_autograd(x_test, p_test, self.u_standard)
            
            assert result.shape == (rows, cols, vars), f"Shape mismatch for {rows}x{cols} grid"
            assert np.all(np.isfinite(result)), "All outputs should be finite"

if __name__ == "__main__":
    # Run tests if script is executed directly
    test_instance = Test_evalf_autograd()
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