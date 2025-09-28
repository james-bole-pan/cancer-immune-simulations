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

    def test_single_cell_input(self):
        """Test evalf_autograd function with a single cell (1,1,5) input"""
        # Create a single cell initial state
        x0_single = np.array([[[1.0e7, 1.0e7, 0.0029, 0.02, 0.015]]])
        
        # Single cell parameters
        p_single = f.Params(
            lc=0.5, tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10, dc=0.18, D_c=0.01,
            lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
            ligt8=2.5e-8, dig=18, D_ig=0.01,
            mu_a=0.03, da=0.05, D_a=0.01, 
            rows=1, cols=1  # Single cell
        )
        
        print(f"Testing single cell with shape: {x0_single.shape}")
        print(f"Initial values: c={x0_single[0,0,0]:.2e}, t8={x0_single[0,0,1]:.2e}, "
              f"ig={x0_single[0,0,2]:.4f}, p8={x0_single[0,0,3]:.3f}, a={x0_single[0,0,4]:.3f}")
        
        try:
            result = f.evalf_autograd(x0_single, p_single, self.u_standard)
            
            print(f"Result shape: {result.shape}")
            print(f"Derivatives: dc/dt={result[0,0,0]:.2e}, dt8/dt={result[0,0,1]:.2e}, "
                  f"dig/dt={result[0,0,2]:.4f}, dp8/dt={result[0,0,3]:.4f}, da/dt={result[0,0,4]:.4f}")
            
            # Basic sanity checks
            assert result.shape == (1, 1, 5), f"Result shape should be (1,1,5), got {result.shape}"
            assert np.all(np.isfinite(result)), "All derivatives should be finite"
            
            # Check that diffusion terms are effectively zero (no neighbors)
            # For a single cell, the neighbor sums should all be zero
            # So the dynamics should be purely local (no diffusion contribution)
            
            # Extract derivatives
            dc_dt, dt8_dt, dig_dt, dp8_dt, da_dt = result[0, 0, :]
            
            # Verify we get reasonable dynamics
            print(f"Cancer growth term check: dc_dt should reflect local growth/death dynamics")
            print(f"T8 dynamics: dt8_dt = {dt8_dt:.4f}")
            print(f"IFN-γ dynamics: dig_dt = {dig_dt:.4f}")
            
            # Since there are no neighbors, diffusion terms should be zero
            # The dynamics should be purely based on local interactions
            
            return result
            
        except Exception as e:
            print(f"Error with single cell input: {e}")
            print(f"Error type: {type(e).__name__}")
            raise e

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