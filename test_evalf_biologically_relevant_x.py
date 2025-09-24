import numpy as np
import pytest
import evalf_autograd as f
import matplotlib.pyplot as plt

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
            rows=self.x0.shape[0],  # rows in grid (from loaded data)
            cols=self.x0.shape[1]   # cols in grid (from loaded data)
        )
        
        self.u_standard = 0.015

    def test_high_lambda_c_trajectory_simulation(self):
        """Simulate trajectory with high lambda_c and plot tumor burden vs immune cells"""
        
        # Set very high lambda_c for aggressive cancer growth
        p_high_lc = f.Params(
            lc=5.0,        # Very high lambda_c (vs standard 0.5)
            tc=5e7, nc=2, k8=3e-7, ng=0.1, ki=10, dc=0.18, D_c=0.01,
            lt8=0.03, rl=3e-7, kq=12.6, dt8=0.1, D_t8=0.01,
            ligt8=2.5e-8, dig=18, D_ig=0.01,
            mu_a=0.03, da=0.05, D_a=0.01, 
            rows=self.x0.shape[0], cols=self.x0.shape[1]
        )
        
        # Simulation parameters
        dt = 0.001  # Time step
        n_steps = 100  # Number of time steps
        
        # Storage for trajectories
        time_points = np.arange(0, n_steps * dt, dt)
        y1_trajectory = []  # Total tumor burden (sum of all cancer cells)
        y2_trajectory = []  # Total immune cells (sum of T8 cells)
        
        # Initial state
        x_current = self.x0.copy()
        
        # Simple Euler integration
        for step in range(n_steps):
            # Calculate current totals
            cancer_total = np.sum(x_current[:, :, 0])  # c (cancer cells)
            immune_total = np.sum(x_current[:, :, 1])  # t8 (T8 cells)
            
            y1_trajectory.append(cancer_total)
            y2_trajectory.append(immune_total)
            
            # Compute derivatives
            dx_dt = f.evalf_autograd(x_current, p_high_lc, self.u_standard)
            
            # Euler step: x(t+dt) = x(t) + dt * dx/dt
            x_current = x_current + dt * dx_dt
            
            # Ensure non-negative values (biological constraint)
            x_current = np.maximum(x_current, 0)
        
        y1_trajectory = np.array(y1_trajectory)
        y2_trajectory = np.array(y2_trajectory)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, y1_trajectory, 'r-', linewidth=2, label='Total Tumor Burden (y1)')
        plt.plot(time_points, y2_trajectory, 'b-', linewidth=2, label='Total Immune Cells (y2)')
        plt.xlabel('Time')
        plt.ylabel('Cell Count')
        plt.title('Tumor vs Immune Cell Dynamics with High λc = 5.0')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('test_evalf_output_figures/high_lambda_c_trajectory.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Initial tumor burden: {y1_trajectory[0]:.2e}")
        print(f"Final tumor burden: {y1_trajectory[-1]:.2e}")
        print(f"Initial immune cells: {y2_trajectory[0]:.2e}")
        print(f"Final immune cells: {y2_trajectory[-1]:.2e}")
        
        # Basic assertions
        assert len(y1_trajectory) == n_steps, "Trajectory length should match number of steps"
        assert len(y2_trajectory) == n_steps, "Trajectory length should match number of steps"
        assert np.all(np.isfinite(y1_trajectory)), "Tumor trajectory should be finite"
        assert np.all(np.isfinite(y2_trajectory)), "Immune trajectory should be finite"
        
        return time_points, y1_trajectory, y2_trajectory

if __name__ == "__main__":
    # Run tests if script is executed directly
    test_instance = Test_evalf_autograd()
    test_instance.setup_method()
    
    print("Running evalf_autograd regression tests...")
    print(f"Loaded data shape: {test_instance.x0.shape}")
    
    # Run the trajectory simulation test
    try:
        test_instance.test_high_lambda_c_trajectory_simulation()
        print("✓ test_high_lambda_c_trajectory_simulation")
    except Exception as e:
        print(f"✗ test_high_lambda_c_trajectory_simulation: {e}")
    
    # Run other test methods if they exist
    test_methods = [method for method in dir(test_instance) if method.startswith('test_') and method != 'test_high_lambda_c_trajectory_simulation']
    
    for test_method in test_methods:
        try:
            getattr(test_instance, test_method)()
            print(f"✓ {test_method}")
        except Exception as e:
            print(f"✗ {test_method}: {e}")
    
    print("Tests completed!")