import numpy as np
import pytest
import evalf_autograd as f
import matplotlib.pyplot as plt
import os
import copy

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
            k8=0.00003,       # kappa_8
            ng=0.1,        # n_g
            ki=10,         # K_i
            dc=0.18,       # d_c
            D_c=0.1,      # D_c
            lt8=0.03,      # lambda_t8
            rl=3e-7,       # rho_l
            kq=12.6,       # K_q
            dt8=0.1,       # d_t8
            D_t8=0.1,     # D_t8
            ligt8=2.5e-8,  # lambda_igt8
            dig=18,        # d_ig
            D_ig=0.1,     # D_ig
            mu_a=0.03,     # mu_a
            da=0.05,       # d_a
            D_a=0.1,      # D_a
            rows=self.x0.shape[0],  # rows in grid (from loaded data)
            cols=self.x0.shape[1]   # cols in grid (from loaded data)
        )
        
        self.u_standard = 0.015
        
        # Simulation parameters
        self.dt = 2  # Time step
        self.n_steps = 5  # Number of time steps
        
        # Output directory for figures
        self.output_dir = 'test_evalf_output_figures'
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _simulate_trajectory(self, params, title_suffix):
        """Helper method to simulate trajectory and return results"""
        # Storage for trajectories
        time_points = np.arange(0, self.n_steps * self.dt, self.dt)
        y1_trajectory = []  # Total tumor burden (sum of all cancer cells)
        y2_trajectory = []  # Total immune cells (sum of T8 cells)
        
        # Initial state
        x_current = self.x0.copy()
        
        # Simple Euler integration
        for step in range(self.n_steps):
            # Calculate current totals
            cancer_total = np.sum(x_current[:, :, 0])  # c (cancer cells)
            immune_total = np.sum(x_current[:, :, 1])  # t8 (T8 cells)
            
            y1_trajectory.append(cancer_total)
            y2_trajectory.append(immune_total)
            
            # Compute derivatives
            dx_dt = f.evalf_autograd(x_current, params, self.u_standard)
            
            # Euler step: x(t+dt) = x(t) + dt * dx/dt
            x_current = x_current + self.dt * dx_dt
            
            # Ensure non-negative values (biological constraint)
            x_current = np.maximum(x_current, 0)
        
        return time_points, np.array(y1_trajectory), np.array(y2_trajectory)
    
    def _simulate_trajectory_with_drug_input(self, params, drug_input, title_suffix):
        """Helper method to simulate trajectory with custom drug input and return results"""
        # Storage for trajectories
        time_points = np.arange(0, self.n_steps * self.dt, self.dt)
        y1_trajectory = []  # Total tumor burden (sum of all cancer cells)
        y2_trajectory = []  # Total immune cells (sum of T8 cells)
        
        # Initial state
        x_current = self.x0.copy()
        
        # Simple Euler integration
        for step in range(self.n_steps):
            # Calculate current totals
            cancer_total = np.sum(x_current[:, :, 0])  # c (cancer cells)
            immune_total = np.sum(x_current[:, :, 1])  # t8 (T8 cells)
            
            y1_trajectory.append(cancer_total)
            y2_trajectory.append(immune_total)
            
            # Compute derivatives with custom drug input
            dx_dt = f.evalf_autograd(x_current, params, drug_input)
            
            # Euler step: x(t+dt) = x(t) + dt * dx/dt
            x_current = x_current + self.dt * dx_dt
            
            # Ensure non-negative values (biological constraint)
            x_current = np.maximum(x_current, 0)
        
        return time_points, np.array(y1_trajectory), np.array(y2_trajectory)
    
    def _create_and_save_plot(self, time_points, y1_trajectory, y2_trajectory, title, filename):
        """Helper method to create and save trajectory plots"""
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, y1_trajectory, 'r-', linewidth=2, label='Total Tumor Burden (y1)')
        plt.plot(time_points, y2_trajectory, 'b-', linewidth=2, label='Total Immune Cells (y2)')
        plt.xlabel('Time')
        plt.ylabel('Cell Count')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _validate_trajectory(self, y1_trajectory, y2_trajectory, param_name=""):
        """Helper method to validate trajectory results"""
        suffix = f" for {param_name}" if param_name else ""
        assert len(y1_trajectory) == self.n_steps, f"Trajectory length should match number of steps{suffix}"
        assert len(y2_trajectory) == self.n_steps, f"Trajectory length should match number of steps{suffix}"
        assert np.all(np.isfinite(y1_trajectory)), f"Tumor trajectory should be finite{suffix}"
        assert np.all(np.isfinite(y2_trajectory)), f"Immune trajectory should be finite{suffix}"

    def test_all_parameters_high_values(self):
        """Test trajectory simulation with each parameter set to high value individually"""
        
        # Define parameter variations - each parameter gets tested with a high value
        param_tests = [
            # (param_name, high_value, title_suffix)
            ('lc', 5.0, 'High λc (Cancer Growth Rate)'),
            ('tc', 5e8, 'High θc (Carrying Capacity)'),
            ('nc', 10, 'High nc (Hill Coefficient)'),
            ('k8', 3, 'High κ8 (T8 Kill Rate)'),
            ('ng', 10.0, 'High ng (IFN-γ Effect)'),
            ('ki', 100, 'High Ki (IFN-γ Half-saturation)'),
            ('dc', 2.0, 'High dc (Cancer Death Rate)'),
            ('D_c', 0.1, 'High Dc (Cancer Diffusion)'),
            ('lt8', 0.3, 'High λt8 (T8 Growth Rate)'),
            ('rl', 3e-5, 'High ρl (P8 Inhibition)'),
            ('kq', 126, 'High Kq (P8 Half-saturation)'),
            ('dt8', 1.0, 'High dt8 (T8 Death Rate)'),
            ('D_t8', 0.1, 'High Dt8 (T8 Diffusion)'),
            ('ligt8', 2.5e-6, 'High λigt8 (IFN-γ Production)'),
            ('dig', 180, 'High dig (IFN-γ Degradation)'),
            ('D_ig', 0.1, 'High Dig (IFN-γ Diffusion)'),
            ('mu_a', 0.3, 'High μa (Drug Effectiveness)'),
            ('da', 0.5, 'High da (Drug Degradation)'),
            ('D_a', 0.1, 'High Da (Drug Diffusion)')
        ]
        
        # Store all results for comparison
        all_results = {}
        
        for param_name, high_value, title_suffix in param_tests:
            print(f"Testing parameter: {param_name} = {high_value}")
            
            # Create parameter set with one parameter set to high value
            p_test = copy.deepcopy(self.p_standard)
            setattr(p_test, param_name, high_value)
            
            # Simulate trajectory
            time_points, y1_trajectory, y2_trajectory = self._simulate_trajectory(p_test, title_suffix)
            
            # Store results
            all_results[param_name] = {
                'time': time_points,
                'tumor': y1_trajectory,
                'immune': y2_trajectory,
                'title': title_suffix
            }
            
            # Create and save individual plot
            filename = f'high_{param_name}_trajectory.png'
            self._create_and_save_plot(time_points, y1_trajectory, y2_trajectory,
                                     f'Tumor vs Immune Cell Dynamics - {title_suffix}',
                                     filename)
            
            # Validate results
            self._validate_trajectory(y1_trajectory, y2_trajectory, param_name)
        
        # Create summary plots
        self._create_parameter_summary_plot(all_results)
        
        print(f"Successfully tested all {len(param_tests)} parameters")
        return all_results
    
    def _create_parameter_summary_plot(self, all_results):
        """Create summary plots comparing effects of all parameters"""
        
        # Create subplots for tumor burden comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Tumor burden trajectories
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_results)))
        for i, (param_name, data) in enumerate(all_results.items()):
            ax1.plot(data['time'], data['tumor'], color=colors[i], 
                    linewidth=1.5, label=f"{param_name}", alpha=0.8)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Total Tumor Burden')
        ax1.set_title('Tumor Burden Trajectories - All Parameter Variations')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Immune cell trajectories
        for i, (param_name, data) in enumerate(all_results.items()):
            ax2.plot(data['time'], data['immune'], color=colors[i], 
                    linewidth=1.5, label=f"{param_name}", alpha=0.8)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Total Immune Cells')
        ax2.set_title('Immune Cell Trajectories - All Parameter Variations')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'all_parameters_summary.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create final values comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        param_names = list(all_results.keys())
        tumor_finals = [all_results[p]['tumor'][-1] for p in param_names]
        immune_finals = [all_results[p]['immune'][-1] for p in param_names]
        
        # Bar plot of final tumor burden
        bars1 = ax1.bar(range(len(param_names)), tumor_finals, color=colors[:len(param_names)])
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Final Tumor Burden')
        ax1.set_title('Final Tumor Burden by Parameter Variation')
        ax1.set_xticks(range(len(param_names)))
        ax1.set_xticklabels(param_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Bar plot of final immune cells
        bars2 = ax2.bar(range(len(param_names)), immune_finals, color=colors[:len(param_names)])
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Final Immune Cells')
        ax2.set_title('Final Immune Cells by Parameter Variation')
        ax2.set_xticks(range(len(param_names)))
        ax2.set_xticklabels(param_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_final_values_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

    def test_high_drug_concentration_trajectory_simulation(self):
        """Test trajectory simulation with high drug concentration (r_a)"""
        
        # Use standard parameters but increase the drug input concentration
        high_drug_input = 1000.0  # 10x higher than standard (0.015)
        
        # Simulate trajectory with high drug input
        time_points, y1_trajectory, y2_trajectory = self._simulate_trajectory_with_drug_input(
            self.p_standard, high_drug_input, 'High Drug Concentration (r_a)'
        )
        
        # Create and save plot
        self._create_and_save_plot(
            time_points, y1_trajectory, y2_trajectory,
            f'Tumor vs Immune Cell Dynamics - High Drug Concentration (r_a = {high_drug_input})',
            'high_drug_concentration_trajectory.png'
        )
        
        # Validate results
        self._validate_trajectory(y1_trajectory, y2_trajectory, 'high drug concentration')
        
        # The drug effectiveness should be visible in the trajectory
        assert np.all(y1_trajectory >= 0), "Tumor burden should remain non-negative"
        assert np.all(y2_trajectory >= 0), "Immune cells should remain non-negative"

if __name__ == "__main__":
    # Run tests if script is executed directly
    test_instance = Test_evalf_autograd()
    test_instance.setup_method()
    
    print("Running evalf_autograd regression tests...")
    print(f"Loaded data shape: {test_instance.x0.shape}")
    
    # Run other test methods if they exist
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    for test_method in test_methods:
        try:
            getattr(test_instance, test_method)()
            print(f"✓ {test_method}")
        except Exception as e:
            print(f"✗ {test_method}: {e}")
    
    print("Tests completed!")