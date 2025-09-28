import numpy as np
import matplotlib.pyplot as plt
import evalf_autograd as f
import evalf_autograd_1dwrapper as wrapper
from SimpleSolver import SimpleSolver
import time

class OmegaConvergenceAnalysis:
    """Systematic analysis of omega parameter for SimpleSolver convergence"""
    
    def __init__(self):
        self.setup_reference_case()
    
    def setup_reference_case(self):
        """Setup the pure decay case as our reference for convergence analysis"""
        
        # Pure decay parameters - simple, analytically solvable case
        self.p_decay = f.Params(
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
            da=0.1,      # Drug decay rate
            D_a=0.0,     # No diffusion
            rows=1, 
            cols=1
        )
        
        # Reference initial condition
        self.x0 = np.array([100.0, 90.0, 80.0, 70.0, 60.0])  # [c, t8, ig, p8, a]
        
        # No drug input
        self.u_func = lambda t: 0.0
        
        # Simulation time
        self.t_final = 5.0
        
        print("REFERENCE CASE SETUP:")
        print("="*50)
        print("Pure exponential decay with rates:")
        print(f"  Cancer (c):    -dc = -{self.p_decay.dc}")
        print(f"  T8 cells (t8): -dt8 = -{self.p_decay.dt8}")
        print(f"  IFN-γ (ig):    -dig = -{self.p_decay.dig}")
        print(f"  P8 cells (p8): coupled to T8 dynamics")
        print(f"  Drug (a):   -da = -{self.p_decay.da}")
        print(f"Initial state: {self.x0}")
        print(f"Simulation time: {self.t_final}")
    
    def analytical_solution(self, t):
        """Compute analytical solution for comparison"""
        # Pure exponential decay: x(t) = x0 * exp(-decay_rate * t)
        decay_rates = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # dc, dt8, dig, effective_p8, da
        return self.x0 * np.exp(-decay_rates * t)
    
    def run_simulation(self, omega, verbose=False):
        """Run simulation with given omega value"""
        
        def eval_f_wrapper(x, p, u):
            return wrapper.evalf_autograd_1dwrapper(x.flatten(), p, u)
        
        # Calculate number of iterations needed
        num_iter = int(self.t_final / omega)
        
        if verbose:
            print(f"Running simulation: ω = {omega}, iterations = {num_iter}")
        
        start_time = time.time()
        
        try:
            X, t = SimpleSolver(
                eval_f_wrapper, 
                self.x0, 
                self.p_decay, 
                self.u_func, 
                num_iter, 
                w=omega, 
                visualize=False
            )
            
            elapsed_time = time.time() - start_time
            final_solution = X[:, -1]
            
            return {
                'omega': omega,
                'final_solution': final_solution,
                'trajectory': X,
                'time_points': t,
                'iterations': num_iter,
                'elapsed_time': elapsed_time,
                'success': True
            }
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            return {
                'omega': omega,
                'success': False,
                'error': str(e)
            }
    
    def analyze_convergence(self, omega_values=None):
        """Analyze convergence for different omega values"""
        
        if omega_values is None:
            # Start with relatively large omega and decrease systematically
            omega_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        
        print("\nOMEGA CONVERGENCE ANALYSIS:")
        print("="*80)
        print(f"{'Omega':>8} {'Iterations':>10} {'Time(s)':>8} {'Error':>12} {'Conv.Rate':>12} {'Status':>10}")
        print("-"*80)
        
        results = []
        analytical_final = self.analytical_solution(self.t_final)
        
        for omega in omega_values:
            result = self.run_simulation(omega)
            
            if result['success']:
                # Calculate error relative to analytical solution
                error = np.linalg.norm(result['final_solution'] - analytical_final)
                relative_error = error / np.linalg.norm(analytical_final)
                
                result['absolute_error'] = error
                result['relative_error'] = relative_error
                
                print(f"{omega:8.4f} {result['iterations']:10d} {result['elapsed_time']:8.3f} "
                      f"{relative_error:12.6f} {omega*relative_error:12.6f} {'SUCCESS':>10}")
                
            else:
                print(f"{omega:8.4f} {'N/A':>10} {'N/A':>8} {'N/A':>12} {'N/A':>12} {'FAILED':>10}")
            
            results.append(result)
        
        return results
    
    def analyze_solution_differences(self, results):
        """Analyze differences between consecutive omega values"""
        
        print("\nSOLUTION CONVERGENCE ANALYSIS:")
        print("="*80)
        print(f"{'Omega1':>8} {'Omega2':>8} {'||Δx||':>12} {'Rel.Diff':>12} {'Converged?':>12}")
        print("-"*80)
        
        successful_results = [r for r in results if r['success']]
        convergence_data = []
        
        for i in range(len(successful_results) - 1):
            r1, r2 = successful_results[i], successful_results[i+1]
            
            # Calculate difference between solutions
            diff = np.linalg.norm(r1['final_solution'] - r2['final_solution'])
            rel_diff = diff / np.linalg.norm(r2['final_solution'])
            
            # Convergence criterion: relative difference < 0.01 (1%)
            converged = rel_diff < 0.01
            
            convergence_data.append({
                'omega1': r1['omega'],
                'omega2': r2['omega'], 
                'absolute_diff': diff,
                'relative_diff': rel_diff,
                'converged': converged
            })
            
            print(f"{r1['omega']:8.4f} {r2['omega']:8.4f} {diff:12.6f} "
                  f"{rel_diff:12.6f} {'YES' if converged else 'NO':>12}")
        
        return convergence_data
    
    def recommend_omega(self, results, convergence_data):
        """Provide omega recommendations based on analysis"""
        
        print("\nOMEGA RECOMMENDATIONS:")
        print("="*50)
        
        successful_results = [r for r in results if r['success']]
        
        # Find first omega where consecutive solutions converge (< 1% difference)
        converged_pairs = [c for c in convergence_data if c['converged']]
        
        if converged_pairs:
            recommended_omega = converged_pairs[0]['omega2']  # The smaller omega
            
            # Find corresponding result
            rec_result = next(r for r in successful_results if r['omega'] == recommended_omega)
            
            print(f"RECOMMENDED OMEGA: {recommended_omega}")
            print(f"Reasoning:")
            print(f"  - First omega where solution difference < 1%")
            print(f"  - Achieves {rec_result['relative_error']:.6f} relative accuracy")
            print(f"  - Requires {rec_result['iterations']} iterations ({rec_result['elapsed_time']:.3f}s)")
            print(f"  - Good balance of accuracy vs computational cost")
        
        # Accuracy tiers
        print(f"\nACCURAY TIERS:")
        for threshold, description in [(0.1, "Quick testing"), (0.01, "Production"), (0.001, "High precision")]:
            suitable = [r for r in successful_results if r['relative_error'] < threshold]
            if suitable:
                best = max(suitable, key=lambda x: x['omega'])  # Largest omega with required accuracy
                print(f"  {description:15} (< {threshold*100:4.1f}% error): ω = {best['omega']:6.4f} "
                      f"({best['iterations']:4d} iter, {best['elapsed_time']:5.3f}s)")
        
        return recommended_omega if converged_pairs else None
    
    def create_convergence_plots(self, results):
        """Create visualization plots"""
        
        successful_results = [r for r in results if r['success']]
        omega_vals = [r['omega'] for r in successful_results]
        errors = [r['relative_error'] for r in successful_results]
        times = [r['elapsed_time'] for r in successful_results]
        iterations = [r['iterations'] for r in successful_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Error vs Omega
        ax1.loglog(omega_vals, errors, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Omega (ω)')
        ax1.set_ylabel('Relative Error')
        ax1.set_title('Accuracy vs Step Size')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='1% error')
        ax1.legend()
        
        # Error vs Iterations  
        ax2.loglog(iterations, errors, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Iterations')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Accuracy vs Computational Cost')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='1% error')
        ax2.legend()
        
        # Time vs Omega
        ax3.semilogy(omega_vals, times, 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Omega (ω)')
        ax3.set_ylabel('Computation Time (s)')
        ax3.set_title('Computation Time vs Step Size')
        ax3.grid(True, alpha=0.3)
        
        # Solution trajectories for different omegas
        selected_omegas = [1.0, 0.1, 0.01, 0.001] if len(successful_results) >= 4 else omega_vals[:4]
        for omega in selected_omegas:
            result = next((r for r in successful_results if r['omega'] == omega), None)
            if result:
                ax4.plot(result['time_points'], result['trajectory'][0, :], 
                        label=f'ω = {omega}', linewidth=2)
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Cancer Cells')
        ax4.set_title('Cancer Trajectory for Different ω')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_evalf_output_figures/omega_convergence_analysis.png', dpi=150, bbox_inches='tight')
        #plt.show()
        
        print(f"\nConvergence plots saved as 'omega_convergence_analysis.png'")
    
    def run_full_analysis(self):
        """Run complete omega analysis"""
        
        print("SYSTEMATIC OMEGA CONVERGENCE ANALYSIS")
        print("="*80)
        print("Objective: Find optimal ω balancing accuracy and computational cost")
        print("Method: Pure decay case with analytical solution for validation")
        
        # Run convergence analysis
        results = self.analyze_convergence()
        
        # Analyze solution differences
        convergence_data = self.analyze_solution_differences(results)
        
        # Get recommendations
        recommended_omega = self.recommend_omega(results, convergence_data)
        
        # Create plots
        self.create_convergence_plots(results)
        
        return results, convergence_data, recommended_omega

if __name__ == "__main__":
    analyzer = OmegaConvergenceAnalysis()
    results, convergence_data, recommended_omega = analyzer.run_full_analysis()