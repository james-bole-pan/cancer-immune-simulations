import numpy as np
import matplotlib.pyplot as plt
import evalf_autograd as f
import evalf_autograd_1dwrapper as wrapper
from SimpleSolver import SimpleSolver
import copy
from functools import partial
from evalf_autograd_1dwrapper import evalf_autograd_1dwrapper

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
            da=0.1,      # drug decay rate
            D_a=0.0,     # No diffusion
            rows=1, 
            cols=1
        )
        
    def constant_input(self, u_val):
        """Create constant input function"""
        def input_func(t):
            return u_val
        return input_func

    def eval_f_wrapper(self, x, p, u):
        return wrapper.evalf_autograd_1dwrapper(x.flatten(), p, u)

    def test_pure_decay(self, w=1.0, num_iter=50):
        """Test pure exponential decay - should follow x(t) = x0 * exp(-decay_rate * t)"""
        
        print("="*60)
        print("TEST 1: Pure Exponential Decay")
        print("="*60)
        print("Expected behavior: All variables should decay exponentially")
        print("Analytical solution: x(t) = x0 * exp(-decay_rate * t)")
        
        # Initial condition: single cell with some values
        x0_1d = np.array([100, 90, 80, 70, 60])  # [c, t8, ig, p8, a]
        
        # No input
        u_func = self.constant_input(0.0)
                
        X, t = SimpleSolver(
            self.eval_f_wrapper,
            x0_1d, 
            self.p_pure_decay, 
            u_func, 
            num_iter, 
            w=w, 
            visualize=True,
            gif_file_name=f"test_evalf_output_figures/pure_decay_w_{w}.gif"
        )

        # Analytical verification
        print("\n" + "-"*50)
        print("ANALYTICAL VERIFICATION:")
        print("-"*50)
        
        # Calculate analytical solution at final time
        final_time = t[-1]
        decay_rates = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # dc, dt8, dig, dt8 (for p8), da
        analytical_final = x0_1d * np.exp(-decay_rates * final_time)
        
        # Get numerical solution at final time
        numerical_final = X[:, -1]
        
        print(f"Final simulation time: {final_time:.3f}")
        print(f"Initial state: {x0_1d}")
        print(f"Analytical final: {analytical_final}")
        print(f"Numerical final:  {numerical_final}")
        
        # Calculate errors
        absolute_error = np.abs(numerical_final - analytical_final)
        relative_error = absolute_error / np.abs(analytical_final)
        max_relative_error = np.max(relative_error)
        
        print(f"\nAbsolute errors: {absolute_error}")
        print(f"Relative errors: {relative_error}")
        print(f"Max relative error: {max_relative_error:.6f} ({max_relative_error*100:.4f}%)")
        
        # Assert the solutions are close (within 1% relative error)
        tolerance = 0.01  # 1% tolerance
        try:
            assert max_relative_error < tolerance, f"Maximum relative error {max_relative_error:.6f} exceeds tolerance {tolerance}"
            print(f"\n✅ ANALYTICAL TEST PASSED: All variables within {tolerance*100}% of analytical solution")
        except AssertionError as e:
            print(f"\n❌ ANALYTICAL TEST FAILED: {e}")
            print(f"Consider reducing omega (w) for higher accuracy or increasing tolerance")
            raise
        
        return X, t, analytical_final, numerical_final

    def test_logistic_growth(self, w=1.0, num_iter=50):
        """Test exponential growth induced by lc with nc=0 (effective dc/dt = (lc/2)*c)."""

        print("="*60)
        print("TEST 2: Exponential Growth (nc=0 ⇒ dc/dt = (lc/2)*c)")
        print("="*60)
        print("Expected behavior: Cancer grows exponentially with rate lc/2 (no saturation)")
        print("Model equation (with nc=0, no killing/decay/diffusion): dc/dt = (lc/2) * c")

        # Setup parameters to realize dc/dt = (lc/2)*c
        p_logistic = copy.deepcopy(self.p_pure_decay)
        p_logistic.lc = 0.1      # intrinsic growth parameter
        p_logistic.tc = 200.0    # irrelevant when nc=0 (term becomes constant)
        p_logistic.nc = 0.0      # makes (c/tc)**0 = 1 ⇒ denominator 1+1 = 2
        p_logistic.dc = 0.0      # no decay
        p_logistic.D_c = 0.0     # no diffusion
        
        # Initial condition — only cancer nonzero
        x0_1d = np.array([50.0, 0.0, 0.0, 0.0, 0.0])  # [c, t8, ig, p8, a]

        # No input
        u_func = self.constant_input(0.0)

        X, t = SimpleSolver(
            self.eval_f_wrapper,
            x0_1d,
            p_logistic,
            u_func,
            num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"test_evalf_output_figures/logistic_growth_w_{w}.gif"
        )

        # ---- Analytical verification ----
        print("\n" + "-"*50)
        print("ANALYTICAL VERIFICATION:")
        print("-"*50)

        final_time = t[-1]
        c0 = x0_1d[0]
        lc = p_logistic.lc
        c_analytical = c0 * np.exp((lc/2) * final_time)

        numerical_final = X[:, -1]
        c_numerical = numerical_final[0]

        print(f"Final simulation time: {final_time:.6f}")
        print(f"Initial cancer (c0):   {c0:.6f}")
        print(f"Effective rate (r):    {lc/2:.6f}  [since nc=0 ⇒ r=lc/2]")
        print(f"Analytical c(T):       {c_analytical:.6f}")
        print(f"Numerical  c(T):       {c_numerical:.6f}")

        # Relative error on cancer channel
        rel_err_c = abs(c_numerical - c_analytical) / max(1e-12, abs(c_analytical))
        print(f"Relative error (c):    {rel_err_c:.6e}")

        # Other channels should remain ~0
        other_abs = np.abs(numerical_final[1:])
        max_other = np.max(other_abs)
        print(f"Max |other channels|:  {max_other:.6e} (should be ~0)")

        # Assertions
        tol_rel = 1e-2  # 1%
        tol_other = 1e-8
        try:
            assert rel_err_c < tol_rel, f"c(T) relative error {rel_err_c:.3e} exceeds {tol_rel:.3e}"
            assert max_other < tol_other, f"Non-cancer channels deviated from 0 (max={max_other:.3e})"
            print(f"\n✅ TEST PASSED: c(t) matches analytical exp growth within {tol_rel*100:.1f}%, others ~0")
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            print("Consider decreasing w or increasing num_iter for tighter accuracy.")
            raise

        return X, t, c_analytical, c_numerical
    
    def test_logistic_growth_t_cell(self, w=1.0, num_iter=50):
        """Test exponential growth induced by lt8"""

        print("="*60)
        print("TEST 3: T Cell Exponential Growth")
        print("="*60)
        print("Expected behavior: T8 grows exponentially with rate lt8")
        print("Model equation: dt8/dt = lt8 * t8")

        p_t8_growth = f.Params(
            lc=0.0,
            tc=1e10,
            nc=0.0,
            k8=0.0,
            ng=0.0,
            ki=1e10,
            dc=0.0,
            D_c=0.0,
            lt8=0.2,
            rl=0.0,
            kq=1e10,
            dt8=0.0,
            D_t8=0.0,
            ligt8=0.0,
            dig=0.0,
            D_ig=0.0,
            mu_a=0.0,
            da=0.0,
            D_a=0.0,
            rows=1,
            cols=1
        )

        # Initial condition — only T8 nonzero
        x0_1d = np.array([40.0, 40.0, 40.0, 40.0, 40.0])  # [c, t8, ig, p8, a]

        # No input
        u_func = self.constant_input(0.0)

        X, t = SimpleSolver(
            self.eval_f_wrapper,
            x0_1d,
            p_t8_growth,
            u_func,
            num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"test_evalf_output_figures/t8_logistic_growth_w_{w}.gif"
        )

        # ---- Analytical verification ----
        print("\n" + "-"*50)
        print("ANALYTICAL VERIFICATION:")
        print("-"*50)

        final_time = t[-1]
        t8_0 = x0_1d[1]
        lt8 = p_t8_growth.lt8
        t8_analytical = t8_0 * np.exp(lt8 * final_time)

        numerical_final = X[:, -1]
        t8_numerical = numerical_final[1]

        print(f"Final simulation time: {final_time:.6f}")
        print(f"Initial T8 (t8_0):     {t8_0:.6f}")
        print(f"Effective rate (r):    {lt8:.6f} ")
        print(f"Analytical t8(T):      {t8_analytical:.6f}")
        print(f"Numerical  t8(T):      {t8_numerical:.6f}")

        # Relative error on T8 channel
        rel_err_t8 = abs(t8_numerical - t8_analytical) / max(1e-12, abs(t8_analytical))
        print(f"Relative error (t8):   {rel_err_t8:.6e}")

        # Assertions
        tol_rel = 1e-2  # 1%
        try:
            assert rel_err_t8 < tol_rel, f"t8(T) relative error {rel_err_t8:.3e} exceeds {tol_rel:.3e}"
            print(f"\n✅ TEST PASSED: t8(t) matches analytical exp growth within {tol_rel*100:.1f}%")
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            print("Consider decreasing w or increasing num_iter for tighter accuracy.")
            raise

        return X, t, t8_analytical, t8_numerical
    
    def test_linear_growth_interferon_gamma(self, w=1.0, num_iter=50):
        """Test linear growth of Interferon gamma (ig) induced by constant ligt8"""

        print("="*60)
        print("TEST 4: Interferon Gamma Linear Growth")
        print("="*60)
        print("Expected behavior: IG grows linearly with rate ligt8 * t8 (t8 constant)")
        print("Model equation: dig/dt = ligt8 * t8 (with t8 constant, no decay)")

        p_ig_growth = f.Params(
            lc=0.0,
            tc=1e10,
            nc=0.0,
            k8=0.0,
            ng=0.0,
            ki=1e10,
            dc=0.0,
            D_c=0.0,
            lt8=0.0,
            rl=0.0,
            kq=1e10,
            dt8=0.0,
            D_t8=0.0,
            ligt8=2.0,   # IG production rate
            dig=0.0,     # No IG decay
            D_ig=0.0,
            mu_a=0.0,
            da=0.0,
            D_a=0.0,
            rows=1,
            cols=1
        )

        # Initial condition — only T8 nonzero, IG zero
        x0_1d = np.array([0.0, 10.0, 5.0, 0.0, 0.0])  # [c, t8, ig, p8, a]

        # No input
        u_func = self.constant_input(0.0)

        X, t = SimpleSolver(
            self.eval_f_wrapper,
            x0_1d,
            p_ig_growth,
            u_func,
            num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"test_evalf_output_figures/ig_linear_growth_w_{w}.gif"
        )

        # ---- Analytical verification ----
        print("\n" + "-"*50)
        print("ANALYTICAL VERIFICATION:")
        print("-"*50)

        final_time = t[-1]
        t8_const = x0_1d[1]
        ligt8 = p_ig_growth.ligt8
        ig0 = x0_1d[2]
        ig_analytical = ig0 + ligt8 * t8_const * final_time  # Linear growth from initial value

        numerical_final = X[:, -1]
        ig_numerical = numerical_final[2]

        print(f"Final simulation time: {final_time:.6f}")
        print(f"Constant T8 (t8):      {t8_const:.6f}")
        print(f"Production rate:        {ligt8:.6f}")
        print(f"Analytical ig(T):       {ig_analytical:.6f}")
        print(f"Numerical  ig(T):       {ig_numerical:.6f}")

        # Relative error on IG channel
        rel_err_ig = abs(ig_numerical - ig_analytical) / max(1e-12, abs(ig_analytical))
        print(f"Relative error (ig):    {rel_err_ig:.6e}")

        # Assertions
        tol_rel = 1e-2  # 1%
        try:
            assert rel_err_ig < tol_rel, f"ig(T) relative error {rel_err_ig:.3e} exceeds {tol_rel:.3e}"
            print(f"\n✅ TEST PASSED: ig(t) matches analytical linear growth within {tol_rel*100:.1f}%")
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            print("Consider decreasing w or increasing num_iter for tighter accuracy.")
            raise

        return X, t, ig_analytical, ig_numerical

    def test_cd8_killing_k8(self, w=0.01, num_iter=50):
        """Test CD8 T cell killing of cancer - k8 parameter
        
        With constant T8=T0 and everything else off, C(t)=C0*exp(-(k8*T0)*t).
        """
        
        print("="*60)
        print("TEST 5: CD8 T Cell Killing (k8 parameter)")
        print("="*60)
        print("Expected behavior: Cancer decays exponentially due to CD8 killing")
        print("Model equation: dc/dt = -(k8*T8)*c")
        print("Analytical solution: c(t) = c0 * exp(-(k8*T0)*t)")
        
        # Setup CD8 killing parameters
        p_cd8_kill = f.Params(
            lc=0.0,
            tc=1e10,
            nc=0.0,
            k8=0.01,
            ng=0.0,
            ki=1e10,
            dc=0.0,
            D_c=0.0,
            lt8=0.0,
            rl=0.0,
            kq=1e10,
            dt8=0.0,
            D_t8=0.0,
            ligt8=0.0,
            dig=0.0, 
            D_ig=0.0,
            mu_a=0.0,
            da=0.0,
            D_a=0.0,
            rows=1,
            cols=1
        )
        
        # Initial condition: cancer and T8 cells present, everything else zero
        c0 = 1000.0
        t8_0 = 20.0
        x0_1d = np.array([c0, t8_0, 0.0, 0.0, 0.0])  # [c, t8, ig, p8, a]
        
        # No drug input
        u_func = self.constant_input(0.0)
        
        X, t = SimpleSolver(
            self.eval_f_wrapper,
            x0_1d,
            p_cd8_kill,
            u_func,
            num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"test_evalf_output_figures/cd8_killing_k8_w_{w}.gif"
        )

        # ---- Analytical verification ----
        print("\n" + "-"*50)
        print("ANALYTICAL VERIFICATION:")
        print("-"*50)
        
        final_time = t[-1]
        k8 = p_cd8_kill.k8
        
        # Analytical solution: c(t) = c0 * exp(-(k8*T0)*t)
        effective_decay_rate = k8 * t8_0
        c_analytical = c0 * np.exp(-effective_decay_rate * final_time)
        t8_analytical = t8_0  # T8 should remain constant
        
        # Get numerical results
        c_numerical = X[0, -1]     # Cancer at final time
        t8_numerical = X[1, -1]    # T8 at final time
        
        print(f"Final simulation time: {final_time:.3f}")
        print(f"Initial cancer (c0): {c0}")
        print(f"Initial T8 (T0): {t8_0}")
        print(f"k8 parameter: {k8}")
        print(f"Effective decay rate (k8*T0): {effective_decay_rate}")
        print(f"Analytical c(T): {c_analytical:.3f}")
        print(f"Numerical c(T): {c_numerical:.3f}")
        print(f"Analytical T8(T): {t8_analytical:.3f} (should be constant)")
        print(f"Numerical T8(T): {t8_numerical:.3f}")
        
        # Calculate relative errors, handle possible division by zero or very small analytical values
        def safe_rel_err(num, ana):
            ana_safe = np.where(np.abs(ana) < 1e-12, 1e-12, ana)
            return np.abs(num - ana) / np.abs(ana_safe)

        rel_err_c = safe_rel_err(c_numerical, c_analytical)
        rel_err_t8 = safe_rel_err(t8_numerical, t8_analytical)
        
        print(f"Cancer relative error: {rel_err_c:.6f} ({rel_err_c*100:.4f}%)")
        print(f"T8 relative error: {rel_err_t8:.6f} ({rel_err_t8*100:.4f}%)")
        
        # Check that other variables remain at zero
        other_vars = X[2:, -1]  # ig, p8, a at final time
        max_other = np.max(np.abs(other_vars))
        print(f"Max deviation of other variables from 0: {max_other:.2e}")
        
        # Assertions
        tol_rel = 2e-2  # 2% tolerance for the main variables
        tol_other = 1e-8  # Very small tolerance for variables that should be zero
        
        try:
            assert rel_err_c < tol_rel, f"c(T) relative error {rel_err_c:.3e} exceeds {tol_rel:.3e}"
            assert rel_err_t8 < tol_rel, f"T8(T) relative error {rel_err_t8:.3e} exceeds {tol_rel:.3e}"
            assert max_other < tol_other, f"Non-relevant channels deviated from 0 (max={max_other:.3e})"
            print(f"\n✅ CD8 KILLING TEST PASSED: c(t) and T8(t) match analytical solutions within {tol_rel*100:.1f}%")
        except AssertionError as e:
            print(f"\n❌ CD8 KILLING TEST FAILED: {e}")
            print("Consider decreasing w or increasing num_iter for tighter accuracy.")
            raise

        return X, t, c_analytical, c_numerical

    def test_ifng_killing_ng_ki(self, w=0.01, num_iter=50):
        """Test IFN-γ killing of cancer cells - ng and ki parameters
        
        C(t)=C0*exp(-ng*I/(I+KI)*t) for constant I; test I=0, KI, 10KI.
        """
        
        print("="*60)
        print("TEST 6: IFN-γ Killing of Cancer (ng, ki parameters)")
        print("="*60)
        print("Expected behavior: Cancer decays based on IFN-γ concentration")
        print("Model equation: dc/dt = -ng*ig/(ig+ki)*c")
        print("Analytical solution: c(t) = c0 * exp(-ng*I/(I+KI)*t)")
        
        # Test cases with different IFN-γ levels
        test_cases = [
            {"name": "I=0 (no IFN-γ)", "ig_level": 0.0, "expected_effect": 0.0},
            {"name": "I=KI (half-saturation)", "ig_level": None, "expected_effect": 0.5},  # Will set ig_level = ki
            {"name": "I=10*KI (high IFN-γ)", "ig_level": None, "expected_effect": 10.0/11.0}  # Will set ig_level = 10*ki
        ]
        
        # Setup IFN-γ killing parameters
        p_ifng_kill = f.Params(
            lc=0.0,
            tc=1e10,
            nc=0.0,
            k8=0.0,
            ng=0.2,
            ki=5.0,
            dc=0.0,
            D_c=0.0,
            lt8=0.0,
            rl=0.0,
            kq=1e10,
            dt8=0.0,
            D_t8=0.0,
            ligt8=0.0,
            dig=0.0, 
            D_ig=0.0,
            mu_a=0.0,
            da=0.0,
            D_a=0.0,
            rows=1,
            cols=1
        )
        
        # Set actual values for test cases
        test_cases[1]["ig_level"] = p_ifng_kill.ki      # I = KI
        test_cases[2]["ig_level"] = 10 * p_ifng_kill.ki # I = 10*KI
        
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\n--- Sub-test {i+1}: {case['name']} ---")
            
            # Initial condition: cancer and specified IFN-γ level
            c0 = 80.0
            ig_level = case["ig_level"]
            x0_1d = np.array([c0, 0.0, ig_level, 0.0, 0.0])  # [c, t8, ig, p8, a]
            
            # No drug input
            u_func = self.constant_input(0.0)
            
            X, t = SimpleSolver(
                self.eval_f_wrapper,
                x0_1d,
                p_ifng_kill,
                u_func,
                num_iter,
                w=w,
                visualize=True,
                gif_file_name=f"test_evalf_output_figures/ifng_killing_case_{i+1}_w_{w}.gif"
            )

            # ---- Analytical verification ----
            final_time = t[-1]
            ng = p_ifng_kill.ng
            ki = p_ifng_kill.ki
            
            # Analytical solution: c(t) = c0 * exp(-ng*I/(I+KI)*t)
            if ig_level == 0.0:
                # No killing when IFN-γ = 0
                effective_decay_rate = 0.0
            else:
                effective_decay_rate = ng * ig_level / (ig_level + ki)
            
            c_analytical = c0 * np.exp(-effective_decay_rate * final_time)
            ig_analytical = ig_level  # IFN-γ should remain constant
            
            # Get numerical results
            c_numerical = X[0, -1]     # Cancer at final time
            ig_numerical = X[2, -1]    # IFN-γ at final time
            
            print(f"IFN-γ level (I): {ig_level}")
            print(f"Half-saturation (KI): {ki}")
            print(f"Killing rate (ng): {ng}")
            print(f"Expected killing fraction: ng*I/(I+KI) = {effective_decay_rate:.4f}")
            print(f"Analytical c(T): {c_analytical:.3f}")
            print(f"Numerical c(T): {c_numerical:.3f}")
            print(f"Analytical IFN-γ(T): {ig_analytical:.3f} (should be constant)")
            print(f"Numerical IFN-γ(T): {ig_numerical:.3f}")
            
            # Calculate relative errors
            if c_analytical > 1e-10:  # Avoid division by very small numbers
                rel_err_c = abs(c_numerical - c_analytical) / c_analytical
            else:
                rel_err_c = abs(c_numerical - c_analytical)  # Absolute error when analytical ~ 0
            
            if ig_analytical > 0:
                rel_err_ig = abs(ig_numerical - ig_analytical) / ig_analytical
            else:
                rel_err_ig = abs(ig_numerical - ig_analytical)  # IFN-γ = 0 case
            
            print(f"Cancer relative error: {rel_err_c:.6f} ({rel_err_c*100:.4f}%)")
            print(f"IFN-γ relative error: {rel_err_ig:.6f} ({rel_err_ig*100:.4f}%)")
            
            # Check that other variables remain at zero
            other_vars = X[[1, 3, 4], -1]  # t8, p8, a at final time
            max_other = np.max(np.abs(other_vars))
            print(f"Max deviation of other variables from 0: {max_other:.2e}")
            
            results.append({
                'case': case['name'],
                'c_analytical': c_analytical,
                'c_numerical': c_numerical,
                'rel_err_c': rel_err_c,
                'rel_err_ig': rel_err_ig,
                'max_other': max_other
            })
        
        # ---- Overall test assertions ----
        print("\n" + "-"*50)
        print("OVERALL TEST VERIFICATION:")
        print("-"*50)
        
        tol_rel = 1e-2  # 1% tolerance for the main variables
        tol_other = 1e-8  # Very small tolerance for variables that should be zero
        
        all_passed = True
        for i, result in enumerate(results):
            case_name = result['case']
            print(f"{case_name}: Cancer error {result['rel_err_c']:.2e}, IFN-γ error {result['rel_err_ig']:.2e}")
            
            if result['rel_err_c'] > tol_rel:
                print(f"  ❌ Cancer error exceeds tolerance")
                all_passed = False
            if result['rel_err_ig'] > tol_rel:
                print(f"  ❌ IFN-γ error exceeds tolerance")
                all_passed = False
            if result['max_other'] > tol_other:
                print(f"  ❌ Other variables not zero")
                all_passed = False
        
        try:
            assert all_passed, "One or more sub-tests failed tolerance checks"
            print(f"\n✅ IFN-γ KILLING TEST PASSED: All cases within {tol_rel*100:.1f}% tolerance")
        except AssertionError as e:
            print(f"\n❌ IFN-γ KILLING TEST FAILED: {e}")
            print("Consider decreasing w or increasing num_iter for tighter accuracy.")
            raise

        return results

    def test_drug_input_ra(self, w=0.01, num_iter=500):
        """Test drug input r_a - verify A grows linearly with rate r_a
        
        When da=0, mu_a=0, D_a=0, the drug equation becomes: dA/dt = r_a
        So A(t) = A0 + r_a * t (linear growth with slope r_a).
        """
        
        print("="*60)
        print("TEST 7: Drug Input Linear Growth (r_a parameter)")
        print("="*60)
        print("Expected behavior: Drug concentration grows linearly with rate r_a")
        print("Model equation: dA/dt = r_a (when da=0, mu_a=0, D_a=0)")
        print("Analytical solution: A(t) = A0 + r_a * t")
        
        # Test cases with different drug input rates
        test_cases = [
            {"name": "r_a = 0 (no input)", "r_a": 0.0},
            {"name": "r_a = 2 (moderate input)", "r_a": 2.0},
            {"name": "r_a = 10 (high input)", "r_a": 10.0}
        ]
        
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\n--- Sub-test {i+1}: {case['name']} ---")
            
            # Setup drug input parameters
            p_drug_input = f.Params(
                lc=0.0,
                tc=1e10,
                nc=0.0,
                k8=0.0,
                ng=0.0,
                ki=1e10,
                dc=0.0,
                D_c=0.0,
                lt8=0.0,
                rl=0.0,
                kq=1e10,
                dt8=0.0,
                D_t8=0.0,
                ligt8=0.0,
                dig=0.0, 
                D_ig=0.0,
                mu_a=0.0,
                da=0.0,
                D_a=0.0,
                rows=1,
                cols=1
            )
            
            # Initial condition: some initial drug concentration
            a0 = 5.0  # Initial drug concentration
            x0_1d = np.array([0.0, 0.0, 0.0, 0.0, a0])  # [c, t8, ig, p8, a]
            
            # Constant drug input rate
            r_a = case["r_a"]
            u_func = self.constant_input(r_a)
            
            X, t = SimpleSolver(
                self.eval_f_wrapper,
                x0_1d,
                p_drug_input,
                u_func,
                num_iter,
                w=w,
                visualize=True,
                gif_file_name=f"test_evalf_output_figures/drug_input_ra_case_{i+1}_w_{w}.gif"
            )

            # ---- Analytical verification ----
            final_time = t[-1]
            
            # Analytical solution: A(t) = A0 + r_a * t
            a_analytical = a0 + r_a * final_time
            
            # Get numerical results
            a_numerical = X[4, -1]    # Drug concentration at final time
            
            print(f"Drug input rate (r_a): {r_a}")
            print(f"Initial drug (A0): {a0}")
            print(f"Final time: {final_time:.3f}")
            print(f"Expected growth: r_a * t = {r_a} * {final_time:.3f} = {r_a * final_time:.3f}")
            print(f"Analytical A(T): {a_analytical:.3f}")
            print(f"Numerical A(T): {a_numerical:.3f}")
            
            # Calculate relative error
            if a_analytical > 1e-10:  # Avoid division by very small numbers
                rel_err_a = abs(a_numerical - a_analytical) / a_analytical
            else:
                rel_err_a = abs(a_numerical - a_analytical)  # Absolute error when analytical ~ 0
            
            print(f"Drug relative error: {rel_err_a:.6f} ({rel_err_a*100:.4f}%)")
            
            # Check that other variables remain at zero
            other_vars = X[0:4, -1]  # c, t8, ig, p8 at final time
            max_other = np.max(np.abs(other_vars))
            print(f"Max deviation of other variables from 0: {max_other:.2e}")
            
            # Additional check: verify linear growth throughout the simulation
            # Check that the slope is approximately constant
            if len(t) > 10:
                # Calculate slope over different intervals
                mid_point = len(t) // 2
                early_slope = (X[4, mid_point] - X[4, 0]) / (t[mid_point] - t[0])
                late_slope = (X[4, -1] - X[4, mid_point]) / (t[-1] - t[mid_point])
                slope_consistency = abs(early_slope - late_slope) / max(abs(early_slope), abs(late_slope), 1e-10)
                
                print(f"Early slope: {early_slope:.6f}, Late slope: {late_slope:.6f}")
                print(f"Slope consistency: {slope_consistency:.6f} (should be small for linear growth)")
            else:
                slope_consistency = 0.0
            
            results.append({
                'case': case['name'],
                'r_a': r_a,
                'a_analytical': a_analytical,
                'a_numerical': a_numerical,
                'rel_err_a': rel_err_a,
                'max_other': max_other,
                'slope_consistency': slope_consistency
            })
        
        # ---- Overall test assertions ----
        print("\n" + "-"*50)
        print("OVERALL TEST VERIFICATION:")
        print("-"*50)
        
        tol_rel = 1e-2  # 1% tolerance for drug concentration
        tol_other = 1e-8  # Very small tolerance for variables that should be zero
        tol_slope = 0.1   # 10% tolerance for slope consistency
        
        all_passed = True
        for i, result in enumerate(results):
            case_name = result['case']
            print(f"{case_name}: Drug error {result['rel_err_a']:.2e}, Slope consistency {result['slope_consistency']:.2e}")
            
            if result['rel_err_a'] > tol_rel:
                print(f"  ❌ Drug concentration error exceeds tolerance")
                all_passed = False
            if result['max_other'] > tol_other:
                print(f"  ❌ Other variables not zero")
                all_passed = False
            if result['slope_consistency'] > tol_slope:
                print(f"  ❌ Growth not sufficiently linear")
                all_passed = False
        
        try:
            assert all_passed, "One or more sub-tests failed tolerance checks"
            print(f"\n✅ DRUG INPUT TEST PASSED: All cases show linear growth with r_a within {tol_rel*100:.1f}% tolerance")
        except AssertionError as e:
            print(f"\n❌ DRUG INPUT TEST FAILED: {e}")
            print("Consider decreasing w or increasing num_iter for tighter accuracy.")
            raise

        return results

    def test_cancer_diffusion_Dc(self, w=0.01, num_iter=500):
        """
        TEST 8: Cancer Cell Diffusion Visualization (D_c parameter)
        
        Expected behavior: Cancer cells should spread spatially via diffusion
        Model equation: dc/dt includes D_c * (sum of neighbor cancer concentrations)
        
        This test visualizes spatial diffusion dynamics on a 3x3 grid.
        No analytical verification yet - visualization only.
        """
        print("="*60)
        print("TEST 8: Cancer Cell Diffusion Visualization (D_c parameter)")
        print("="*60)
        print("Expected behavior: Cancer cells should spread spatially via diffusion")
        print("Model equation: dc/dt includes D_c * (sum of neighbor cancer concentrations)")
        print("Visualization only - no analytical verification yet")
        
        # Test different diffusion constants
        test_cases = [
            {"D_c": 0.0, "description": "no diffusion"},
            {"D_c": 0.005, "description": "slow diffusion"},
            {"D_c": 0.01, "description": "fast diffusion"}
        ]
        
        for case in test_cases:
            print(f"\n--- Sub-test: D_c = {case['D_c']} ({case['description']}) ---")
            
            # Create diffusion parameters
            grid_size = 3
            p_diffusion = f.Params(
                lc=0.0,
                tc=1e10,
                nc=0.0,
                k8=0.0,
                ng=0.0,
                ki=1e10,
                dc=0.0,
                D_c=case['D_c'],
                lt8=0.0,
                rl=0.0,
                kq=1e10,
                dt8=0.0,
                D_t8=0.0,
                ligt8=0.0,
                dig=0.0, 
                D_ig=0.0,
                mu_a=0.0,
                da=0.0,
                D_a=0.0,
                rows=grid_size,
                cols=grid_size
            )
            
            # Create initial condition: concentrated cancer in center, sparse elsewhere
            total_cells = grid_size * grid_size
            x0_1d = np.zeros(total_cells * 5)  # 5 variables per cell
            
            # Reshape to work with grid structure
            x0_grid = x0_1d.reshape(total_cells, 5)
            
            # Set up initial cancer distribution - high concentration in center
            center_idx = (grid_size // 2) * grid_size + (grid_size // 2)  # Center cell
            x0_grid[center_idx, 0] = 100.0  # High cancer concentration in center
            
            x0_1d_reshaped = x0_grid.flatten()
            
            # No drug input
            u_func = self.constant_input(0.0)
            
            print(f"Grid size: {grid_size}x{grid_size}")
            print(f"Total cells: {total_cells}")
            print(f"Diffusion constant D_c: {case['D_c']}")
            print(f"Initial cancer in center: {x0_grid[center_idx, 0]}")
            print(f"Initial total cancer: {np.sum(x0_grid[:, 0])}")
            
            X, t = SimpleSolver(
                self.eval_f_wrapper,
                x0_1d_reshaped,
                p_diffusion,
                u_func,
                num_iter,
                w=w,
                visualize=True,
                gif_file_name=f"test_evalf_output_figures/cancer_diffusion_Dc_{case['D_c']}_w_{w}.gif"
            )

            # ---- Analysis and Visualization ----
            final_time = t[-1]
            
            # Reshape results back to grid format for analysis
            X_reshaped = X.reshape(total_cells, 5, -1)  # [cell, variable, time]
            cancer_data = X_reshaped[:, 0, :]  # Cancer concentration over time
            
            # Analyze spatial distribution at different time points
            initial_cancer = cancer_data[:, 0]
            final_cancer = cancer_data[:, -1]
            
            print(f"\nSpatial Analysis:")
            print(f"Initial total cancer: {np.sum(initial_cancer):.3f}")
            print(f"Final total cancer: {np.sum(final_cancer):.3f}")
            
            # Check conservation (should be preserved in pure diffusion)
            conservation_error = abs(np.sum(final_cancer) - np.sum(initial_cancer)) / np.sum(initial_cancer)
            print(f"Cancer conservation: {1 - conservation_error:.3f}")
            
            # Analyze spatial spreading
            def spatial_spread(cancer_grid):
                """Calculate spatial spread using variance of position"""
                if np.sum(cancer_grid) == 0:
                    return 0.0
                
                total_cancer = np.sum(cancer_grid)
                center_of_mass_x = 0
                center_of_mass_y = 0
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        idx = i * grid_size + j
                        center_of_mass_x += i * cancer_grid[idx] / total_cancer
                        center_of_mass_y += j * cancer_grid[idx] / total_cancer
                
                variance = 0
                for i in range(grid_size):
                    for j in range(grid_size):
                        idx = i * grid_size + j
                        distance_sq = (i - center_of_mass_x)**2 + (j - center_of_mass_y)**2
                        variance += distance_sq * cancer_grid[idx] / total_cancer
                
                return np.sqrt(variance)
            
            initial_spread = spatial_spread(initial_cancer)
            final_spread = spatial_spread(final_cancer)
            spread_increase = final_spread - initial_spread
            
            print(f"Initial spatial spread: {initial_spread:.3f}")
            print(f"Final spatial spread: {final_spread:.3f}")
            print(f"Spread increase: {spread_increase:.3f}")
            
            # Display spatial distributions in grid format
            print(f"\nInitial Cancer Distribution (center region):")
            initial_grid = initial_cancer.reshape(grid_size, grid_size)
            for row in initial_grid:
                print("  " + "  ".join(f"{val:5.1f}" for val in row))
            
            print(f"\nFinal Cancer Distribution (center region):")
            final_grid = final_cancer.reshape(grid_size, grid_size)
            for row in final_grid:
                print("  " + "  ".join(f"{val:5.1f}" for val in row))
        
        # Summary of diffusion test results
        print(f"\n" + "-"*50)
        print("DIFFUSION TEST SUMMARY:")
        print("-"*50)
        for case in test_cases:
            print(f"D_c = {case['D_c']} ({case['description']}):")
            print(f"  Spatial dynamics visualized successfully")
        
        print(f"\n✅ CANCER DIFFUSION VISUALIZATION COMPLETE")
        print("Check the generated GIF files to see spatial dynamics!")

    def run_all_tests(self, total_simulation_time=15, omega=0.01):
        """Run all test cases"""
        print("Starting comprehensive SimpleSolver tests with evalf_autograd_1dwrapper")
        print("="*80)

        num_iterations = int(total_simulation_time / omega)
        
        print(f"Using omega = {omega} for {num_iterations} iterations over {total_simulation_time} time units")
        print(f"Testing {self.count_tested_parameters()} out of 20 model parameters")
        
        # Run all tests
        try:
            print("\n🧪 Running Test Suite...")
            
            self.test_pure_decay(w=omega, num_iter=num_iterations)
            self.test_logistic_growth(w=omega, num_iter=num_iterations) 
            self.test_logistic_growth_t_cell(w=omega, num_iter=num_iterations)
            self.test_linear_growth_interferon_gamma(w=omega, num_iter=num_iterations)
            self.test_cd8_killing_k8(w=omega, num_iter=num_iterations)
            self.test_ifng_killing_ng_ki(w=omega, num_iter=num_iterations)
            self.test_drug_input_ra(w=omega, num_iter=num_iterations)
            self.test_cancer_diffusion_Dc(w=omega, num_iter=num_iterations)
            
            print("\n" + "="*80)
            print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("✅ Pure Decay Test: dc, dt8, dig, da")
            print("✅ Logistic Growth Test: lc")  
            print("✅ T Cell Growth Test: lt8")
            print("✅ Interferon Gamma Growth Test: ligt8")
            print("✅ CD8 T Cell Killing Test: k8")
            print("✅ IFN-γ Killing Test: ng, ki")
            print("✅ Drug Input Test: r_a (drug input rate)")
            print("✅ Cancer Diffusion Visualization: D_c")
            print(f"📊 Parameters tested: {self.count_tested_parameters()}/20")
            
        except AssertionError as e:
            print("\n" + "="*80)
            print("❌ TEST SUITE FAILED")
            print("="*80)
            print(f"Error: {e}")
            raise
    
    def count_tested_parameters(self):
        """Count how many of the 20 parameters we're testing"""
        # Parameters tested across all test functions:
        tested_params = [
            'dc', 'dt8', 'dig', 'da',  # Pure decay test (4)
            'lc',                 # Logistic growth test (1) 
            'lt8',                # T Cell growth test (1)
            'ligt8',              # Interferon Gamma growth test (1)
            'k8',                 # CD8 killing test (1)
            'ng', 'ki',           # IFN-γ killing test (2)
            'r_a',                # Drug input test (1) - this represents the drug input function u(t)
            'D_c'                 # Cancer diffusion test (1)
        ]
        return len(tested_params)  # Total: 12 parameters
        
if __name__ == "__main__":
    tester = TestSimpleSolver()

    total_simulation_time = 5.0
    omega = 0.1  # Use higher precision for complex dynamics

    tester.run_all_tests(total_simulation_time = total_simulation_time, omega = omega)