from linearize import linearize_FiniteDifference, linearize_autograd
from eval_f import eval_f, Params
from eval_u_keytruda_input import eval_u_keytruda_input
from SimpleSolver import SimpleSolver
import matplotlib.pyplot as plt
import os

class TestLinearize:
    def __init__(self):
        self.p_default = Params(
            lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
            lambda_T=3.0, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
            d_A=0.0315, rows=1, cols=1
        )
        self.figure_dir = "test_linearize_output_figures/"
        os.makedirs(self.figure_dir, exist_ok=True)
    
    def test_scalar_functions(self):
        assert False, "Not implemented yet"
        print(f"TEST PASSED ✅: Linearization of scalar functions.")

    def test_linear_functions(self):
        assert False, "Not implemented yet"
        print(f"TEST PASSED ✅: Linearization of linear functions.")

    def test_eval_f_approx_equal(self):
        assert False, "Not implemented yet"
        print(f"TEST PASSED ✅: eval_f approximately equal to its linearization near x0.")

    def test_forward_euler_approx_equal(self):
        assert False, "Not implemented yet"
        print(f"TEST PASSED ✅: Forward Euler with linearized system approximates nonlinear system.")

    def run_all_tests(self):
        self.test_scalar_functions()
        self.test_linear_functions()
        self.test_eval_f_approx_equal()
        self.test_forward_euler_approx_equal()

if __name__ == "__main__":
    tester = TestLinearize()
    tester.run_all_tests()
