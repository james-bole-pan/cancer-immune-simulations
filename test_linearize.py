from linearize import linearize_FiniteDifference, linearize_autograd
from eval_f import eval_f, Params
from eval_u_keytruda_input import eval_u_keytruda_input
from SimpleSolver import SimpleSolver
from test_visualizeNetwork import TestVisualizeNetwork
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import autograd.numpy as anp

class BasicParameters:
    def __init__(self, dxFD=None):
        if dxFD is not None:
            self.dxFD = dxFD

class TestLinearize:
    def __init__(self):
        self.p_default = self.p_default = Params(
            lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
            lambda_T=3.0, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
            d_A=0.0315, rows=3, cols=3
        )
        self.figure_dir = "test_linearize_output_figures/"
        os.makedirs(self.figure_dir, exist_ok=True)
    
    def test_scalar_functions1(self):
        # test on the function f(x, p, u) = x^2 (x + u)
        def eval_f_scalar(x, p, u):
            return x**2 * (x + u)
        
        x0 = np.array([[6.0]])
        u0 = np.array([[7.0]])
        p = BasicParameters(dxFD=1e-8)

        expected_A = 3 * x0**2 + 2 * x0 * u0
        expected_B = np.hstack((-2 * x0**2 * (x0 + u0), x0**2))

        A_fd, B_fd = linearize_FiniteDifference(eval_f_scalar, x0, u0, p)
        A_ag, B_ag = linearize_autograd(eval_f_scalar, x0, u0, p)

        atol = 1e-6
        rtol = 1e-6

        assert np.allclose(A_fd, expected_A, atol=atol, rtol=rtol), f"Finite Difference A mismatch: {A_fd} vs {expected_A}"
        assert np.allclose(B_fd, expected_B, atol=atol, rtol=rtol), f"Finite Difference B mismatch: {B_fd} vs {expected_B}"
        assert np.allclose(A_ag, expected_A, atol=atol, rtol=rtol), f"Autograd A mismatch: {A_ag} vs {expected_A}"
        assert np.allclose(B_ag, expected_B, atol=atol, rtol=rtol), f"Autograd B mismatch: {B_ag} vs {expected_B}"

        print(f"TEST PASSED ✅: Linearization of scalar functions.")

    def test_scalar_functions2(self):
        def eval_f_scalar2(x, p, u):
            return anp.sin(x + 2.0 * u)
        
        x0 = np.array([[0.5]])
        u0 = np.array([[0.25]])
        p = BasicParameters(dxFD=1e-8)

        expected_A = np.cos(x0 + 2.0 * u0)
        expected_B = np.hstack((np.sin(x0 + 2.0 * u0) - np.cos(x0 + 2.0 * u0) * (x0 + 2.0 * u0), 2.0 * np.cos(x0 + 2.0 * u0)))

        A_fd, B_fd = linearize_FiniteDifference(eval_f_scalar2, x0, u0, p)
        A_ag, B_ag = linearize_autograd(eval_f_scalar2, x0, u0, p)

        atol = 1e-6
        rtol = 1e-6

        assert np.allclose(A_fd, expected_A, atol=atol, rtol=rtol), f"Finite Difference A mismatch: {A_fd} vs {expected_A}"
        assert np.allclose(B_fd, expected_B, atol=atol, rtol=rtol), f"Finite Difference B mismatch: {B_fd} vs {expected_B}"
        assert np.allclose(A_ag, expected_A, atol=atol, rtol=rtol), f"Autograd A mismatch: {A_ag} vs {expected_A}"
        assert np.allclose(B_ag, expected_B, atol=atol, rtol=rtol), f"Autograd B mismatch: {B_ag} vs {expected_B}"

        print(f"TEST PASSED ✅: Linearization of scalar functions 2.")

    def test_linear_functions(self):
        A_fun = np.array([[1.0, 2.0, -3.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]])
        B_fun = np.array([[0.5, -1.0], [1.5, 2.0], [-2.5, 3.0]])

        def eval_f_linear(x, p, u):
            return A_fun @ x + B_fun @ u
        
        p = BasicParameters(dxFD=1e-8)

        expected_A = A_fun
        expected_B = np.hstack((np.zeros((A_fun.shape[0], 1)), B_fun))

        atol = 1e-4
        rtol = 1e-4
        np.random.seed(3290)
        
        # loop over multiple random x0, u0
        for _ in range(100):
            x0 = np.random.uniform(-3, 3, (A_fun.shape[1], 1))
            u0 = np.random.uniform(-3, 3, (B_fun.shape[1], 1))

            A_fd, B_fd = linearize_FiniteDifference(eval_f_linear, x0, u0, p)
            A_ag, B_ag = linearize_autograd(eval_f_linear, x0, u0, p)

            assert np.allclose(A_fd, expected_A, atol=atol, rtol=rtol), f"Finite Difference A mismatch: {A_fd} vs {expected_A}"
            assert np.allclose(B_fd, expected_B, atol=atol, rtol=rtol), f"Finite Difference B mismatch: {B_fd} vs {expected_B}"
            assert np.allclose(A_ag, expected_A, atol=atol, rtol=rtol), f"Autograd A mismatch: {A_ag} vs {expected_A}"
            assert np.allclose(B_ag, expected_B, atol=atol, rtol=rtol), f"Autograd B mismatch: {B_ag} vs {expected_B}"

        print(f"TEST PASSED ✅: Linearization of linear functions.")

    def test_eval_f_approx_equal(self):
        p = self.p_default
        p.dxFD = 1e-8
        x0 = TestVisualizeNetwork._make_3x3_state()
        u0 = 1.0

        A_fd, B_fd = linearize_FiniteDifference(eval_f, x0, u0, p)
        A_ag, B_ag = linearize_autograd(eval_f, x0, u0, p)

        atol = 1e-4
        rtol = 1e-4
        np.random.seed(1150)

        for _ in range(100):
            dx = np.random.uniform(-1e-2, 1e-2, x0.shape)
            du = np.random.uniform(-1e-2, 1e-2, 1).item()

            new_x = x0 + dx
            new_u = u0 + du
            new_u_with_1 = np.array([[1], [new_u]])

            f_actual = eval_f(new_x, p, new_u)
            f_linear_fd = A_fd @ new_x + B_fd @ new_u_with_1
            f_linear_ag = A_ag @ new_x + B_ag @ new_u_with_1

            assert np.allclose(f_actual, f_linear_fd, atol=atol, rtol=rtol), f"Finite Difference linearization mismatch: {f_actual} vs {f_linear_fd}"
            assert np.allclose(f_actual, f_linear_ag, atol=atol, rtol=rtol), f"Autograd linearization mismatch: {f_actual} vs {f_linear_ag}"

        print(f"TEST PASSED ✅: eval_f approximately equal to its linearization near x0.")

    def test_forward_euler_approx_equal(self):
        p = self.p_default
        p.dxFD = 1e-8
        x0 = TestVisualizeNetwork._make_3x3_state()
        
        dt = 5e-3
        total_time = 0.2
        u0 = eval_u_keytruda_input(dt)(0.0)
        print(f"Initial u0: {u0}")

        A_fd, B_fd = linearize_FiniteDifference(eval_f, x0, u0, p)
        A_ag, B_ag = linearize_autograd(eval_f, x0, u0, p)

        def linear_eval_f_fd(x, p, u):
            u_with_1 = np.array([[1], [u]])
            return A_fd @ x + B_fd @ u_with_1
        
        def linear_eval_f_ag(x, p, u):
            u_with_1 = np.array([[1], [u]])
            return A_ag @ x + B_ag @ u_with_1

        atol = 1e-4
        rtol = 1e-4

        soln_nonlinear, _ = SimpleSolver(eval_f, x0, p, eval_u_keytruda_input(dt), total_time / dt, dt, visualize=True, gif_file_name=self.figure_dir + "test1.gif")
        soln_linear_fd, _ = SimpleSolver(linear_eval_f_fd, x0, p, eval_u_keytruda_input(dt), total_time / dt, dt, visualize=True, gif_file_name=self.figure_dir + "test1_fd.gif")
        soln_linear_ag, _ = SimpleSolver(linear_eval_f_ag, x0, p, eval_u_keytruda_input(dt), total_time / dt, dt, visualize=True, gif_file_name=self.figure_dir + "test1_ag.gif")

        assert np.allclose(soln_nonlinear, soln_linear_fd, atol=atol, rtol=rtol), f"Finite Difference linearized forward Euler mismatch."
        assert np.allclose(soln_nonlinear, soln_linear_ag, atol=atol, rtol=rtol), f"Autograd linearized forward Euler mismatch."

        print(f"TEST PASSED ✅: Forward Euler with linearized system approximates nonlinear system.")

    def run_all_tests(self):
        print("--------------------------------")
        print("Running all tests in test_linearize.py...")
        print("--------------------------------")

        self.test_scalar_functions1()
        self.test_scalar_functions2()
        self.test_linear_functions()
        self.test_eval_f_approx_equal()
        self.test_forward_euler_approx_equal()

        print("--------------------------------")
        print("All tests completed.")
        print("--------------------------------")

if __name__ == "__main__":
    tester = TestLinearize()
    tester.run_all_tests()
