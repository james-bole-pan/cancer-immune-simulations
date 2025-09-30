from eval_f import eval_f, Params
import numpy as np
from SimpleSolver import SimpleSolver
import os
import copy

class TestEvalF:
    def __init__(self):
        self.p_default = Params(
            lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
            lambda_T=3.0, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
            d_A=0.0315, rows=1, cols=1
        )
        self.figure_dir = "test_evalf_output_figures/"
        os.makedirs(self.figure_dir, exist_ok=True)

    def constant_input(self, u_val):
        return lambda t: u_val

    def test_logistic_growth(self, w, num_iter):
        p = copy.deepcopy(self.p_default)
        # zero-out irrelevant terms
        p.d_C = 0; p.k_T = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0; p.D_T = 0
        p.d_A = 0

        # Column vector (3,1): [C, T, A]
        C0 = 0.1
        x0 = np.array([[C0],
                    [0.0],
                    [0.0]])

        u_func = self.constant_input(0.0)

        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"{self.figure_dir}/logistic_growth_w_{w}.gif"
        )

        # Numerical trajectory
        C_num = X[0, :]

        # Analytical logistic solution
        lam = p.lambda_C
        K = p.K_C
        C_analytical = K / (1 + ((K - C0) / C0) * np.exp(-lam * t))

        # --- Assertions ---
        # Growth check
        assert C_num[-1] > C_num[0], "Tumor should grow under logistic-only dynamics."

        # Match check (within tolerance)
        diff = np.linalg.norm(C_num - C_analytical) / np.linalg.norm(C_analytical)
        assert diff < 0.1, f"Numerical and analytical logistic growth differ too much (rel error={diff:.2e})"
        print(f"Logistic growth test passed for w={w} with relative error {diff:.2e}")

    def test_pure_decay(self, w, num_iter):
        # Add later
        pass

    def run_all_tests(self, w, num_iter):
        self.test_logistic_growth(w, num_iter)
        self.test_pure_decay(w, num_iter)

if __name__ == "__main__":
    tester = TestEvalF()
    tester.run_all_tests(w=1, num_iter=84)
