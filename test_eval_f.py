from eval_f import eval_f, Params
from scipy.special import lambertw
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
    
    def actual_drug_input(self, dose=200.0, interval=21.0):
        def r(t):
            return dose if (t % interval == 0.0) else 0.0
        return r

    def test_logistic_growth(self, w, num_iter):
        p = copy.deepcopy(self.p_default)
        # zero-out irrelevant terms
        p.d_C = 0; p.k_T = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0; p.D_T = 0
        p.d_A = 0

        # --- grid setup ---
        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        # Initial condition: each cell starts with C0 tumor cells
        C0 = 0.1
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0   # set C for each cell
        x0[1::3, 0] = 0.0  # T cells
        x0[2::3, 0] = 0.0  # Drug

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

        # --- Numerical total tumor count (sum over all cells) ---
        C_num_total = np.sum(X[0::3, :], axis=0)

        # --- Analytical total tumor count ---
        C_single = p.K_C / (1 + ((p.K_C - C0) / C0) * np.exp(-p.lambda_C * t))
        C_analytical_total = n_cells * C_single

        # --- Assertions ---
        assert C_num_total[-1] > C_num_total[0], "Total tumor count should grow under logistic-only dynamics."

        diff = np.linalg.norm(C_num_total - C_analytical_total) / np.linalg.norm(C_analytical_total)
        assert diff < 0.1, f"Numerical vs analytical total tumor count differ too much (rel error={diff:.2})"
        print(f"TEST PASSED ✅: Logistic growth test passed for w={w} with relative error {diff:.2}")

    def test_pure_decay(self, w, num_iter):
        p = copy.deepcopy(self.p_default)
        p.lambda_C = 0; p.k_T = 0; p.D_C = 0
        p.lambda_T = 0; p.k_A = 0; p.D_T = 0

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0 = 25.0
        T0 = 25.0
        A0 = 25.0
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0
        x0[1::3, 0] = T0
        x0[2::3, 0] = A0

        u_func = self.constant_input(0.0)

        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"{self.figure_dir}/pure_decay_w_{w}.gif"
        )

        # --- Numerical total counts ---
        C_num_total = np.sum(X[0::3, :], axis=0)
        T_num_total = np.sum(X[1::3, :], axis=0)
        A_num_total = np.sum(X[2::3, :], axis=0)

        # --- Analytical total counts ---
        C_analytical_total = n_cells * C0 * np.exp(-p.d_C * t)
        T_analytical_total = n_cells * T0 * np.exp(-p.d_T * t)
        A_analytical_total = n_cells * A0 * np.exp(-p.d_A * t)

        # --- Assertions ---
        assert C_num_total[-1] < C_num_total[0], "Tumor should decay under pure decay dynamics."
        assert T_num_total[-1] < T_num_total[0], "T cells should decay under pure decay dynamics."
        assert A_num_total[-1] < A_num_total[0], "Drug should decay under pure decay dynamics."

        # Numerical vs analytical check (within 10% relative error)
        diff_C = np.linalg.norm(C_num_total - C_analytical_total) / np.linalg.norm(C_analytical_total)
        diff_T = np.linalg.norm(T_num_total - T_analytical_total) / np.linalg.norm(T_analytical_total)
        diff_A = np.linalg.norm(A_num_total - A_analytical_total) / np.linalg.norm(A_analytical_total)

        assert diff_C < 0.1, f"C decay mismatch too large (rel error={diff_C:.2})"
        assert diff_T < 0.1, f"T decay mismatch too large (rel error={diff_T:.2})"
        assert diff_A < 0.1, f"A decay mismatch too large (rel error={diff_A:.2})"

        print(f"TEST PASSED ✅: Pure decay test passed for w={w} with errors: C={diff_C:.2}, T={diff_T:.2}, A={diff_A:.2}")

    def test_spatial_diffusion(self, w, num_iter):
        p = copy.deepcopy(self.p_default)
        # disable all growth, decay, and interactions
        p.lambda_C = 0; p.d_C = 0; p.k_T = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0
        p.d_A = 0

        p.D_C = 0.01   # tumor diffusion coefficient
        p.D_T = 0.1   # T cell diffusion coefficient

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        # Initial condition: put tumor cells (C) and T cells (T) only in the center cell
        C0, T0 = 10.0, 5.0
        x0 = np.zeros((n_cells * 3, 1))
        center_idx = (rows // 2) * cols + (cols // 2)
        x0[3 * center_idx + 0, 0] = C0  # tumor in center
        x0[3 * center_idx + 1, 0] = T0  # T cells in center

        u_func = self.constant_input(0.0)

        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"{self.figure_dir}/spatial_diffusion_w_{w}.gif"
        )

        # --- Numerical totals over time ---
        C_num_total = np.sum(X[0::3, :], axis=0)
        T_num_total = np.sum(X[1::3, :], axis=0)

        # --- Print before/after states (reshape into 3x3 grids) ---
        C_grid_initial = x0[0::3, 0].reshape(rows, cols)
        T_grid_initial = x0[1::3, 0].reshape(rows, cols)

        C_grid_final = X[0::3, -1].reshape(rows, cols)
        T_grid_final = X[1::3, -1].reshape(rows, cols)

        print("Initial Tumor (C) grid:\n", C_grid_initial)
        print("Final Tumor (C) grid:\n", C_grid_final)

        print("Initial T cell (T) grid:\n", T_grid_initial)
        print("Final T cell (T) grid:\n", T_grid_final)

        # --- Conservation assertions ---
        assert np.isclose(C_num_total[-1], C0, rtol=1e-6), \
            f"Tumor diffusion should conserve total: initial={C0}, final={C_num_total[-1]}"
        assert np.isclose(T_num_total[-1], T0, rtol=1e-6), \
            f"T cell diffusion should conserve total: initial={T0}, final={T_num_total[-1]}"

        print(f"TEST PASSED ✅: Spatial diffusion test passed for w={w}. "
            f"C initial={C0}, final={C_num_total[-1]}, "
            f"T initial={T0}, final={T_num_total[-1]}")
        
    def test_kT_killing_grid(self, w, num_iter):
        p = copy.deepcopy(self.p_default)

        p.lambda_C = 0; p.d_C = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0; p.D_T = 0
        p.d_A = 0

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0 = 1000.0
        T_const = 1.0

        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0       # tumor in each cell
        x0[1::3, 0] = T_const  # constant T in each cell
        x0[2::3, 0] = 0.0      # no drug

        u_func = self.constant_input(0.0)

        # --- numerical integration ---
        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"{self.figure_dir}/kT_killing_w_{w}.gif"
        )

        # sum across all tumor states
        C_num_total = np.sum(X[0::3, :], axis=0)

        # --- analytical solution per cell ---
        alpha = p.k_T * T_const
        K = p.K_K

        def C_analytical(t):
            return K * lambertw((C0 / K) * np.exp((C0 - alpha * t) / K)).real

        C_single = np.array([C_analytical(tt) for tt in t])
        C_total_analytical = n_cells * C_single

        # --- check closeness ---
        rel_err = np.linalg.norm(C_num_total - C_total_analytical) / np.linalg.norm(C_total_analytical)
        assert rel_err < 0.1, f"3x3 grid k_T killing mismatch (rel error={rel_err:.2})"

        print(f"TEST PASSED ✅: k_T killing test passed (relative error={rel_err:.2}); numerical tumor count is {C_num_total[-1]:.2f}, analytical is {C_total_analytical[-1]:.2f}")


    def test_lambda_T_recruitment(self, w, num_iter):
        p = copy.deepcopy(self.p_default)

        p.lambda_C = 0; p.d_C = 0; p.k_T = 0; p.D_C = 0
        p.d_T = 0; p.k_A = 0; p.D_T = 0
        p.d_A = 0

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        # --- initial conditions ---
        C0 = 6.0
        T0 = 0.0
        A0 = 0.0
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0       # tumor in each cell
        x0[1::3, 0] = T0       # T cells
        x0[2::3, 0] = A0       # drug

        u_func = self.constant_input(0.0)

        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"{self.figure_dir}/lambdaT_recruitment_w_{w}.gif"
        )

        # numerical total T cells (sum across all 9 cells)
        T_num_total = np.sum(X[1::3, :], axis=0)

        # analytical per-cell slope
        r = p.lambda_T * (C0 / (C0 + p.K_R))
        T_analytical_single = T0 + r * t

        # total analytical = 9 cells
        T_analytical_total = n_cells * T_analytical_single

        # check closeness
        rel_err = np.linalg.norm(T_num_total - T_analytical_total) / np.linalg.norm(T_analytical_total)
        assert rel_err < 0.1, f"λ_T recruitment mismatch (rel error={rel_err:.2})"

        print(f"TEST PASSED ✅: λ_T recruitment test passed (relative error={rel_err:.2}); numerical T count is {T_num_total[-1]:.2f}, analytical is {T_analytical_total[-1]:.2f}")
    
    def test_k_A_drug_boost(self, w, num_iter):
        p = copy.deepcopy(self.p_default)

        p.lambda_C = 0; p.d_C = 0; p.k_T = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.D_T = 0
        p.d_A = 0   # no drug decay, constant A

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0 = 0.0
        T0 = 1.0
        A0 = 10.0   # drug present in every grid
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0   # tumor
        x0[1::3, 0] = T0   # T cells
        x0[2::3, 0] = A0   # drug

        u_func = self.constant_input(0.0)

        # numerical solver
        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"{self.figure_dir}/kA_drug_boost_w_{w}.gif"
        )

        # numerical total T cells (sum across all 9 cells)
        T_num_total = np.sum(X[1::3, :], axis=0)

        # analytical growth rate per cell: exponential with rate (k_A*A)/(A+K_A)
        r = (p.k_A * A0) / (A0 + p.K_A)
        T_analytical_single = T0 * np.exp(r * t)

        # total analytical = 9 cells
        T_analytical_total = n_cells * T_analytical_single

        # check closeness
        rel_err = np.linalg.norm(T_num_total - T_analytical_total) / np.linalg.norm(T_analytical_total)
        assert rel_err < 0.1, f"k_A drug-boost mismatch (rel error={rel_err:.2})"

        print(f"TEST PASSED ✅: k_A drug-boost test passed (relative error={rel_err:.2}); "
            f"numerical T count is {T_num_total[-1]:.2f}, analytical is {T_analytical_total[-1]:.2f}")

    def test_drug_pulses_pk(self, w, num_iter):
        p = copy.deepcopy(self.p_default)
        p.lambda_C = 0; p.d_C = 0; p.k_T = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0; p.D_T = 0

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        x0 = np.zeros((n_cells * 3, 1))

        interval = 21.0
        dose = 200.0

        bolus = self.actual_drug_input(dose=dose, interval=interval)

        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=bolus,
            NumIter=num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"{self.figure_dir}/drug_pulses_w_{w}.gif",
        )

        A_num_total = np.sum(X[2::3, :], axis=0)

        # ---- discrete "analytical" trajectory that matches forward Euler + r(t_n)=dose/w ----
        # time grid is t_n = n*w, n = 0..NumIter
        Nsteps = len(t) - 1           # number of forward-Euler updates performed
        q = 1.0 - w * p.d_A           # Euler decay multiplier per step
        # indices of dosing steps: n such that t_n % interval == 0, and n <= Nsteps-1 (last eval step)
        dose_idx = [n for n in range(Nsteps) if ( (n * w) % interval ) == 0.0]

        # build A_single[n] for n=0..Nsteps iteratively
        A_single = np.zeros(Nsteps + 1)
        # A_single[0] = 0 initially (x0 starts with A=0)

        for n in range(Nsteps):
            if n in dose_idx:
                A_single[n+1] = q * A_single[n] + dose
            else:
                A_single[n+1] = q * A_single[n]

        # total across 9 cells
        A_total_analytical = n_cells * A_single

        # ---- closeness of the full trajectories ----
        rel_err = np.linalg.norm(A_num_total - A_total_analytical) / (np.linalg.norm(A_total_analytical) + 1e-12)
        assert rel_err < 1e-6, f"Drug pulse PK trajectory mismatch (rel error={rel_err:.2e})"

        print(f"TEST PASSED ✅: Drug pulse PK trajectory rel error = {rel_err:.2e}") 

    def run_all_tests(self, w, num_iter):
        self.test_logistic_growth(w, num_iter)
        self.test_pure_decay(w, num_iter)
        self.test_spatial_diffusion(w, num_iter)
        self.test_kT_killing_grid(w, num_iter)
        self.test_lambda_T_recruitment(w, num_iter)
        self.test_k_A_drug_boost(w, num_iter)
        self.test_drug_pulses_pk(w, num_iter)

if __name__ == "__main__":
    tester = TestEvalF()
    tester.run_all_tests(w=1, num_iter=84)
