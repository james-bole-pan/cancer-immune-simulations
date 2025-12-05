from eval_f import eval_f, Params
from scipy.special import lambertw
import numpy as np
from SimpleSolver import SimpleSolver
from SimpleSolver_autograd import SimpleSolver_autograd
from eval_u_keytruda_input import eval_u_keytruda_input
import matplotlib.pyplot as plt
import os
import copy

# Optional toy-gradient test
import autograd.numpy as anp
from autograd import grad


class TestEvalF:
    def __init__(self, visualize_default=True, figure_dir="test_evalf_output_figures_autograd/"):
        self.p_default = Params(
            lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
            lambda_T=3.0, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
            d_A=0.0315, rows=1, cols=1
        )
        self.figure_dir = figure_dir
        os.makedirs(self.figure_dir, exist_ok=True)

        # Global default for whether tests will generate gifs
        self.visualize_default = visualize_default

    # -----------------------------
    # Visualization resolver
    # -----------------------------
    def _viz(self, visualize):
        """Resolve per-test visualize override vs class default."""
        return self.visualize_default if visualize is None else bool(visualize)

    # -----------------------------
    # Input functions
    # -----------------------------
    def constant_input(self, u_val):
        return lambda t: u_val
    
    # kept for completeness; not used in drug PK test anymore
    def actual_drug_input(self, dose=200.0, interval=21.0):
        def r(t):
            return dose if (t % interval == 0.0) else 0.0
        return r

    # -----------------------------
    # Core helper: run BOTH solvers
    # -----------------------------
    def run_both(self, p, x0, u_func, num_iter, w, visualize=None, tag=""):
        """
        Run numpy SimpleSolver and autograd SimpleSolver_autograd
        with identical inputs and return both outputs.

        NOTE:
        - This assumes u_func is NOT stateful across calls.
          For stateful inputs like eval_u_keytruda_input, do not use this helper.
        """
        visualize = self._viz(visualize)

        # Deepcopy params so each solver can safely mutate internal fields if needed
        p_np = copy.deepcopy(p)
        p_ag = copy.deepcopy(p)

        # Keep a copy of initial condition to test immutability
        x0_before = np.array(x0, copy=True)

        X_np, t_np = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p_np,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=visualize,
            gif_file_name=f"{self.figure_dir}/{tag}_np.gif" if visualize else "ignore.gif"
        )

        X_ag, t_ag = SimpleSolver_autograd(
            eval_f,
            x_start=x0,
            p=p_ag,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=visualize,
            gif_file_name=f"{self.figure_dir}/{tag}_ag.gif" if visualize else "ignore.gif"
        )

        # --- Input immutability check ---
        assert np.allclose(x0_before, x0), "x_start (x0) was modified in-place by a solver."

        # --- Shape checks ---
        assert X_np.shape == X_ag.shape, f"Shape mismatch: np {X_np.shape} vs ag {X_ag.shape}"
        assert t_np.shape == t_ag.shape, f"Time shape mismatch: np {t_np.shape} vs ag {t_ag.shape}"

        # --- Finite checks ---
        assert np.isfinite(X_np).all(), "NumPy solver produced NaN/Inf"
        assert np.isfinite(X_ag).all(), "Autograd solver produced NaN/Inf"

        # --- Time grid correctness ---
        expected_t = np.arange(num_iter + 1) * w
        assert np.allclose(t_np, expected_t), "NumPy time grid unexpected"
        assert np.allclose(t_ag, expected_t), "Autograd time grid unexpected"

        return X_np, t_np, X_ag, t_ag

    def assert_close_np_ag(self, X_np, X_ag, rtol=1e-8, atol=1e-10, msg=""):
        """Strict numerical equivalence check for forward Euler trajectories."""
        assert np.allclose(X_np, X_ag, rtol=rtol, atol=atol), \
            f"NumPy vs Autograd trajectory mismatch. {msg}"

    # -----------------------------
    # Basic mechanical tests
    # -----------------------------
    def test_shapes_determinism_and_equivalence(self, w, num_iter, visualize=None):
        p = copy.deepcopy(self.p_default)
        p.rows, p.cols = 2, 2
        n_cells = p.rows * p.cols

        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = 0.5
        x0[1::3, 0] = 0.2
        x0[2::3, 0] = 1.0

        u_func = self.constant_input(0.0)

        X_np1, t_np1, X_ag1, t_ag1 = self.run_both(
            p, x0, u_func, num_iter, w, visualize=visualize, tag="basic1"
        )
        X_np2, t_np2, X_ag2, t_ag2 = self.run_both(
            p, x0, u_func, num_iter, w, visualize=visualize, tag="basic2"
        )

        # Determinism
        assert np.allclose(X_np1, X_np2), "NumPy solver not deterministic"
        assert np.allclose(X_ag1, X_ag2), "Autograd solver not deterministic"

        # Equivalence
        self.assert_close_np_ag(X_np1, X_ag1, msg="basic equivalence")

        print(f"TEST PASSED ✅: Shapes/determinism/equivalence for w={w}")

    def test_random_small_grids_equivalence(self, w, num_iter, seed=0, visualize=None):
        rng = np.random.default_rng(seed)

        for rows, cols in [(1, 3), (3, 1), (2, 2), (3, 3)]:
            p = copy.deepcopy(self.p_default)
            p.rows, p.cols = rows, cols
            n_cells = rows * cols

            x0 = rng.uniform(low=0.0, high=1.0, size=(n_cells * 3, 1))
            u_func = self.constant_input(0.0)

            X_np, t_np, X_ag, t_ag = self.run_both(
                p, x0, u_func, num_iter, w, visualize=visualize, tag=f"rand_{rows}x{cols}"
            )
            self.assert_close_np_ag(X_np, X_ag, msg=f"random grid {rows}x{cols}")

        print(f"TEST PASSED ✅: Random small grid equivalence for w={w}")

    # -----------------------------
    # Physics/biology tests
    # -----------------------------
    def test_logistic_growth(self, w, num_iter, visualize=None):
        p = copy.deepcopy(self.p_default)
        # zero-out irrelevant terms
        p.d_C = 0; p.k_T = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0; p.D_T = 0
        p.d_A = 0

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0 = 0.1
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0
        x0[1::3, 0] = 0.0
        x0[2::3, 0] = 0.0

        u_func = self.constant_input(0.0)

        X_np, t_np, X_ag, t_ag = self.run_both(
            p, x0, u_func, num_iter, w, visualize=visualize, tag=f"logistic_w_{w}"
        )
        self.assert_close_np_ag(X_np, X_ag, msg="logistic forward match")
        t = t_np

        C_num_total = np.sum(X_np[0::3, :], axis=0)

        C_single = p.K_C / (1 + ((p.K_C - C0) / C0) * np.exp(-p.lambda_C * t))
        C_analytical_total = n_cells * C_single

        assert C_num_total[-1] > C_num_total[0], \
            "Total tumor count should grow under logistic-only dynamics."

        diff = np.linalg.norm(C_num_total - C_analytical_total) / np.linalg.norm(C_analytical_total)
        assert diff < 0.1, \
            f"Numerical vs analytical total tumor count differ too much (rel error={diff:.2})"

        print(f"TEST PASSED ✅: Logistic growth test passed for w={w} with relative error {diff:.2}")

    def test_pure_decay(self, w, num_iter, visualize=None):
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

        X_np, t_np, X_ag, t_ag = self.run_both(
            p, x0, u_func, num_iter, w, visualize=visualize, tag=f"decay_w_{w}"
        )
        self.assert_close_np_ag(X_np, X_ag, msg="decay forward match")
        t = t_np

        C_num_total = np.sum(X_np[0::3, :], axis=0)
        T_num_total = np.sum(X_np[1::3, :], axis=0)
        A_num_total = np.sum(X_np[2::3, :], axis=0)

        C_analytical_total = n_cells * C0 * np.exp(-p.d_C * t)
        T_analytical_total = n_cells * T0 * np.exp(-p.d_T * t)
        A_analytical_total = n_cells * A0 * np.exp(-p.d_A * t)

        assert C_num_total[-1] < C_num_total[0]
        assert T_num_total[-1] < T_num_total[0]
        assert A_num_total[-1] < A_num_total[0]

        diff_C = np.linalg.norm(C_num_total - C_analytical_total) / np.linalg.norm(C_analytical_total)
        diff_T = np.linalg.norm(T_num_total - T_analytical_total) / np.linalg.norm(T_analytical_total)
        diff_A = np.linalg.norm(A_num_total - A_analytical_total) / np.linalg.norm(A_analytical_total)

        assert diff_C < 0.1, f"C decay mismatch too large (rel error={diff_C:.2})"
        assert diff_T < 0.1, f"T decay mismatch too large (rel error={diff_T:.2})"
        assert diff_A < 0.1, f"A decay mismatch too large (rel error={diff_A:.2})"

        print(f"TEST PASSED ✅: Pure decay test passed for w={w} with errors: "
              f"C={diff_C:.2}, T={diff_T:.2}, A={diff_A:.2}")

    def test_spatial_diffusion(self, w, num_iter, visualize=None):
        p = copy.deepcopy(self.p_default)
        # disable all growth, decay, and interactions
        p.lambda_C = 0; p.d_C = 0; p.k_T = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0
        p.d_A = 0

        p.D_C = 0.01
        p.D_T = 0.1

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0, T0 = 10.0, 5.0
        x0 = np.zeros((n_cells * 3, 1))
        center_idx = (rows // 2) * cols + (cols // 2)
        x0[3 * center_idx + 0, 0] = C0
        x0[3 * center_idx + 1, 0] = T0

        u_func = self.constant_input(0.0)

        X_np, t_np, X_ag, t_ag = self.run_both(
            p, x0, u_func, num_iter, w, visualize=visualize, tag=f"diffusion_w_{w}"
        )
        self.assert_close_np_ag(X_np, X_ag, msg="diffusion forward match")

        C_num_total = np.sum(X_np[0::3, :], axis=0)
        T_num_total = np.sum(X_np[1::3, :], axis=0)

        assert np.isclose(C_num_total[-1], C0, rtol=1e-6), \
            f"Tumor diffusion should conserve total: initial={C0}, final={C_num_total[-1]}"
        assert np.isclose(T_num_total[-1], T0, rtol=1e-6), \
            f"T cell diffusion should conserve total: initial={T0}, final={T_num_total[-1]}"

        print(f"TEST PASSED ✅: Spatial diffusion test passed for w={w}.")

    def test_kT_killing_grid(self, w, num_iter, visualize=None):
        p = copy.deepcopy(self.p_default)

        p.lambda_C = 0; p.d_C = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0; p.D_T = 0; p.K_K = 500
        p.d_A = 0

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0 = 1000.0
        T_const = 1.0

        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0
        x0[1::3, 0] = T_const
        x0[2::3, 0] = 0.0

        u_func = self.constant_input(0.0)

        X_np, t_np, X_ag, t_ag = self.run_both(
            p, x0, u_func, num_iter, w, visualize=visualize, tag=f"kT_kill_w_{w}"
        )
        self.assert_close_np_ag(X_np, X_ag, msg="kT killing forward match")
        t = t_np

        C_num_total = np.sum(X_np[0::3, :], axis=0)

        alpha = p.k_T * T_const
        K = p.K_K

        def C_analytical(tt):
            return K * lambertw((C0 / K) * np.exp((C0 - alpha * tt) / K)).real

        C_single = np.array([C_analytical(tt) for tt in t])
        C_total_analytical = n_cells * C_single

        rel_err = np.linalg.norm(C_num_total - C_total_analytical) / np.linalg.norm(C_total_analytical)
        assert rel_err < 0.1, f"3x3 grid k_T killing mismatch (rel error={rel_err:.2})"

        print(f"TEST PASSED ✅: k_T killing test passed (relative error={rel_err:.2})")

    def test_lambda_T_recruitment(self, w, num_iter, visualize=None):
        p = copy.deepcopy(self.p_default)

        p.lambda_C = 0; p.d_C = 0; p.k_T = 0; p.D_C = 0
        p.d_T = 0; p.k_A = 0; p.D_T = 0
        p.d_A = 0

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0 = 6.0
        T0 = 0.0
        A0 = 0.0
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0
        x0[1::3, 0] = T0
        x0[2::3, 0] = A0

        u_func = self.constant_input(0.0)

        X_np, t_np, X_ag, t_ag = self.run_both(
            p, x0, u_func, num_iter, w, visualize=visualize, tag=f"lambdaT_w_{w}"
        )
        self.assert_close_np_ag(X_np, X_ag, msg="lambda_T forward match")
        t = t_np

        T_num_total = np.sum(X_np[1::3, :], axis=0)

        r = p.lambda_T * (C0 / (C0 + p.K_R))
        T_analytical_single = p.K_T + (T0 - p.K_T) * np.exp((-r/p.K_T) * t)
        T_analytical_total = n_cells * T_analytical_single

        rel_err = np.linalg.norm(T_num_total - T_analytical_total) / np.linalg.norm(T_analytical_total)
        assert rel_err < 0.1, f"λ_T recruitment mismatch (rel error={rel_err:.2})"

        print(f"TEST PASSED ✅: λ_T recruitment test passed (relative error={rel_err:.2})")

    def test_k_A_drug_boost(self, w, num_iter, visualize=None):
        p = copy.deepcopy(self.p_default)

        p.lambda_C = 0; p.d_C = 0; p.k_T = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.D_T = 0
        p.d_A = 0  # constant A everywhere

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0 = 0.0
        T0 = 1.0
        A0 = 10.0
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0
        x0[1::3, 0] = T0
        x0[2::3, 0] = A0

        u_func = self.constant_input(0.0)

        X_np, t_np, X_ag, t_ag = self.run_both(
            p, x0, u_func, num_iter, w, visualize=visualize, tag=f"kA_w_{w}"
        )
        self.assert_close_np_ag(X_np, X_ag, msg="k_A forward match")
        t = t_np

        T_num_total = np.sum(X_np[1::3, :], axis=0)

        r = (p.k_A * A0) / (A0 + p.K_A)
        T_analytical_single = T0 * np.exp(r * t)
        T_analytical_total = n_cells * T_analytical_single

        rel_err = np.linalg.norm(T_num_total - T_analytical_total) / np.linalg.norm(T_analytical_total)
        assert rel_err < 0.1, f"k_A drug-boost mismatch (rel error={rel_err:.2})"

        print(f"TEST PASSED ✅: k_A drug-boost test passed (relative error={rel_err:.2})")

    def test_drug_pulses_pk(self, w, num_iter, visualize=None):
        """
        Drug PK test using eval_u_keytruda_input (stateful).
        We DO NOT use run_both here to avoid shared-state issues.
        """
        visualize = self._viz(visualize)

        p = copy.deepcopy(self.p_default)

        # isolate drug dynamics
        p.lambda_C = 0; p.d_C = 0; p.k_T = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0; p.D_T = 0

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        x0 = np.zeros((n_cells * 3, 1))

        interval = 21.0
        dose = 200.0

        # IMPORTANT: separate stateful u funcs
        u_np = eval_u_keytruda_input(w=w, dose=dose, interval=interval)
        u_ag = eval_u_keytruda_input(w=w, dose=dose, interval=interval)

        X_np, t_np = SimpleSolver(
            eval_f,
            x_start=x0,
            p=copy.deepcopy(p),
            eval_u=u_np,
            NumIter=num_iter,
            w=w,
            visualize=visualize,
            gif_file_name=f"{self.figure_dir}/drug_pulses_keytruda_np_w_{w}.gif" if visualize else "ignore.gif",
        )

        X_ag, t_ag = SimpleSolver_autograd(
            eval_f,
            x_start=x0,
            p=copy.deepcopy(p),
            eval_u=u_ag,
            NumIter=num_iter,
            w=w,
            visualize=visualize,
            gif_file_name=f"{self.figure_dir}/drug_pulses_keytruda_ag_w_{w}.gif" if visualize else "ignore.gif",
        )

        assert np.allclose(t_np, t_ag), "Time grids differ between NumPy and Autograd solvers."

        A_num_total_np = np.sum(X_np[2::3, :], axis=0)
        A_num_total_ag = np.sum(X_ag[2::3, :], axis=0)

        # solvers agree
        assert np.allclose(A_num_total_np, A_num_total_ag, rtol=1e-8, atol=1e-10), \
            "Drug PK trajectory mismatch between NumPy and Autograd solvers."

        # Build discrete expected trajectory matching Euler update and the SAME u(t_n)
        u_anal = eval_u_keytruda_input(w=w, dose=dose, interval=interval)

        Nsteps = len(t_np) - 1
        A_single = np.zeros(Nsteps + 1)

        for n in range(Nsteps):
            r_n = u_anal(t_np[n])
            A_single[n+1] = A_single[n] + w * (r_n - p.d_A * A_single[n])

        A_total_analytical = n_cells * A_single

        rel_err = np.linalg.norm(A_num_total_np - A_total_analytical) / (np.linalg.norm(A_total_analytical) + 1e-12)
        assert rel_err < 1e-8, f"Drug pulse PK mismatch vs discrete expected (rel error={rel_err:.2e})"

        print(f"TEST PASSED ✅: Drug pulse PK using eval_u_keytruda_input passed for w={w} "
              f"(NumPy==Autograd and matches discrete expected; rel err={rel_err:.2e})")

    def test_sim_one_grid(self, w, num_iter, visualize=None):
        p = copy.deepcopy(self.p_default)
        p.D_C = 0.0
        p.D_T = 0.0

        rows, cols = 1, 1
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0 = 4
        T0 = 0.5
        A0 = 10.0 
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0 
        x0[1::3, 0] = T0 
        x0[2::3, 0] = A0 

        u_func = self.constant_input(0.0)

        X_np, t_np, X_ag, t_ag = self.run_both(
            p, x0, u_func, num_iter, w, visualize=visualize, tag=f"onegrid_w_{w}"
        )
        self.assert_close_np_ag(X_np, X_ag, msg="one-grid forward match")

        A_num = X_np[2, :]
        assert A_num[-1] < A0, "Drug A should decay with d_A > 0 and r_A = 0."

        initial_dC_dt = eval_f(x0, p, r_A=0.0)[0, 0]
        assert initial_dC_dt > 0, "Initial dC/dt must be positive for default-like params."

        initial_dT_dt = eval_f(x0, p, r_A=0.0)[1, 0]
        assert initial_dT_dt > 0, "Initial dT/dt must be positive for default-like params."

        print(f"TEST PASSED ✅: One-grid dynamics sanity test passed for w={w}")

    # -----------------------------
    # Optional gradient sanity on a toy system
    # -----------------------------
    def test_toy_gradient_wrt_param(self):
        """
        This does NOT test your biological eval_f.
        It tests whether SimpleSolver_autograd can be placed inside an autograd graph
        for a trivial linear system.

        If this fails due to in-place writes in X, you'll know immediately.
        """
        class ToyP:
            def __init__(self, a):
                self.a = a

        def toy_eval_f(x_col, p, u):
            return p.a * x_col

        def u0(t):
            return 0.0

        def loss(a_val):
            p = ToyP(a_val)
            x0 = anp.array([[1.0]])
            X, t = SimpleSolver_autograd(
                toy_eval_f, x0, p, u0, NumIter=10, w=0.1, visualize=False
            )
            xf = X[0, -1]
            return (xf - 2.0) ** 2

        try:
            g = grad(loss)(1.0)
            assert anp.isfinite(g), "Gradient is not finite in toy test."
            print(f"TEST PASSED ✅: Toy gradient sanity check (grad={float(g):.4f})")
        except Exception as e:
            print("⚠️ Toy gradient test raised an exception. "
                  "This may be due to in-place writes in X.\n"
                  f"Exception: {repr(e)}")

    # -----------------------------
    # Run all tests at a given w
    # -----------------------------
    def run_all_tests(self, w, num_iter, visualize=None):
        self.test_shapes_determinism_and_equivalence(w, num_iter, visualize=visualize)
        self.test_random_small_grids_equivalence(w, num_iter, visualize=visualize)

        self.test_logistic_growth(w, num_iter, visualize=visualize)
        self.test_pure_decay(w, num_iter, visualize=visualize)
        self.test_spatial_diffusion(w, num_iter, visualize=visualize)
        self.test_kT_killing_grid(w, num_iter, visualize=visualize)
        self.test_lambda_T_recruitment(w, num_iter, visualize=visualize)
        self.test_k_A_drug_boost(w, num_iter, visualize=visualize)
        self.test_drug_pulses_pk(w, num_iter, visualize=visualize)
        self.test_sim_one_grid(w, num_iter, visualize=visualize)

        self.test_toy_gradient_wrt_param()

    # -----------------------------
    # Sweep multiple w values
    # -----------------------------
    def run_sweep(self, w_list=(1.0, 0.1, 0.01), num_iter=84, visualize=None):
        for w in w_list:
            print("\n" + "=" * 80)
            print(f"Running full test suite with w={w}, NumIter={num_iter}")
            print("=" * 80)
            self.run_all_tests(w=w, num_iter=num_iter, visualize=visualize)


if __name__ == "__main__":
    # Option 1: default visualize OFF, enable per run
    # tester = TestEvalF()
    # tester.run_all_tests(w=1, num_iter=84, visualize=True)

    # Option 2: default visualize ON for everything
    tester = TestEvalF(visualize_default=False)

    # Run a single suite
    tester.run_all_tests(w=1, num_iter=84)

    # Or sweep step sizes
    # tester.run_sweep(w_list=(1.0, 0.1, 0.01), num_iter=84)
