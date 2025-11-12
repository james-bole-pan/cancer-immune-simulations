import matplotlib.pyplot as plt
import numpy as np
from newtonNd import newtonNd
from eval_f import eval_f, Params
from eval_Jf_autograd import eval_Jf_autograd
import copy
from visualize_combined import DoubleVis
from SimpleSolver import SimpleSolver

class RunPM4CombinedVis:

    def __init__(self):
        self.eval_f = eval_f
        self.eval_Jf = eval_Jf_autograd
        self.figure_dir = "test_combined_vis_figs/"

        # default parameter set
        self.p_default = Params(
            lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
            lambda_T=3.0, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
            d_A=0.0315, rows=1, cols=1
        )

        # NewtonNd solver settings
        self.u = 0
        self.errf = 1e-12
        self.errDeltax = 1e-12
        self.relDeltax = 1e-12
        self.MaxIter = 200
        self.visualize = 0

    def constant_input(self, u_val):
        return lambda t: u_val
    
    def actual_drug_input(self, dose=200.0, interval=21.0):
        def r(t):
            return dose if (t % interval == 0.0) else 0.0
        return r

    # ------------------- Example: Logistic growth -------------------
    def run_nonlinear_solve_logistic(self, w, num_iter=100, cell_index=1):
        print("Running nonlinear solve test for logistic growth model:")
        p = copy.deepcopy(self.p_default)
        # zero-out irrelevant terms
        p.d_C = 0; p.k_T = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0; p.D_T = 0
        p.d_A = 0

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = 15.0
        x0[1::3, 0] = 0.0
        x0[2::3, 0] = 0.0

        u_func = self.constant_input(0.0)
        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=False,
            gif_file_name=f"{self.figure_dir}/logistic_growth_w_{w}.gif"
        )

        x0_flat = x0.flatten()
        c = 3
        x_newton, converged, errf_k, errDeltax_k, relDeltax_k, iterations, X_an = newtonNd(
            lambda x, p, u: c * self.eval_f(x, p, u),
            x0_flat,
            p,
            self.u,
            self.errf,
            self.errDeltax,
            self.relDeltax,
            self.MaxIter,
            self.visualize,
            0,
            lambda eval_f, x, p, u: c * self.eval_Jf(eval_f, x, p, u)
        )
        t_newton = np.arange(X_an.shape[1])
        print(f"Newton converged: {converged}, iterations: {iterations}")

        # DoubleVis
        save_path = f"{self.figure_dir}/logistic_growth_doublevis.png"
        DoubleVis(
            t, X, t_newton, X_an,
            grid_size=(rows, cols),
            cell_index=cell_index,
            save_path=save_path,
            w=w,
            newton_converged=converged,
            newton_iterations=iterations
        )

    # ------------------- 3x3 default -------------------
    def run_nonlinear_solve_3x3_default(self, cell_index=1, w=1, num_iter=100):
        print("Running nonlinear solve test for 3x3 default model:")
        p = copy.deepcopy(self.p_default)
        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = 2.0
        x0[1::3, 0] = 1.0
        x0[2::3, 0] = 0.0

        u_func = self.constant_input(0.0)
        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=False,
            gif_file_name=f"{self.figure_dir}/3x3_default_simple_w_{w}.gif"
        )

        x0_flat = x0.flatten()
        c = 3
        x_newton, converged, errf_k, errDeltax_k, relDeltax_k, iterations, X_an = newtonNd(
            lambda x, p, u: c * self.eval_f(x, p, u),
            x0_flat,
            p,
            self.u,
            self.errf,
            self.errDeltax,
            self.relDeltax,
            self.MaxIter,
            self.visualize,
            0,
            lambda eval_f, x, p, u: c * self.eval_Jf(eval_f, x, p, u)
        )
        t_newton = np.arange(X_an.shape[1])
        print(f"Newton converged: {converged}, iterations: {iterations}")

        save_path = f"{self.figure_dir}/3x3_default_doublevis.png"
        DoubleVis(
            t, X, t_newton, X_an,
            grid_size=(rows, cols),
            cell_index=cell_index,
            save_path=save_path,
            w=w,
            newton_converged=converged,
            newton_iterations=iterations
        )

    # ------------------- T cell kill -------------------
    def run_nonlinear_solve_T_cell_kill(self, cell_index=1, w=1, num_iter=100):
        print("Running nonlinear solve test for T cell killing model:")
        p = copy.deepcopy(self.p_default)
        p.lambda_C = 0; p.d_C = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0; p.D_T = 0; p.K_K = 500
        p.d_A = 0
        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0 = 10.0
        T_const = 1.0
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0
        x0[1::3, 0] = T_const
        x0[2::3, 0] = 0.0

        u_func = self.constant_input(0.0)
        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=False,
            gif_file_name=f"{self.figure_dir}/T_cell_kill_simple_w_{w}.gif"
        )

        x0_flat = x0.flatten()
        c = 3
        x_newton, converged, errf_k, errDeltax_k, relDeltax_k, iterations, X_an = newtonNd(
            lambda x, p, u: c * self.eval_f(x, p, u),
            x0_flat,
            p,
            self.u,
            self.errf,
            self.errDeltax,
            self.relDeltax,
            self.MaxIter,
            self.visualize,
            0,
            lambda eval_f, x, p, u: c * self.eval_Jf(eval_f, x, p, u)
        )
        t_newton = np.arange(X_an.shape[1])
        print(f"Newton converged: {converged}, iterations: {iterations}")

        save_path = f"{self.figure_dir}/T_cell_kill_doublevis.png"
        DoubleVis(
            t, X, t_newton, X_an,
            grid_size=(rows, cols),
            cell_index=cell_index,
            save_path=save_path,
            w=w,
            newton_converged=converged,
            newton_iterations=iterations
        )

    # ------------------- Spatial diffusion -------------------
    def run_nonlinear_solve_spatial_diffusion(self, cell_index=1, w=1, num_iter=100):
        print("Running nonlinear solve test for spatial diffusion model:")
        p = copy.deepcopy(self.p_default)
        p.lambda_C = 0; p.d_C = 0; p.k_T = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0
        p.d_A = 0
        p.D_C = 0.01
        p.D_T = 0.1

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0, T0 = 18.0, 9.0
        x0 = np.zeros((n_cells * 3, 1))
        center_idx = (rows // 2) * cols + (cols // 2)
        x0[3 * center_idx + 0, 0] = C0
        x0[3 * center_idx + 1, 0] = T0

        u_func = self.constant_input(0.0)
        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=False,
            gif_file_name=f"{self.figure_dir}/spatial_diffusion_simple_w_{w}.gif"
        )

        x0_flat = x0.flatten()
        c = 3
        x_newton, converged, errf_k, errDeltax_k, relDeltax_k, iterations, X_an = newtonNd(
            lambda x, p, u: c * self.eval_f(x, p, u),
            x0_flat,
            p,
            self.u,
            self.errf,
            self.errDeltax,
            self.relDeltax,
            self.MaxIter,
            self.visualize,
            0,
            lambda eval_f, x, p, u: c * self.eval_Jf(eval_f, x, p, u)
        )
        t_newton = np.arange(X_an.shape[1])
        print(f"Newton converged: {converged}, iterations: {iterations}")

        save_path = f"{self.figure_dir}/spatial_diffusion_doublevis.png"
        DoubleVis(
            t, X, t_newton, X_an,
            grid_size=(rows, cols),
            cell_index=cell_index,
            save_path=save_path,
            w=w,
            newton_converged=converged,
            newton_iterations=iterations
        )

    # ------------------- C and T growth -------------------
    def run_nonlinear_solve_3x3_C_and_T_growth(self, cell_index=1, w=1, num_iter=100):
        print("Running nonlinear solve test for C and T growth model:")
        p = copy.deepcopy(self.p_default)
        p.k_T = 0.05
        p.lambda_T = 5
        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = 24.0
        x0[1::3, 0] = 15.0
        x0[2::3, 0] = 1.0

        u_func = self.constant_input(0.0)
        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=u_func,
            NumIter=num_iter,
            w=w,
            visualize=False,
            gif_file_name=f"{self.figure_dir}/C_T_growth_simple_w_{w}.gif"
        )

        x0_flat = x0.flatten()
        c = 3
        x_newton, converged, errf_k, errDeltax_k, relDeltax_k, iterations, X_an = newtonNd(
            lambda x, p, u: c * self.eval_f(x, p, u),
            x0_flat,
            p,
            self.u,
            self.errf,
            self.errDeltax,
            self.relDeltax,
            self.MaxIter,
            self.visualize,
            0,
            lambda eval_f, x, p, u: c * self.eval_Jf(eval_f, x, p, u)
        )
        t_newton = np.arange(X_an.shape[1])
        print(f"Newton converged: {converged}, iterations: {iterations}")

        save_path = f"{self.figure_dir}/C_T_growth_doublevis.png"
        DoubleVis(
            t, X, t_newton, X_an,
            grid_size=(rows, cols),
            cell_index=cell_index,
            save_path=save_path,
            w=w,
            newton_converged=converged,
            newton_iterations=iterations
        )

    # ------------------- Run all -------------------
    def run_all(self, w=1, num_iter=100, cell_index=1):
        self.run_nonlinear_solve_logistic(cell_index=cell_index, w=w, num_iter=num_iter)
        self.run_nonlinear_solve_3x3_default(cell_index=cell_index, w=w, num_iter=num_iter)
        self.run_nonlinear_solve_T_cell_kill(cell_index=cell_index, w=w, num_iter=num_iter)
        self.run_nonlinear_solve_spatial_diffusion(cell_index=cell_index, w=w, num_iter=num_iter)
        self.run_nonlinear_solve_3x3_C_and_T_growth(cell_index=cell_index, w=w, num_iter=num_iter)

if __name__ == "__main__":
    runner = RunPM4CombinedVis()
    runner.run_all(w=1, num_iter=100, cell_index=1)
