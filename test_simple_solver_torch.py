import numpy as np
import torch
from pathlib import Path

from final_report_run_sgd_optimization_pytorch import (
    Params,
    eval_f_torch,
    SimpleSolver_torch,
)
from VisualizeNetwork import create_network_evolution_gif


class TestSimpleSolverTorch:
    def setup_method(self):
        self.device = torch.device("cpu")

    def constant_input(self, value: float):
        return lambda t: value

    def test_logistic_growth_matches_analytic(self):
        # Single-cell logistic growth: dC/dt = lambda_C * C * (1 - C/K_C)
        p = Params(
            lambda_C=0.33,
            K_C=28.0,
            d_C=0.0,
            k_T=0.0,
            K_K=1.0,
            D_C=0.0,
            lambda_T=0.0,
            K_T=1.0,
            K_R=1.0,
            d_T=0.0,
            k_A=0.0,
            K_A=1.0,
            D_T=0.0,
            d_A=0.0,
            rows=1,
            cols=1,
        )

        theta_pos = torch.tensor(
            [p.K_K, p.D_C, p.lambda_T, p.K_R, p.D_T], device=self.device
        )

        C0 = 0.1
        x0 = torch.tensor([[C0], [0.0], [0.0]], device=self.device)

        NumIter = 400
        w = 0.01
        u_func = self.constant_input(0.0)

        X, t_vec = SimpleSolver_torch(
            eval_f_torch,
            x_start=x0,
            theta_pos=theta_pos,
            p_fixed=p,
            eval_u=u_func,
            NumIter=NumIter,
            w=w,
            device=self.device,
        )

        C_num = X[0].cpu().numpy()
        t = t_vec.cpu().numpy()

        C_analytic = p.K_C / (1 + ((p.K_C - C0) / C0) * np.exp(-p.lambda_C * t))

        rel_error = np.linalg.norm(C_num - C_analytic) / np.linalg.norm(C_analytic)
        assert C_num[-1] > C_num[0], "Tumor should grow under logistic dynamics."
        assert rel_error < 0.05, f"Logistic growth mismatch too large (rel error={rel_error:.3f})"

    def test_pure_decay_matches_analytic(self):
        # Single-cell exponential decay for C, T, A with zero input and diffusion
        p = Params(
            lambda_C=0.0,
            K_C=1.0,
            d_C=0.3,
            k_T=0.0,
            K_K=1.0,
            D_C=0.0,
            lambda_T=0.0,
            K_T=1.0,
            K_R=1.0,
            d_T=0.2,
            k_A=0.0,
            K_A=1.0,
            D_T=0.0,
            d_A=0.4,
            rows=1,
            cols=1,
        )

        theta_pos = torch.tensor(
            [p.K_K, p.D_C, p.lambda_T, p.K_R, p.D_T], device=self.device
        )

        C0, T0, A0 = 5.0, 3.0, 2.0
        x0 = torch.tensor([[C0], [T0], [A0]], device=self.device)

        NumIter = 200
        w = 0.01
        u_func = self.constant_input(0.0)

        X, t_vec = SimpleSolver_torch(
            eval_f_torch,
            x_start=x0,
            theta_pos=theta_pos,
            p_fixed=p,
            eval_u=u_func,
            NumIter=NumIter,
            w=w,
            device=self.device,
        )

        t = t_vec.cpu().numpy()
        C_num = X[0].cpu().numpy()
        T_num = X[1].cpu().numpy()
        A_num = X[2].cpu().numpy()

        C_analytic = C0 * np.exp(-p.d_C * t)
        T_analytic = T0 * np.exp(-p.d_T * t)
        A_analytic = A0 * np.exp(-p.d_A * t)

        def rel_err(num, ref):
            return np.linalg.norm(num - ref) / np.linalg.norm(ref)

        assert C_num[-1] < C_num[0] and T_num[-1] < T_num[0] and A_num[-1] < A_num[0]
        assert rel_err(C_num, C_analytic) < 0.05, "C decay mismatch too large"
        assert rel_err(T_num, T_analytic) < 0.05, "T decay mismatch too large"
        assert rel_err(A_num, A_analytic) < 0.05, "A decay mismatch too large"

    def test_output_shapes_and_types(self):
        p = Params(
            lambda_C=0.1,
            K_C=10.0,
            d_C=0.0,
            k_T=0.0,
            K_K=1.0,
            D_C=0.0,
            lambda_T=0.0,
            K_T=1.0,
            K_R=1.0,
            d_T=0.0,
            k_A=0.0,
            K_A=1.0,
            D_T=0.0,
            d_A=0.0,
            rows=2,
            cols=2,
        )

        theta_pos = torch.tensor(
            [p.K_K, p.D_C, p.lambda_T, p.K_R, p.D_T], device=self.device
        )

        n_cells = p.rows * p.cols
        x0 = torch.zeros((n_cells * 3, 1), device=self.device)

        NumIter = 5
        w = 0.1
        u_func = self.constant_input(0.0)

        X, t_vec = SimpleSolver_torch(
            eval_f_torch,
            x_start=x0,
            theta_pos=theta_pos,
            p_fixed=p,
            eval_u=u_func,
            NumIter=NumIter,
            w=w,
            device=self.device,
        )

        assert isinstance(X, torch.Tensor) and isinstance(t_vec, torch.Tensor)
        assert X.shape == (n_cells * 3, NumIter + 1)
        assert t_vec.shape == (NumIter + 1,)
        assert X.device == self.device and t_vec.device == self.device

    def test_clinical_data_visualization(self):
        # Load fake clinical spatial data (20x20x3 int64)
        data = np.load("data/fake_spatial_data_tumor_int.npy")
        assert data.ndim == 3 and data.shape[2] >= 1, "Expected (rows, cols, channels)"

        rows, cols, _ = data.shape

        # Map data into solver state: use channel 0 as Cancer, zeros for T and A
        C0 = data[:, :, 0].astype(np.float32)
        T0 = data[:, :, 1].astype(np.float32)
        A0 = data[:, :, 2].astype(np.float32)
        x0_stack = np.stack([C0, T0, A0], axis=2)  # (rows, cols, 3)
        x0_flat = x0_stack.reshape(-1, 1)

        p = Params(
            lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
            lambda_T=3.0, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
            d_A=0.0315, rows=rows, cols=cols
        )

        theta_pos = torch.tensor(
            [p.K_K, p.D_C, p.lambda_T, p.K_R, p.D_T], device=self.device
        )

        x0 = torch.tensor(x0_flat, dtype=torch.float32, device=self.device)
        NumIter = 200
        w = 0.1
        u_func = self.constant_input(0.0)

        X, t_vec = SimpleSolver_torch(
            eval_f_torch,
            x_start=x0,
            theta_pos=theta_pos,
            p_fixed=p,
            eval_u=u_func,
            NumIter=NumIter,
            w=w,
            device=self.device,
        )

        # Create GIF from trajectory
        X_np = X.detach().cpu().numpy()
        # initial T cell count
        T0_initial = X_np[1::3, 0].reshape(p.rows, p.cols)
        print(f"Initial total T cells: {np.sum(T0_initial)}")
        # final T cell count
        T_final = X_np[1::3, -1].reshape(p.rows, p.cols)
        print(f"Final total T cells: {np.sum(T_final)}")
        out_dir = "test_clinical_data_visualization_torch"
        gif_path = create_network_evolution_gif(
            X_np,
            p,
            output_dir=out_dir,
            title_prefix="fake_clinical_torch",
            save=True,
            show=False,
            fps=10,
            dpi=120,
        )

        assert Path(gif_path).exists(), "Expected GIF output to be created"


if __name__ == "__main__":
    tester = TestSimpleSolverTorch()
    tester.setup_method()
    tester.test_logistic_growth_matches_analytic()
    tester.test_pure_decay_matches_analytic()
    tester.test_output_shapes_and_types()
    tester.test_clinical_data_visualization()
    print("All SimpleSolver_torch tests passed.")
