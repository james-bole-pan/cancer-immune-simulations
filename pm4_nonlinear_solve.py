import matplotlib.pyplot as plt
import numpy as np
from newtonNd import newtonNd
from eval_f import eval_f, Params
from eval_Jf_autograd import eval_Jf_autograd
import copy
from VisualizeNetwork import visualizeNetwork, create_network_evolution_gif

class RunPM4NonlinearSolve:

    def __init__(self):
        # assign imported functions
        self.eval_f = eval_f
        self.eval_Jf = eval_Jf_autograd

        # default parameter set
        self.p_default = Params(
            lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
            lambda_T=3.0, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
            d_A=0.0315, rows=1, cols=1
        )

        # solver settings
        self.u = 0
        self.errf = 1e-12
        self.errDeltax = 1e-12
        self.relDeltax = 1e-12
        self.MaxIter = 200
        self.visualize = 1

    def run_nonlinear_solve_logistic(self):
        print("Running nonlinear solve test for logistic growth model:")
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

        # Initial condition: all zeros
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = 15.0  # Cancer cells
        x0[1::3, 0] = 0.0  # T cells
        x0[2::3, 0] = 0.0  # Drug

        # turn shape of x0 into a vector
        x0 = x0.flatten()
        print(f"The shape of x0 is {x0.shape}")

        print(' ')
        print('Solving with provided eval_Jf Jacobian function')
        FiniteDifference=0;  
        x_AnalyticJacobian,converged,errf_k,errDeltax_k,relDeltax_k,iterations,X_an = newtonNd(
            self.eval_f,
            x0,
            p,
            self.u,
            self.errf,
            self.errDeltax,
            self.relDeltax,
            self.MaxIter,
            self.visualize,
            FiniteDifference,
            self.eval_Jf)
        plt.title('Intermediate Newton Solutions with provided eval Jf function')
        plt.show()
        print(f"Errors after {iterations} iterations: errf_k={errf_k}, errDeltax_k={errDeltax_k}, relDeltax_k={relDeltax_k}")
        print(f"Converged solution x:\n{x_AnalyticJacobian}")
        #visualizeNetwork(x_AnalyticJacobian.reshape(-1, 1), p, visualize=False, save=False)
        create_network_evolution_gif(X_an, p, save=False, show=True, fps=10)
        

    def run_nonlinear_solve_3x3_default(self):
        print("Running nonlinear solve test for logistic growth model:")
        p = copy.deepcopy(self.p_default)

        # --- grid setup ---
        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        # Initial guess
        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = 2.0  # Cancer cells
        x0[1::3, 0] = 1.0  # T cells
        x0[2::3, 0] = 0.0  # Drug

        print(f"Initial guess x0:\n{x0.flatten()}")

        # turn shape of x0 into a vector
        x0 = x0.flatten()
        print(f"The shape of x0 is {x0.shape}")

        print(' ')
        print('Solving with provided eval_Jf Jacobian function')
        FiniteDifference=0;  
        x_AnalyticJacobian,converged,errf_k,errDeltax_k,relDeltax_k,iterations,X_an = newtonNd(
            self.eval_f,
            x0,
            p,
            self.u,
            self.errf,
            self.errDeltax,
            self.relDeltax,
            self.MaxIter,
            self.visualize,
            FiniteDifference,
            self.eval_Jf)
        plt.title('Intermediate Newton Solutions with provided eval Jf function')
        plt.show()
        print(f"Errors after {iterations} iterations: errf_k={errf_k}, errDeltax_k={errDeltax_k}, relDeltax_k={relDeltax_k}")
        print(f"Converged solution x:\n{x_AnalyticJacobian}")
        create_network_evolution_gif(X_an, p, save=False, show=True, fps=10)

    def run_nonlinear_solve_T_cell_kill(self):
        print("Running nonlinear solve test for T cell killing model:")
        p = copy.deepcopy(self.p_default)
        p.lambda_C = 0; p.d_C = 0; p.D_C = 0
        p.lambda_T = 0; p.d_T = 0; p.k_A = 0; p.D_T = 0; p.K_K = 500
        p.d_A = 0; 

        rows, cols = 3, 3
        p.rows = rows
        p.cols = cols
        n_cells = rows * cols

        C0 = 10.0
        T_const = 1.0

        x0 = np.zeros((n_cells * 3, 1))
        x0[0::3, 0] = C0       # tumor in each cell
        x0[1::3, 0] = T_const  # constant T in each cell
        x0[2::3, 0] = 0.0      # no drug

        print(f"Initial guess x0:\n{x0.flatten()}")

        # turn shape of x0 into a vector
        x0 = x0.flatten()
        print(f"The shape of x0 is {x0.shape}")

        print(' ')
        print('Solving with provided eval_Jf Jacobian function')
        FiniteDifference=0;  
        x_AnalyticJacobian,converged,errf_k,errDeltax_k,relDeltax_k,iterations,X_an = newtonNd(
            self.eval_f,
            x0,
            p,
            self.u,
            self.errf,
            self.errDeltax,
            self.relDeltax,
            self.MaxIter,
            self.visualize,
            FiniteDifference,
            self.eval_Jf)
        plt.title('Intermediate Newton Solutions with provided eval Jf function')
        plt.show()
        print(f"Errors after {iterations} iterations: errf_k={errf_k}, errDeltax_k={errDeltax_k}, relDeltax_k={relDeltax_k}")
        print(f"Converged solution x:\n{x_AnalyticJacobian}")
        create_network_evolution_gif(X_an, p, save=False, show=True, fps=10)

    def run_nonlinear_solve_spatial_diffusion(self):
        print("Running nonlinear solve test for spatial diffusion model:")
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
        C0, T0 = 18.0, 9.0
        x0 = np.zeros((n_cells * 3, 1))
        center_idx = (rows // 2) * cols + (cols // 2)
        x0[3 * center_idx + 0, 0] = C0  # tumor in center
        x0[3 * center_idx + 1, 0] = T0  # T cells in center

        print(f"Initial guess x0:\n{x0.flatten()}")

        # turn shape of x0 into a vector
        x0 = x0.flatten()
        print(f"The shape of x0 is {x0.shape}")

        print(' ')
        print('Solving with provided eval_Jf Jacobian function')
        FiniteDifference=0;  
        x_AnalyticJacobian,converged,errf_k,errDeltax_k,relDeltax_k,iterations,X_an = newtonNd(
            self.eval_f,
            x0,
            p,
            self.u,
            self.errf,
            self.errDeltax,
            self.relDeltax,
            self.MaxIter,
            self.visualize,
            FiniteDifference,
            self.eval_Jf)
        plt.title('Intermediate Newton Solutions with provided eval Jf function')
        plt.show()
        print(f"Errors after {iterations} iterations: errf_k={errf_k}, errDeltax_k={errDeltax_k}, relDeltax_k={relDeltax_k}")
        print(f"Converged solution x:\n{x_AnalyticJacobian}")
        create_network_evolution_gif(X_an, p, save=False, show=True, fps=10)

    def run_fake_spatial_data_tumor(self):
        print("Running nonlinear solve test for fake spatial tumor data:")
        spatial_data = np.load("data/fake_spatial_data_tumor_int.npy")
        rows, cols, channels = spatial_data.shape

        p = copy.deepcopy(self.p_default)
        p.rows = rows
        p.cols = cols

        x0 = spatial_data.reshape(rows * cols * channels, 1)  # (n_state, 1)
        x0 = x0.flatten()
        print(f"The shape of x0 is {x0.shape}")

        print(' ')
        print('Solving with provided eval_Jf Jacobian function')
        FiniteDifference=0;  
        x_AnalyticJacobian,converged,errf_k,errDeltax_k,relDeltax_k,iterations,X_an = newtonNd(
            self.eval_f,
            x0,
            p,
            self.u,
            self.errf,
            self.errDeltax,
            self.relDeltax,
            self.MaxIter,
            self.visualize,
            FiniteDifference,
            self.eval_Jf)
        plt.title('Intermediate Newton Solutions with provided eval Jf function')
        plt.show()

    def run_all(self):
        #self.run_nonlinear_solve_logistic()
        #self.run_nonlinear_solve_3x3_default()
        #self.run_nonlinear_solve_T_cell_kill()
        self.run_nonlinear_solve_spatial_diffusion()

if __name__ == "__main__":
    runner = RunPM4NonlinearSolve()
    runner.run_all()