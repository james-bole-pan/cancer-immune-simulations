# jacobian_two_methods.py
import os
import numpy as np
import matplotlib.pyplot as plt

from eval_f import eval_f, Params
from eval_Jf_autograd import eval_Jf_autograd
from eval_Jf_FiniteDifference import eval_Jf_FiniteDifference


class JacobianComparator:
    """
    Compare Jacobians from Autograd vs Finite Differences over a range of dx,
    report errors, and produce a convergence plot.

    Usage:
        jc = JacobianComparator()
        jc.run()                           # uses default dx grid
        jc.plot(save_dir="test_evalf_output_figures", show=False)
        # or customize:
        dx_vals = 10.0 ** np.arange(-16, 1, 0.25)
        jc.run(dx_values=dx_vals)
    """

    def __init__(
        self,
        rows: int = 2,
        cols: int = 2,
        interleaved_order: bool = True,
        x0_C: float = 15.0,
        x0_T: float = 2.0,
        x0_A: float = 0.0,
        params: Params | None = None,
        u: float = 0.0,
    ):
        self.rows = rows
        self.cols = cols
        self.dof_per_cell = 3  # [C, T, A]
        self.N = rows * cols * self.dof_per_cell
        self.u = u

        # State init (interleaved by default)
        x0_flat = np.zeros(self.N, dtype=float)
        if interleaved_order:
            x0_flat[0::3] = x0_C  # C
            x0_flat[1::3] = x0_T  # T
            x0_flat[2::3] = x0_A  # A
        else:
            # species-blocked layout (C... T... A...) if ever needed
            n_cells = rows * cols
            x0_flat[0:n_cells] = x0_C
            x0_flat[n_cells:2*n_cells] = x0_T
            x0_flat[2*n_cells:3*n_cells] = x0_A
        self.x0 = x0_flat.reshape((-1, 1))

        # Default parameters (match your script) unless provided
        self.p = params or Params(
            lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
            lambda_T=3.0, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
            d_A=0.0315, rows=rows, cols=cols
        )

        # Hold results
        self.J_autograd = None
        self.dx_values = None
        self.errors = None
        self.dx_machine = None
        self.dx_nitsol = None
        self.dx_optimal = None

    @staticmethod
    def default_dx_values():
        """Geometric grid from 1e-16 to 1 with step 10**0.25."""
        return 10.0 ** np.arange(-16, 1, 0.25)

    def compute_autograd(self):
        """Compute and cache the autograd Jacobian."""
        self.J_autograd = eval_Jf_autograd(eval_f, self.x0, self.p, self.u)
        return self.J_autograd

    def sweep_fd(self, dx_values: np.ndarray):
        """
        For each dx, compute FD Jacobian and Frobenius error vs autograd.
        Returns the error array (same length as dx_values).
        """
        if self.J_autograd is None:
            self.compute_autograd()

        errs = []
        for dx in dx_values:
            # Set FD step on params (your FD implementation reads p.dxFD)
            self.p.dxFD = float(dx)
            J_fd, _ = eval_Jf_FiniteDifference(eval_f, self.x0.copy(), self.p, self.u)
            err = np.linalg.norm(J_fd - self.J_autograd, ord='fro')
            errs.append(err)
        return np.asarray(errs)

    def run(self, dx_values: np.ndarray | None = None):
        """Compute autograd J, run FD sweep, record reference dx’s and optimal dx."""
        self.dx_values = dx_values if dx_values is not None else self.default_dx_values()
        self.compute_autograd()
        self.errors = self.sweep_fd(self.dx_values)

        # References
        eps = np.finfo(float).eps
        self.dx_machine = np.sqrt(eps)
        norm_inf = np.linalg.norm(self.x0.reshape(-1), np.inf)
        self.dx_nitsol = 2.0 * np.sqrt(eps) * max(1.0, norm_inf)

        # Optimal dx (min error)
        idx_opt = int(np.argmin(self.errors))
        self.dx_optimal = float(self.dx_values[idx_opt])
        return {
            "dx_values": self.dx_values,
            "errors": self.errors,
            "dx_machine": self.dx_machine,
            "dx_nitsol": self.dx_nitsol,
            "dx_optimal": self.dx_optimal,
        }

    def plot(self, save_dir: str = "test_evalf_output_figures", filename: str = "Jacobian_FD_vs_Autograd.png", show: bool = False):
        """Make the log–log error plot and save it."""
        assert self.dx_values is not None and self.errors is not None, "Call run() before plot()."

        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(7, 5))
        plt.loglog(self.dx_values, self.errors, linewidth=2, label="‖J_FD(dx) − J_AD‖_F")

        # Reference lines
        if self.dx_machine is not None:
            plt.axvline(self.dx_machine, linestyle="--", alpha=0.7, label=r"$\sqrt{\varepsilon}$")
        if self.dx_nitsol is not None:
            plt.axvline(self.dx_nitsol, linestyle="-.", alpha=0.7, color="green",
                        label=r"$2\sqrt{\varepsilon}\max(1,\|x\|_{\infty})$")
        if self.dx_optimal is not None:
            plt.axvline(self.dx_optimal, linestyle=":", alpha=0.9, color="red",
                        label=f"Optimal dx = {self.dx_optimal:.2e}")

        plt.xlabel(r"$dx$")
        plt.ylabel(r"$\|J_{\mathrm{FD}}(dx)-J_{\mathrm{AD}}\|_F$")
        plt.title("Jacobian comparison: Finite Difference vs Autograd")
        plt.grid(True, which="both", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
        return out_path


# Optional: CLI entry point
if __name__ == "__main__":
    jc = JacobianComparator()
    info = jc.run()  # default dx grid
    path = jc.plot(show=False)
    print(f"Saved plot to: {path}")
    print(f"dx* (optimal): {info['dx_optimal']:.3e} | dx_machine: {info['dx_machine']:.3e} | dx_nitsol: {info['dx_nitsol']:.3e}")