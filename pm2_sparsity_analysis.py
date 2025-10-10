import numpy as np
import copy
import os
import eval_f as f
from eval_Jf_autograd import eval_Jf_autograd
from test_eval_f import TestEvalF
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

class sparsity_analysis:
    
    def compute_jacobian(self, square_width=2):
        print(f"Computing Jacobian for {square_width}x{square_width} grid ({square_width**2 * 3} state variables)...")
        self.eval_f_framework = TestEvalF()
        self.p_standard = copy.deepcopy(self.eval_f_framework.p_default)
        
        rows, cols = square_width, square_width
        self.p_standard.rows = rows
        self.p_standard.cols = cols
        
        # Create initial state as column vector (n_cells * 3, 1)
        n_cells = rows * cols
        self.x0_basic = np.zeros((n_cells * 3, 1))
        
        # Initialize with reasonable values
        for i in range(n_cells):
            self.x0_basic[i*3 + 0, 0] = 15  # Cancer cells (scaled down from original)
            self.x0_basic[i*3 + 1, 0] = 2  # T cells (scaled down)
            self.x0_basic[i*3 + 2, 0] = 0  # Drug concentration

        self.u_standard = 200

        J = eval_Jf_autograd(f.eval_f, self.x0_basic, self.p_standard, self.u_standard)

        sparsity = self.analyze_sparsity(J)
        print(f"Sparsity (fraction of non-zero elements): {sparsity:.2%}")

        self.plot_jacobian_heatmap(J, sparsity, outdir="pm2_output_figures")
        
    def analyze_sparsity(self, J, tol=1e-12):
        total_elements = J.size
        nonzero_elements = np.sum(np.abs(J) > tol)
        sparsity = (nonzero_elements / total_elements)
        return sparsity

    def plot_jacobian_heatmap(self, J, sparsity, outdir, tol=1e-12):
        sign_mat = np.zeros_like(J, dtype=int)
        sign_mat[J >  tol] =  1
        sign_mat[J < -tol] = -1

        # 3-color categorical map: negative / zero / positive
        cmap_sign = ListedColormap(["#2b6cb0", "#e2e8f0", "#e53e3e"])   # blue / light gray / red
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm_sign = BoundaryNorm(bounds, cmap_sign.N)

        # --- Log magnitude (add epsilon to avoid log(0)) ---
        eps = 1e-12
        logmag = np.log10(np.abs(J) + eps)
        cmap_mag = "hot"  # or "viridis", etc.

        os.makedirs(outdir, exist_ok=True)

        legend_patches = [
            Patch(facecolor="#2b6cb0", edgecolor="none", label="Negative (< -tol)"),
            Patch(facecolor="#e2e8f0", edgecolor="none", label=f"Zero (|J| ≤ {tol:g})"),
            Patch(facecolor="#e53e3e", edgecolor="none", label="Positive (> tol)"),
        ]

        # Combined two-panel figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        imA = axes[0].imshow(sign_mat, cmap=cmap_sign, norm=norm_sign, interpolation="nearest")
        axes[0].set_title("Sign")
        axes[0].set_xlabel("Column (cause)")
        axes[0].set_ylabel("Row (effect)")
        axes[0].legend(handles=legend_patches, loc="upper right", frameon=False)

        imB = axes[1].imshow(logmag, cmap=cmap_mag, interpolation="nearest")
        axes[1].set_title("log10(|J| + ε)")
        axes[1].set_xlabel("Column (cause)")
        axes[1].set_ylabel("Row (effect)")
        fig.colorbar(imB, ax=axes[1])

        fig.suptitle(f"Jacobian Heatmaps ({J.shape[0]}×{J.shape[1]}); Percentage of Non-Zero Elements: {sparsity:.2%}", y=1.02, fontsize=12)
        fig.savefig(os.path.join(outdir, f"jacobian_heatmap_{J.shape[0]}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def run(self):
        self.compute_jacobian(square_width=2) 
        self.compute_jacobian(square_width=5) 
        self.compute_jacobian(square_width=10)
        self.compute_jacobian(square_width=20)

if __name__ == "__main__":
    analysis = sparsity_analysis()
    analysis.run()

