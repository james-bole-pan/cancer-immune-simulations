from eval_f import eval_f, Params
import numpy as np
from SimpleSolver import SimpleSolver
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

class TestEvalF_RealWorldData:
    def __init__(self):
        self.p_default = Params(
            lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
            lambda_T=3.0, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
            d_A=0.0315, rows=1, cols=1
        )
        self.figure_dir = "test_evalf_output_figures/"
        os.makedirs(self.figure_dir, exist_ok=True)

        self.DOSE = 200.0  # drug dose
        self.INTERVAL = 21.0  # drug administration interval in days
        self.CHANNELS = 3  # number of channels in the spatial data (C, T, A)

    def actual_drug_input(self, dose=200.0, interval=21.0):
        def r(t):
            return dose if (t % interval == 0.0) else 0.0
        return r

    def test_real_world_data(self, w, num_iter, data_path):
        spatial_data = np.load(data_path)                     # (rows, cols, channels)
        rows, cols, channels = spatial_data.shape
        assert channels == self.CHANNELS, f"Expected {self.CHANNELS} channels, got {channels}"

        p = copy.deepcopy(self.p_default)
        p.rows = rows
        p.cols = cols

        x0 = spatial_data.reshape(rows * cols * channels, 1)  # (n_state, 1)
        print(f"Initial state reshaped to {x0.shape}")

        bolus = self.actual_drug_input(dose=self.DOSE, interval=self.INTERVAL)

        X, t = SimpleSolver(
            eval_f,
            x_start=x0,
            p=p,
            eval_u=bolus,
            NumIter=num_iter,
            w=w,
            visualize=True,
            gif_file_name=f"{self.figure_dir}/real_world_data_w_{w}.gif",
        )
        X = np.asarray(X)  # (n_state, num_frames)
        t = np.asarray(t)  # (num_frames,)

        # save X to npy for later analysis
        #np.save(f"{self.figure_dir}/realworld_data_w{w}_X.npy", X)

        self.create_visualization(X, t, rows, cols)

    def create_visualization(self, X, t, rows, cols):
        cancer_cmap = LinearSegmentedColormap.from_list('cancer', ['white', 'red', 'darkred'])
        immune_cmap = LinearSegmentedColormap.from_list('immune', ['white', 'blue', 'darkblue'])
        drug_cmap   = LinearSegmentedColormap.from_list('drug',   ['white', 'green', 'darkgreen'])

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        axes = axes.flatten()

        def frame_to_grid(frame_vec):
            return frame_vec.reshape(rows, cols, self.CHANNELS)

        num_frames = X.shape[1]
        all_frames_reshaped = np.empty((num_frames, rows, cols, self.CHANNELS), dtype=X.dtype)
        for i in range(num_frames):
            all_frames_reshaped[i] = frame_to_grid(X[:, i])

        cmin, cmax = np.min(all_frames_reshaped[..., 0]), np.max(all_frames_reshaped[..., 0])
        imin, imax = np.min(all_frames_reshaped[..., 1]), np.max(all_frames_reshaped[..., 1])
        dmin, dmax = np.min(all_frames_reshaped[..., 2]), np.max(all_frames_reshaped[..., 2])

        # --- pre-create images (frame 0) ---
        f0 = all_frames_reshaped[0]
        cancer_img = axes[0].imshow(f0[:, :, 0], cmap=cancer_cmap, vmin=cmin, vmax=cmax,
                                    interpolation='nearest', origin='upper')
        immune_img = axes[1].imshow(f0[:, :, 1], cmap=immune_cmap, vmin=imin, vmax=imax,
                                    interpolation='nearest', origin='upper')
        drug_img   = axes[2].imshow(f0[:, :, 2], cmap=drug_cmap,   vmin=dmin, vmax=dmax,
                                    interpolation='nearest', origin='upper')

        # combined RGB image; keep a handle and update with set_data
        combined0 = np.zeros((rows, cols, 3), dtype=float)
        if cmax > 0: combined0[:, :, 0] = np.clip(f0[:, :, 0] / cmax, 0, 1)
        if imax > 0: combined0[:, :, 2] = np.clip(f0[:, :, 1] / imax, 0, 1)
        combined_img = axes[3].imshow(combined0, interpolation='nearest', origin='upper')

        # fixed titles (don’t include t here to avoid width changes)
        axes[0].set_title('Cancer Cells')
        axes[1].set_title('Immune Cells')
        axes[2].set_title('Drug Concentration')
        axes[3].set_title('Combined (R=cancer, B=immune)')

        # freeze geometry so frames don’t move
        for ax in axes:
            ax.set_xlim(-0.5, cols - 0.5)
            ax.set_ylim(rows - 0.5, -0.5)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([]); ax.set_yticks([])

        # persistent texts to update (instead of recreating)
        txt_c = axes[0].text(0.02, 0.98, '', transform=axes[0].transAxes, va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        txt_i = axes[1].text(0.02, 0.98, '', transform=axes[1].transAxes, va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        txt_d = axes[2].text(0.02, 0.98, '', transform=axes[2].transAxes, va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # one global time label (so subplot titles don’t change width)
        time_text = fig.suptitle(f"t = {t[0]:.2f}")

        def animate(k):
            frame = all_frames_reshaped[k]

            # update image arrays
            cancer_img.set_data(frame[:, :, 0])
            immune_img.set_data(frame[:, :, 1])
            drug_img.set_data(frame[:, :, 2])

            comb = np.zeros_like(combined0)
            if cmax > 0: comb[:, :, 0] = np.clip(frame[:, :, 0] / cmax, 0, 1)
            if imax > 0: comb[:, :, 2] = np.clip(frame[:, :, 1] / imax, 0, 1)
            combined_img.set_data(comb)

            # update texts
            txt_c.set_text(f"Total: {frame[:, :, 0].sum():.0f}")
            txt_i.set_text(f"Total: {frame[:, :, 1].sum():.0f}")
            txt_d.set_text(f"Mean: {frame[:, :, 2].mean():.2f}")
            time_text.set_text(f"t = {t[k]:.2f} days")

            # return artists (use blit=True if you like)
            return [cancer_img, immune_img, drug_img, combined_img, txt_c, txt_i, txt_d, time_text]

        print(f"Creating animation with {num_frames} frames...")
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=500, blit=False, repeat=True)
        anim.save(f"{self.figure_dir}/realworld_animation.gif", writer='pillow', fps=10)

        # save first/last
        animate(0);  fig.savefig(f"{self.figure_dir}/realworld_animation_initial.png", dpi=150, bbox_inches='tight')
        animate(num_frames - 1); fig.savefig(f"{self.figure_dir}/realworld_animation_final.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    tester = TestEvalF_RealWorldData()
    tester.test_real_world_data(
        w=0.1,
        num_iter=100,
        data_path="data/fake_spatial_data_tumor_int.npy",
    )
