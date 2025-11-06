# test_visualize_network_class.py
import unittest
import tempfile
from pathlib import Path
import numpy as np

from eval_f import Params
from VisualizeNetwork import visualizeNetwork

class TestVisualizeNetwork(unittest.TestCase):
    def setUp(self):
        # defaults, but rows/cols set to 3x3 to match the constructed state
        self.p_default = Params(
            lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
            lambda_T=3.0, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
            d_A=0.0315, rows=3, cols=3
        )

    @staticmethod
    def _make_3x3_state() -> np.ndarray:
        """Create (3,3,3) field [C,T,A] and flatten to (N,1) in row-major order."""
        rows, cols = 3, 3
        yy, xx = np.mgrid[0:rows, 0:cols]

        # Cancer (C): Gaussian bump at center
        C = np.exp(-(((xx - 1.0) ** 2 + (yy - 1.0) ** 2) / 1.5))
        # T cells (T): left->right gradient
        T = xx.astype(float) / (cols - 1)
        # Drug (A): top->bottom gradient
        A = yy.astype(float) / (rows - 1)

        x_rc3 = np.stack([C, T, A], axis=-1)     # (3,3,3)
        x_col = x_rc3.reshape(-1, 1, order="C")  # (N,1) with N=27
        return x_col

    def test_saves_single_combined_image(self):
        x_col = self._make_3x3_state()
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "figs"
            fig, axes, save_path = visualizeNetwork(
                x_col,
                self.p_default,
                title_prefix="demo_3x3_",
                output_dir=str(outdir),
                dpi=200,
            )
            # Assertions
            self.assertTrue(save_path.exists(), "Combined image was not saved.")
            self.assertTrue(save_path.is_file(), "Save path is not a file.")
            self.assertGreater(save_path.stat().st_size, 0, "Saved file is empty.")
            self.assertEqual(len(axes), 3, "Expected 3 subplots (C,T,A).")
            # Cleanup figure to avoid memory leaks in repeated runs
            fig.clf()

    def test_bad_shape_raises(self):
        rows, cols = self.p_default.rows, self.p_default.cols
        N = rows * cols * 3
        with tempfile.TemporaryDirectory() as td:
            # Wrong shape: (N,) instead of (N,1)
            x_bad1 = np.zeros((N,), dtype=float)
            with self.assertRaises(ValueError):
                visualizeNetwork(x_bad1, self.p_default, output_dir=td)

            # Wrong length: (N+1,1)
            x_bad2 = np.zeros((N + 1, 1), dtype=float)
            with self.assertRaises(ValueError):
                visualizeNetwork(x_bad2, self.p_default, output_dir=td)

    def test_bad_params_type_raises(self):
        x_col = self._make_3x3_state()

        class NotParams:
            rows = 3
            cols = 3

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(TypeError):
                visualizeNetwork(x_col, NotParams(), output_dir=td)

    def test_persistent_output_dir(self):
        """Save a single combined figure into 'test_evalf_output_figures/' and verify it exists."""
        x_col = self._make_3x3_state()

        # persistent output folder
        outdir = Path("test_evalf_output_figures")
        outdir.mkdir(parents=True, exist_ok=True)

        # unique prefix to avoid collisions on repeated runs/CI
        run_tag = "persistent_test"
        title_prefix = f"unittest_{run_tag}_"

        fig, axes, save_path = visualizeNetwork(
            x_col,
            self.p_default,
            title_prefix=title_prefix,
            output_dir=str(outdir),
            dpi=200,
        )

        # Assertions: path is in the desired folder, exists, non-empty
        self.assertTrue(save_path.exists(), "Combined image was not saved.")
        self.assertTrue(str(save_path).startswith(str(outdir)), "Image not saved in requested output_dir.")
        self.assertGreater(save_path.stat().st_size, 0, "Saved file is empty.")
        self.assertEqual(len(axes), 3, "Expected 3 subplots (C,T,A).")

        # Optional: leave file for manual inspection; just close fig
        fig.clf()
        print(f"[TEST INFO] Persistent figure saved at: {save_path}")


if __name__ == "__main__":
    unittest.main()