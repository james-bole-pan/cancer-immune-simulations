import numpy as np
from test_eval_f import TestEvalF
from VisualizeNetwork import visualizeNetwork
from pathlib import Path
from eval_f import Params

# --- directories ---
data_dir = Path("/Users/jamesbolepan/Documents/cancer-immune-simulations/data/nature_immune_processed")
plot_dir = Path("/Users/jamesbolepan/Documents/cancer-immune-simulations/data/nature_immune_processed_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

test_eval_f = TestEvalF()
p_default = test_eval_f.p_default

# --- iterate over all npy files ---
npy_files = sorted(data_dir.glob("*.npy"))
if not npy_files:
    raise FileNotFoundError(f"No .npy files found in {data_dir}")

print(f"[INFO] Found {len(npy_files)} npy files in {data_dir}")


for npy_file in npy_files:
    slide_name = npy_file.stem
    print(f"[PROCESSING] {slide_name}")

    # Load the 3D array
    x_arr = np.load(npy_file)
    rows, cols, channels = x_arr.shape

    # Update Params for this grid
    p_default.rows = rows
    p_default.cols = cols

    # Flatten (rows * cols * 3, 1)
    x_col = x_arr.reshape(rows * cols * channels, 1)

    # Generate visualization and save
    fig, axes, save_path = visualizeNetwork(
        x=x_col,
        p=p_default,
        title_prefix=f"{slide_name}_",
        output_dir=str(plot_dir),
        dpi=200,
    )

    print(f"[SAVED FIGURE] {slide_name} → {save_path}")

print(f"[DONE] Plots saved in: {plot_dir}")