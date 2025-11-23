import os
import numpy as np
from eval_f import eval_f, Params
from eval_f_output  import eval_f_output
from eval_u_keytruda_input import eval_u_keytruda_input
from VisualizeNetwork import visualizeNetwork, create_network_evolution_gif
from SimpleSolver import SimpleSolver
import pandas as pd

import numpy as np

def sample_trajectory_for_gif(X, max_frames=20):
    """
    Return a sampled trajectory matrix X_sampled (N x T_sampled),
    where T_sampled <= max_frames and sampling is uniform.
    """
    X = np.asarray(X)
    N, T = X.shape

    # Select at most max_frames uniformly spaced indices
    if T <= max_frames:
        frame_indices = np.arange(T)
    else:
        frame_indices = np.linspace(0, T - 1, max_frames, dtype=int)

    # Stack selected columns into a matrix (N × T_sampled)
    X_sampled = X[:, frame_indices]

    return X_sampled

def pct_change(final, initial):
                if initial == 0:
                    return float('nan')  # or return None if you prefer
                return ((final - initial) / initial) * 100

if __name__ == "__main__":
    p_default = Params(
                        lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
                        lambda_T=3.0, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
                        d_A=0.0315, rows=1, cols=1
                    )
    clinical_data_path = "data/nature_immune_processed"
    slide_response_path = "data/nature_immune_processed/slide_responses.csv"
    slide_response_df = pd.read_csv(slide_response_path)
    output_figures_path = "test_clinical_data_visualization"

    u_fun = eval_u_keytruda_input()

    for file in os.listdir(clinical_data_path):
        if file.endswith(".npy"):
            print("=" * 80)
            print(f"📁 Processing sample: {file}")
            print("=" * 80)

            file_path = os.path.join(clinical_data_path, file)
            assert os.path.exists(file_path), f"❌ ERROR: File {file_path} does not exist."

            sample_id = file.replace(".npy", "")
            response_value = slide_response_df.loc[
                slide_response_df["Slide.id.Ab"] == sample_id, "Response"
            ].values[0]

            print(f"🧬 Sample ID: {sample_id}")
            print(f"🩺 Clinical Response (Ground Truth): {response_value}")
            print("-" * 80)

            x_arr = np.load(file_path)
            rows, cols, channels = x_arr.shape
            print(f"📏 Loaded spatial tensor: {rows} × {cols} × {channels}")

            # If transpose needed
            if rows < cols:
                print("🔄 Rotating image to enforce rows ≥ cols...")
                print(f"   Original shape: {x_arr.shape}")
                x_arr = x_arr.transpose(1, 0, 2)
                rows, cols, channels = x_arr.shape
                print(f"   New shape:      {x_arr.shape}")

            # Update params
            p_default.rows = rows
            p_default.cols = cols

            print(f"📦 Flattening tensor → x_col with shape ({rows*cols*channels}, 1)")
            x_col = x_arr.reshape(rows * cols * channels, 1)

            # Initial burden
            total_cancer_burden, total_tcell_count = eval_f_output(x_col)
            print(f"📊 Initial State:")
            print(f"   - Total cancer burden: {total_cancer_burden:.2f}")
            print(f"   - Total T-cell count:  {total_tcell_count:.2f}")
            print("-" * 80)

            # Visualization before simulation
            print("🖼️ Generating initial network visualization...")
            fig, axes, save_path = visualizeNetwork(
                x=x_col, p=p_default, save=False, visualize=False
            )

            # Run solver
            print("🚀 Running SimpleSolver simulation...")
            NumIter = 100
            w = 0.01
            X, t = SimpleSolver(
                eval_f, x_col, p_default, u_fun, 
                NumIter, w=w, visualize=False,
                gif_file_name=f"{output_figures_path}/{sample_id}_simplesolver_visualization.gif"
            )

            # Check for numeric issues
            if not np.isfinite(X).all():
                print("❌ WARNING: Simulation produced NaN or Inf values!")
            else:
                print("✅ Numerical check passed: all values finite.")

            # Final burden
            total_cancer_burden_final, total_tcell_count_final = eval_f_output(
                X[:, -1].reshape(-1, 1)
            )

            print(f"📊 Final State:")
            print(f"   - Total cancer burden: {total_cancer_burden_final:.2f}")
            print(f"   - Total T-cell count:  {total_tcell_count_final:.2f}")            

            cancer_pct = pct_change(total_cancer_burden_final, total_cancer_burden)
            tcell_pct  = pct_change(total_tcell_count_final, total_tcell_count)

            print(f"📈 Percentage change:")
            print(f"   - Tumor burden change: {cancer_pct:.2f}%")
            print(f"   - T-cell count change: {tcell_pct:.2f}%")

            # Compare model prediction
            responder_status = "R" if total_cancer_burden_final < total_cancer_burden else "NR"
            if response_value == responder_status:
                print(f"🎉 Model prediction MATCHES clinical response → {responder_status}")
            else:
                print(f"⚠️ Model prediction mismatch:")
                print(f"   - Predicted: {responder_status}")
                print(f"   - Actual:    {response_value}")

            # save X for future analysis (with NumIter and w in the filename)
            npy_save_path = os.path.join(
                output_figures_path,
                f"{sample_id}_X_NumIter{NumIter}_w{w}.npy"
            )
            np.save(npy_save_path, X)
            print(f"💾 Saved simulation trajectory X to {npy_save_path}")

            # GIF Visualization
            print("🎞️ Generating network evolution GIF (max 20 frames)...")

            # Sample frames BEFORE creating the GIF
            X_sampled = sample_trajectory_for_gif(X, max_frames=20)
            
            # Now pass the sampled frames to the GIF creator
            create_network_evolution_gif(
                X_sampled,
                p_default,
                save=True,
                show=False,
                output_dir=output_figures_path,
                title_prefix=f"{sample_id}_evolution_",
                fps=5
            )

            print("=" * 80)
            print("✔️ Finished processing sample.")
            print("=" * 80)

            break
    print("All clinical data files have been processed.")