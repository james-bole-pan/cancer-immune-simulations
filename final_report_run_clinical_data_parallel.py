import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from eval_f import eval_f, Params
from eval_f_output import eval_f_output
from eval_u_keytruda_input import eval_u_keytruda_input
from VisualizeNetwork import visualizeNetwork, create_network_evolution_gif
from SimpleSolver import SimpleSolver


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def sample_trajectory_for_gif(X, max_frames=20):
    X = np.asarray(X)
    N, T = X.shape

    if T <= max_frames:
        frame_indices = np.arange(T)
    else:
        frame_indices = np.linspace(0, T - 1, max_frames, dtype=int)

    return X[:, frame_indices]


def pct_change(final, initial):
    if initial == 0:
        return float('nan')
    return ((final - initial) / initial) * 100


# ------------------------------------------------------------
# Worker: processes ONE .npy file on ONE CPU core
# ------------------------------------------------------------

def process_single_sample(file):
    clinical_data_path = "data/nature_immune_processed"
    output_figures_path = "test_clinical_data_visualization"
    slide_response_path = "data/nature_immune_processed/slide_responses.csv"

    slide_response_df = pd.read_csv(slide_response_path)

    p_default = Params(
        lambda_C=0.33, K_C=30, d_C=0.01, k_T=4, K_K=5, D_C=0.2,
        lambda_T=0.5, K_T=10, K_R=20, d_T=0.01, k_A=0.16, K_A=100, D_T=0.2,
        d_A=0.0315, rows=1, cols=1
    )

    u_fun = eval_u_keytruda_input()

    file_path = os.path.join(clinical_data_path, file)
    sample_id = file.replace(".npy", "")
    response_value = slide_response_df.loc[
        slide_response_df["Slide.id.Ab"] == sample_id, "Response"
    ].values[0]

    x_arr = np.load(file_path)
    rows, cols, channels = x_arr.shape

    if rows < cols:
        x_arr = x_arr.transpose(1, 0, 2)
        rows, cols, channels = x_arr.shape

    p_default.rows = rows
    p_default.cols = cols

    x_col = x_arr.reshape(rows * cols * channels, 1)

    total_C0, total_T0 = eval_f_output(x_col)

    #visualizeNetwork(x=x_col, p=p_default, save=False, visualize=False)

    NumIter = 8400
    w = 0.01

    X, t = SimpleSolver(
        eval_f, x_col, p_default, u_fun,
        NumIter, w=w, visualize=False,
        gif_file_name=f"{output_figures_path}/{sample_id}_simplesolver.gif"
    )

    # print the total number of frames generated
    print(f"Sample {sample_id}: Generated {X.shape[1]} frames.")

    total_Cf, total_Tf = eval_f_output(X[:, -1].reshape(-1, 1))

    predicted = "R" if total_Cf < total_C0 else "NR"

    npy_save_path = os.path.join(
        output_figures_path,
        f"{sample_id}_X_NumIter{NumIter}_w{w}.npy"
    )
    #np.save(npy_save_path, X)

    X_sampled = sample_trajectory_for_gif(X, max_frames=20)

    create_network_evolution_gif(
        X_sampled,
        p_default,
        save=True,
        show=False,
        output_dir=output_figures_path,
        title_prefix=f"{sample_id}_evolution",
        fps=5
    )

    return {
        "sample_id": sample_id,
        "ground_truth": response_value,
        "predicted": predicted,
        "initial_cancer": float(total_C0),
        "final_cancer": float(total_Cf),
        "initial_tcells": float(total_T0),
        "final_tcells": float(total_Tf),
        "pct_change_cancer": float(pct_change(total_Cf, total_C0)),
        "pct_change_tcells": float(pct_change(total_Tf, total_T0)),
        "trajectory_path": npy_save_path,
    }


# ------------------------------------------------------------
# Parallel Driver
# ------------------------------------------------------------

if __name__ == "__main__":
    clinical_data_path = "data/nature_immune_processed"
    files = [f for f in os.listdir(clinical_data_path) if f.endswith(".npy")]
    # pick the first 10 files for testing
    files = files[:10]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_sample, files)

    # Save summary results
    df = pd.DataFrame(results)
    # calculate accuracy
    df["correct"] = df["ground_truth"] == df["predicted"]
    accuracy = df["correct"].mean()
    print(f"Prediction accuracy: {accuracy * 100:.2f}%")
    df.to_csv("test_clinical_data_visualization/summary_results.csv", index=False)
