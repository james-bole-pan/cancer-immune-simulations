import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from eval_f import eval_f, Params
from eval_f_output import eval_f_output
from eval_u_keytruda_input import eval_u_keytruda_input
from VisualizeNetwork import visualizeNetwork, create_network_evolution_gif
from SimpleSolver import SimpleSolver


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def sample_trajectory_for_gif(X, max_frames):
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

    # p_default = Params(
    #     lambda_C=0.33, K_C=30, d_C=0.01, k_T=4, K_K=5, D_C=0.2,
    #     lambda_T=0.5, K_T=10, K_R=20, d_T=0.01, k_A=0.16, K_A=100, D_T=0.2,
    #     d_A=0.0315, rows=1, cols=1
    # )

    p_default = Params(
        lambda_C=0.33, K_C=28, d_C=0.01, k_T=4, K_K=5, D_C=0.01,
        lambda_T=1, K_T=10, K_R=20, d_T=0.01, k_A=0.16, K_A=100, D_T=0.2,
        d_A=0.0315, rows=1, cols=1
    )

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

    u_fun = eval_u_keytruda_input(w=w)

    X, t = SimpleSolver(
        eval_f, x_col, p_default, u_fun,
        NumIter, w=w, visualize=False,
        gif_file_name=f"{output_figures_path}/{sample_id}_simplesolver.gif"
    )

    # create a plot of the concentration of drug A over time at one grid
    drug_A_concentration = X[2, :]
    plt.plot(np.arange(len(drug_A_concentration)), drug_A_concentration)
    plt.xlabel("Time Step")
    plt.ylabel("Drug A Concentration")
    plt.title(f"Drug A Concentration Over Time for {sample_id}")
    plt.savefig(f"{output_figures_path}/{sample_id}_drug_A_concentration.png")
    plt.close()

    if not np.isfinite(X).all():
        print(f"❌ WARNING: Simulation for {sample_id} produced NaN or Inf values!")
    else:
        print(f"✅ Numerical check passed for {sample_id}: all values finite.")

    # print the total number of frames generated
    print(f"Sample {sample_id}: Generated {X.shape[1]} frames.")

    total_Cf, total_Tf = eval_f_output(X[:, -1].reshape(-1, 1))

    predicted = "R" if total_Cf < total_C0 else "NR"

    npy_save_path = os.path.join(
        output_figures_path,
        f"{sample_id}_X_NumIter{NumIter}_w{w}.npy"
    )
    #np.save(npy_save_path, X)

    X_sampled = sample_trajectory_for_gif(X, max_frames=100)

    create_network_evolution_gif(
        X_sampled,
        p_default,
        save=True,
        show=False,
        output_dir=output_figures_path,
        title_prefix=f"{sample_id}_evolution",
        fps=20
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
    CSV_PATH = "data_preprocessing_notebooks/npy_dimensions_sorted.csv" 
    TOP_N = 1                   

    df = pd.read_csv(CSV_PATH)
    df_sorted = df.sort_values(by="first_two_product", ascending=True)
    df_topN = df_sorted.head(TOP_N)
    
    files_to_run = df_topN["full_path"].tolist()

    print(f"Running on top {TOP_N} smallest .npy files:")
    for fp in files_to_run:
        print(" -", fp)

    with Pool(cpu_count()) as pool:
        file_names_only = [os.path.basename(fp) for fp in files_to_run]
        results = pool.map(process_single_sample, file_names_only)

    df_results = pd.DataFrame(results)
    
    df_results["correct"] = df_results["ground_truth"] == df_results["predicted"]
    accuracy = df_results["correct"].mean()
    
    print(f"Prediction accuracy: {accuracy * 100:.2f}%")

    os.makedirs("test_clinical_data_visualization", exist_ok=True)
    df_results.to_csv("test_clinical_data_visualization/summary_results.csv", index=False)
