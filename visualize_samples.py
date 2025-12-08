"""
Standalone visualization script for specific samples.

Usage:
    python visualize_samples.py <sample1> <sample2> ... <sampleN>

Example:
    python visualize_samples.py 2e82_08 2e77_02 2e83_01

This script:
1. Loads learned parameters from final_report_drug_modeling_counterfactual_minpath/learned_parameters.csv
2. For each sample specified on command line:
   - Loads the sample data from data/nature_immune_processed/{sample}.npy
   - Runs simulation WITH drug and NO drug using SimpleSolver_torch
   - Generates GIFs using create_network_evolution_gif from VisualizeNetwork.py
3. Saves GIFs to ./sample_visualizations/{with_drug, no_drug}/
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from eval_u_keytruda_input import eval_u_keytruda_input
from VisualizeNetwork import create_network_evolution_gif

FPS = 40

# ============================================================
# Import from final_report_run_sgd_optimization_pytorch_delta_loss
# ============================================================
from final_report_run_sgd_optimization_pytorch_delta_loss import (
    Params,
    OPT_NAMES,
    SIGNED_PARAMS,
    softplus_torch,
    inv_softplus_torch,
    inv_tanh,
    theta_raw_to_params,
    laplacian4_vec,
    eval_f_torch,
    SimpleSolver_torch,
    total_cancer_from_xcol_torch,
    pct_change_torch,
    make_eval_u_zero,
    load_labels,
    response_to_y,
    load_sample_xcol,
    CLAMP_MIN_STATE,
    CLAMP_MAX_STATE,
    CLAMP_PCT_MIN,
    CLAMP_PCT_MAX,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def plot_evolution_comparison(
    X_drug_ts,
    X_nodrug_ts,
    rows,
    cols,
    w,
    sample_id,
    output_dir="sample_visualizations",
):
    """
    Plot evolution of total cancer, total tumor, and drug concentration over time.
    Three panels: one for each variable.
    Solid line for WITH drug, dotted line for NO drug.
    
    Parameters
    ----------
    X_drug_ts : torch.Tensor
        Simulation trajectory WITH drug, shape (num_timesteps, state_size)
    X_nodrug_ts : torch.Tensor
        Simulation trajectory NO drug, shape (num_timesteps, state_size)
    rows : int
        Number of rows in grid
    cols : int
        Number of columns in grid
    w : float
        Time step
    sample_id : str
        Sample ID for title
    output_dir : str
        Directory to save the plot
    """
    # Convert to numpy for easier indexing
    X_drug_np = X_drug_ts.cpu().detach().numpy()  # (num_timesteps, state_size)
    X_nodrug_np = X_nodrug_ts.cpu().detach().numpy()  # (num_timesteps, state_size)
    
    num_timesteps_drug = X_drug_np.shape[0]
    num_timesteps_nodrug = X_nodrug_np.shape[0]
    state_size = X_drug_np.shape[1]
    
    # Compute time arrays
    time_drug = np.arange(num_timesteps_drug) * w
    time_nodrug = np.arange(num_timesteps_nodrug) * w
    
    # Extract channel data (assume C, T, A are interleaved: index 0, 1, 2, ...)
    # For each grid point, we have [C_00, T_00, A_00, C_01, T_01, A_01, ...]
    # Total cancer = sum of all C values; total tumor = sum of all T values
    
    # Total cancer (channel 0): indices 0, 3, 6, ...
    cancer_indices = np.arange(0, state_size, 3)
    # Total tumor (channel 1): indices 1, 4, 7, ...
    tumor_indices = np.arange(1, state_size, 3)
    # Drug (channel 2): indices 2, 5, 8, ..., use first grid point (index 2)
    drug_index = 2
    
    total_cancer_drug = X_drug_np[:, cancer_indices].sum(axis=1)
    total_tumor_drug = X_drug_np[:, tumor_indices].sum(axis=1)
    drug_conc_drug = X_drug_np[:, drug_index]
    
    total_cancer_nodrug = X_nodrug_np[:, cancer_indices].sum(axis=1)
    total_tumor_nodrug = X_nodrug_np[:, tumor_indices].sum(axis=1)
    drug_conc_nodrug = X_nodrug_np[:, drug_index]
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel 1: Total cancer cells
    axes[0].plot(time_drug, total_cancer_drug, 'b-', linewidth=2, label='WITH drug')
    axes[0].plot(time_nodrug, total_cancer_nodrug, 'b--', linewidth=2, label='NO drug')
    axes[0].set_xlabel('Time', fontsize=11)
    axes[0].set_ylabel('Total Cancer Cells', fontsize=11)
    axes[0].set_title('Cancer Evolution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Panel 2: Total tumor cells (immune)
    axes[1].plot(time_drug, total_tumor_drug, 'r-', linewidth=2, label='WITH drug')
    axes[1].plot(time_nodrug, total_tumor_nodrug, 'r--', linewidth=2, label='NO drug')
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylabel('Total Tumor Cells (T cells)', fontsize=11)
    axes[1].set_title('Immune Response Evolution', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    # Panel 3: Drug concentration (at one grid point)
    axes[2].plot(time_drug, drug_conc_drug, 'g-', linewidth=2, label='WITH drug')
    axes[2].plot(time_nodrug, drug_conc_nodrug, 'g--', linewidth=2, label='NO drug')
    axes[2].set_xlabel('Time', fontsize=11)
    axes[2].set_ylabel('Drug Concentration (A)', fontsize=11)
    axes[2].set_title('Drug Concentration at Grid Point (0,0)', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3)
    
    fig.suptitle(f'Sample {sample_id}: Evolution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{sample_id}_evolution_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def load_learned_parameters(params_csv_path):
    """Load learned parameters from CSV and return as dict."""
    if not os.path.exists(params_csv_path):
        raise FileNotFoundError(f"Learned parameters CSV not found: {params_csv_path}")
    
    df = pd.read_csv(params_csv_path)
    # CSV has one row with columns matching OPT_NAMES
    params_dict = df.iloc[0].to_dict()
    return params_dict


def dict_to_theta_raw(params_dict, device):
    """
    Convert learned parameter dict back to theta_raw tensor.
    Since we have theta_pos (the learned parameters after transform),
    we need to invert the transform to get theta_raw.
    """
    theta_raw_list = []
    for name in OPT_NAMES:
        val = float(params_dict[name])
        if name in SIGNED_PARAMS:
            # Invert tanh: if theta_pos = tanh(theta_raw), then theta_raw = inv_tanh(theta_pos)
            theta_raw_list.append(inv_tanh(torch.tensor(val, dtype=torch.float32, device=device)))
        else:
            # Invert softplus: if theta_pos = softplus(theta_raw), then theta_raw = inv_softplus(theta_pos)
            theta_raw_list.append(inv_softplus_torch(torch.tensor(val, dtype=torch.float32, device=device)))
    
    theta_raw = torch.stack(theta_raw_list)
    theta_raw.requires_grad_(False)  # We're not optimizing, just visualizing
    return theta_raw


def visualize_sample(
    sample_id,
    learned_theta_raw,
    p_default,
    device,
    clinical_data_path="data/nature_immune_processed",
    slide_response_path="data/nature_immune_processed/slide_responses.csv",
    output_dir="sample_visualizations",
    NumIter=100,
    w=0.1,
    temp=10.0,
    dose=200.0,
    interval=21.0,
):
    """
    Generate WITH-DRUG and NO-DRUG GIFs for a single sample.
    
    Parameters
    ----------
    sample_id : str
        Sample ID (e.g., "2e82_08")
    learned_theta_raw : torch.Tensor
        Learned theta_raw parameters
    p_default : Params
        Default parameter object
    device : torch.device
        Computation device
    clinical_data_path : str
        Path to data directory
    slide_response_path : str
        Path to slide_responses.csv for labels
    output_dir : str
        Base output directory for GIFs
    NumIter : int
        Number of simulation iterations
    w : float
        Time step
    temp : float
        Temperature for logit transformation
    dose : float
        Drug dose
    interval : float
        Drug dosing interval
    """
    
    # Load label info (for printing)
    label_map = load_labels(slide_response_path)
    
    # Build file path
    sample_file = os.path.join(clinical_data_path, f"{sample_id}.npy")
    if not os.path.exists(sample_file):
        print(f"[SKIP] Sample file not found: {sample_file}")
        return False
    
    # Load sample data
    try:
        x_col, rows, cols = load_sample_xcol(sample_file)
    except Exception as e:
        print(f"[SKIP] Failed to load sample {sample_id}: {e}")
        return False
    
    # Get response label if available
    resp_str = label_map.get(sample_id, "unknown")
    y_true = response_to_y(resp_str) if resp_str in ["R", "NR"] else None
    
    print(f"\n[INFO] Visualizing sample: {sample_id} (response: {resp_str}, y: {y_true})")
    print(f"  Grid size: {rows}x{cols}")
    
    # Create p_fixed for this sample
    p_fixed = copy.deepcopy(p_default)
    p_fixed.rows = rows
    p_fixed.cols = cols
    
    # Get theta_pos from theta_raw
    theta_pos = theta_raw_to_params(learned_theta_raw)
    
    # Convert x_col to torch tensor
    x_col_torch = torch.from_numpy(x_col).to(device=device, dtype=torch.float32)
    
    # Create output directories
    viz_with = os.path.join(output_dir)
    viz_without = os.path.join(output_dir)
    os.makedirs(viz_with, exist_ok=True)
    os.makedirs(viz_without, exist_ok=True)
    
    # ========== WITH DRUG ==========
    try:
        u_drug = eval_u_keytruda_input(w=w, dose=dose, interval=interval)
        X_drug, _ = SimpleSolver_torch(
            eval_f_torch,
            x_start=x_col_torch,
            theta_pos=theta_pos,
            p_fixed=p_fixed,
            eval_u=u_drug,
            NumIter=NumIter,
            w=w,
            device=device,
        )
        X_drug_np = X_drug.detach().cpu().numpy()
        # X_drug has shape (state_size, num_timesteps) due to SimpleSolver stacking on dim=1
        # Need to transpose to (num_timesteps, state_size) for proper indexing
        X_drug_ts = X_drug.t()  # Transpose to (num_timesteps, state_size)
        X_drug_ts_for_plot = X_drug_ts.clone()  # Keep for plotting
        
        # Sample 100 frames evenly from the trajectory
        num_frames = 100
        total_timesteps = X_drug_ts.shape[0]
        frame_indices = np.linspace(0, total_timesteps - 1, num_frames, dtype=int)
        X_drug_sampled_ts = X_drug_ts[frame_indices]
        # Transpose back to (state_size, num_frames) for create_network_evolution_gif
        X_drug_sampled_np = X_drug_sampled_ts.t().detach().cpu().numpy()
        
        # Calculate percent change in cancer cells (final vs initial)
        total_cancer_initial = total_cancer_from_xcol_torch(x_col_torch)
        # Get final state (last timestep)
        X_drug_final = X_drug_ts[-1]  # Last timestep
        total_cancer_final_drug = total_cancer_from_xcol_torch(X_drug_final)
        pct_change_drug = pct_change_torch(total_cancer_final_drug, total_cancer_initial)
        
        print(f"  WITH drug: {total_cancer_initial.item():.0f} → {total_cancer_final_drug.item():.0f} (Percent change: {pct_change_drug.item():.2f}%)")
        gif_path_drug = create_network_evolution_gif(
            X_drug_sampled_np,
            p_fixed,
            output_dir=viz_with,
            title_prefix=f"{sample_id}_with_drug",
            fps=FPS,
            dpi=120,
            save=True,
            show=False,
        )
        print(f"  [SUCCESS] WITH-DRUG GIF: {gif_path_drug}")
    except Exception as e:
        print(f"  [FAILED] WITH-DRUG GIF: {e}")
    
    # ========== NO DRUG ==========
    try:
        u_zero = make_eval_u_zero()
        X_nodrug, _ = SimpleSolver_torch(
            eval_f_torch,
            x_start=x_col_torch,
            theta_pos=theta_pos,
            p_fixed=p_fixed,
            eval_u=u_zero,
            NumIter=NumIter,
            w=w,
            device=device,
        )
        X_nodrug_np = X_nodrug.detach().cpu().numpy()
        # X_nodrug has shape (state_size, num_timesteps) due to SimpleSolver stacking on dim=1
        # Need to transpose to (num_timesteps, state_size) for proper indexing
        X_nodrug_ts = X_nodrug.t()  # Transpose to (num_timesteps, state_size)
        X_nodrug_ts_for_plot = X_nodrug_ts.clone()  # Keep for plotting
        
        # Sample 100 frames evenly from the trajectory
        num_frames = 100
        total_timesteps = X_nodrug_ts.shape[0]
        frame_indices = np.linspace(0, total_timesteps - 1, num_frames, dtype=int)
        X_nodrug_sampled_ts = X_nodrug_ts[frame_indices]
        # Transpose back to (state_size, num_frames) for create_network_evolution_gif
        X_nodrug_sampled_np = X_nodrug_sampled_ts.t().detach().cpu().numpy()
        
        # Calculate percent change in cancer cells (final vs initial)
        # Get final state (last timestep)
        X_nodrug_final = X_nodrug_ts[-1]  # Last timestep
        total_cancer_final_nodrug = total_cancer_from_xcol_torch(X_nodrug_final)
        pct_change_nodrug = pct_change_torch(total_cancer_final_nodrug, total_cancer_initial)
        
        print(f"  NO drug: {total_cancer_initial.item():.0f} → {total_cancer_final_nodrug.item():.0f} (Percent change: {pct_change_nodrug.item():.2f}%)")
        gif_path_nodrug = create_network_evolution_gif(
            X_nodrug_sampled_np,
            p_fixed,
            output_dir=viz_without,
            title_prefix=f"{sample_id}_no_drug",
            fps=FPS,
            dpi=120,
            save=True,
            show=False,
        )
        print(f"  [SUCCESS] NO-DRUG GIF: {gif_path_nodrug}")
    except Exception as e:
        print(f"  [FAILED] NO-DRUG GIF: {e}")
    
    # ========== EVOLUTION PLOT ==========
    try:
        if 'X_drug_ts_for_plot' in locals() and 'X_nodrug_ts_for_plot' in locals():
            plot_path = plot_evolution_comparison(
                X_drug_ts_for_plot,
                X_nodrug_ts_for_plot,
                rows,
                cols,
                w,
                sample_id,
                output_dir=output_dir,
            )
            print(f"  [SUCCESS] Evolution plot: {plot_path}")
    except Exception as e:
        print(f"  [FAILED] Evolution plot: {e}")
    
    return True


if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python visualize_samples.py <sample1> <sample2> ... <sampleN>")
        print("Example: python visualize_samples.py 2e82_08 2e77_02 2e83_01")
        sys.exit(1)
    
    sample_ids = sys.argv[1:]
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Paths
    learned_params_path = "final_report_drug_modeling_counterfactual_minpath/learned_parameters.csv"
    clinical_data_path = "data/nature_immune_processed"
    slide_response_path = "data/nature_immune_processed/slide_responses.csv"
    output_base = "sample_visualizations"
    
    # Load learned parameters
    print(f"\nLoading learned parameters from: {learned_params_path}")
    try:
        learned_params = load_learned_parameters(learned_params_path)
        print(f"  Loaded {len(learned_params)} parameters")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Convert to theta_raw
    theta_raw_star = dict_to_theta_raw(learned_params, device)
    
    # Default parameters (must match what was used in training)
    p_default = Params(
        lambda_C=0.33, K_C=40, d_C=0.01, k_T=2, K_K=25, D_C=0.01,
        lambda_T=0.0001, K_T=25, K_R=20, d_T=0.3, k_A=0.5, K_A=100, D_T=0.1,
        d_A=0.0315, rows=1, cols=1
    )
    
    # Simulation hyperparams (match training)
    NumIter = 630
    w = 0.1
    temp = 10.0
    dose = 200.0
    interval = 21.0
    
    # Generate visualizations for each sample
    print(f"\nVisualizing {len(sample_ids)} sample(s)...")
    success_count = 0
    for sample_id in sample_ids:
        if visualize_sample(
            sample_id,
            theta_raw_star,
            p_default,
            device,
            clinical_data_path=clinical_data_path,
            slide_response_path=slide_response_path,
            output_dir=output_base,
            NumIter=NumIter,
            w=w,
            temp=temp,
            dose=dose,
            interval=interval,
        ):
            success_count += 1
    
    print(f"\n[SUMMARY] Successfully visualized {success_count}/{len(sample_ids)} samples")
    print(f"GIFs saved to: {output_base}/{{with_drug, no_drug}}/")
