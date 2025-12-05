import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from eval_u_keytruda_input import eval_u_keytruda_input

torch.set_default_dtype(torch.float32)

# ============================================================
# 0) Device helper
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# 1) Params
# ============================================================

class Params:
    """
    Keep the same field names as your original Params.
    We'll treat p_default as the source of FIXED parameters.
    Optimized parameters are provided separately by theta_pos.
    """
    def __init__(self, lambda_C, K_C, d_C, k_T, K_K, D_C,
                       lambda_T, K_T, K_R, d_T, k_A, K_A, D_T,
                       d_A, rows, cols):
        self.lambda_C = lambda_C
        self.K_C = K_C
        self.d_C = d_C

        self.k_T = k_T
        self.K_K = K_K
        self.D_C = D_C

        self.lambda_T = lambda_T
        self.K_T = K_T
        self.K_R = K_R
        self.d_T = d_T

        self.k_A = k_A
        self.K_A = K_A
        self.D_T = D_T

        self.d_A = d_A

        self.rows = rows
        self.cols = cols


# ============================================================
# 2) Optimized subset
# ============================================================

OPT_NAMES = ["K_K", "D_C", "lambda_T", "K_R", "D_T"]

def softplus_torch(x):
    return F.softplus(x)

def inv_softplus_torch(y, eps=1e-8):
    """
    Numerically safe inverse softplus:
      softplus(x) = log(1 + exp(x))
      => x = log(exp(y) - 1)
    """
    return torch.log(torch.exp(y) - 1.0 + eps)

def pack_from_default_torch(p_default, device):
    """
    Initialize theta_raw so softplus(theta_raw) ~ default values.
    """
    vals = torch.tensor([getattr(p_default, k) for k in OPT_NAMES],
                        dtype=torch.float32, device=device)
    theta_raw = inv_softplus_torch(vals)
    theta_raw.requires_grad_(True)
    return theta_raw

def theta_raw_to_positive(theta_raw):
    return softplus_torch(theta_raw)


# ============================================================
# 3) Autograd-safe 4-neighbor Laplacian (vectorized)
# ============================================================

def laplacian4_vec(X):
    """
    X: (rows, cols) tensor
    Returns 4-neighbor unnormalized Laplacian:
        sum(neighbors) - n_nb * center
    with correct edge handling.
    """
    rows, cols = X.shape
    device = X.device
    dtype = X.dtype

    sum_nb = torch.zeros((rows, cols), device=device, dtype=dtype)
    nnb = torch.zeros((rows, cols), device=device, dtype=dtype)

    # up neighbor contributes to cell below
    if rows > 1:
        sum_nb[1:, :] = sum_nb[1:, :] + X[:-1, :]
        nnb[1:, :] = nnb[1:, :] + 1.0

        # down neighbor contributes to cell above
        sum_nb[:-1, :] = sum_nb[:-1, :] + X[1:, :]
        nnb[:-1, :] = nnb[:-1, :] + 1.0

    if cols > 1:
        # left neighbor contributes to cell right
        sum_nb[:, 1:] = sum_nb[:, 1:] + X[:, :-1]
        nnb[:, 1:] = nnb[:, 1:] + 1.0

        # right neighbor contributes to cell left
        sum_nb[:, :-1] = sum_nb[:, :-1] + X[:, 1:]
        nnb[:, :-1] = nnb[:, :-1] + 1.0

    return sum_nb - nnb * X


# ============================================================
# 4) eval_f_torch with consistent signature
# ============================================================

def eval_f_torch(x_col, theta_pos, p_fixed, r_A):
    """
    x_col: (N,1) tensor with layout [C,T,A] repeating
    theta_pos: (7,) tensor mapped via softplus
    p_fixed: Params (fixed baseline values)
    r_A: scalar tensor
    """
    rows = p_fixed.rows
    cols = p_fixed.cols

    # Fixed params
    lambda_C = p_fixed.lambda_C
    K_C = p_fixed.K_C
    d_C = p_fixed.d_C

    k_T = p_fixed.k_T
    K_T = p_fixed.K_T
    d_T = p_fixed.d_T
    d_A = p_fixed.d_A
    k_A = p_fixed.k_A
    K_A = p_fixed.K_A

    # Learned subset
    K_K = theta_pos[0]
    D_C = theta_pos[1]
    lambda_T = theta_pos[2]
    K_R = theta_pos[3]
    D_T = theta_pos[4]

    eps = 1e-12

    # reshape to grid
    x_flat = x_col.view(-1)
    N = x_flat.numel()
    assert N == rows * cols * 3, f"State length {N} != rows*cols*3 ({rows*cols*3})"

    x_grid = x_flat.view(rows, cols, 3)
    C = x_grid[:, :, 0]
    T = x_grid[:, :, 1]
    A = x_grid[:, :, 2]

    # diffusion
    lapC = laplacian4_vec(C)
    lapT = laplacian4_vec(T)

    # tumor dynamics
    dCdt = (
        lambda_C * C * (1.0 - C / (K_C + eps))
        - d_C * C
        - (k_T * C * T) / (C + K_K + eps)
        + D_C * lapC
    )

    # T cell dynamics
    drug_boost = (k_A * A) / (A + K_A + eps)
    dTdt = (
        lambda_T * (C / (C + K_R + eps)) * (1.0 - T / (K_T + eps))
        - d_T * T
        + drug_boost * T
        + D_T * lapT
    )

    # drug PK per cell
    dAdt = r_A - d_A * A

    f_grid = torch.stack([dCdt, dTdt, dAdt], dim=2)  # (rows, cols, 3)
    f_flat = f_grid.reshape(-1, 1)                   # (N,1)
    return f_flat


# ============================================================
# 5) SimpleSolver_torch (no in-place history writes)
# ============================================================

def SimpleSolver_torch(
    eval_f,
    x_start,
    theta_pos,
    p_fixed,
    eval_u,
    NumIter,
    w,
    device
):
    """
    Forward Euler integrator.
    Returns:
      X: (N, NumIter+1) tensor
      t: (NumIter+1,) tensor
    """
    NumIter = int(NumIter)

    # x0: ensure tensor on device
    x0 = torch.as_tensor(x_start, dtype=torch.float32, device=device)
    if x0.ndim == 1:
        x0 = x0.view(-1, 1)

    X_list = [x0]
    t_list = [torch.tensor(0.0, dtype=torch.float32, device=device)]

    for k in range(NumIter):
        x_curr = X_list[-1]
        t_curr = t_list[-1]

        # python float time for stateful dosing logic
        t_float = float(t_curr.item())
        u_val = eval_u(t_float)

        r_A = torch.tensor(u_val, dtype=torch.float32, device=device)

        f_val = eval_f(x_curr, theta_pos, p_fixed, r_A)
        x_next = x_curr + w * f_val

        X_list.append(x_next)
        t_list.append(t_curr + w)

    # Stack into (N, T)
    X = torch.stack([xi.squeeze(1) for xi in X_list], dim=1)
    t = torch.stack(t_list)

    return X, t


# ============================================================
# 6) Cancer summary helpers
# ============================================================

def total_cancer_from_xcol_torch(x_col):
    """
    x_col: (N,1) or (N,) or X[:, t] extraction
    Layout: [C,T,A] repeating
    """
    if x_col.ndim == 2:
        C_vals = x_col[0::3, 0]
    else:
        C_vals = x_col[0::3]
    return C_vals.sum()

def pct_change_torch(final, initial, eps=1e-12):
    return (final - initial) / (initial + eps) * 100.0


# ============================================================
# 7) Simulate → pct change → logit
# ============================================================

def simulate_pct_change_cancer_torch(
    x_col_np,
    p_default,
    theta_raw,
    rows,
    cols,
    NumIter=8400,
    w=0.01,
    dose=200.0,
    interval=21.0,
    temp=10.0,
    device=None,
):
    if device is None:
        device = get_device()

    # x_col tensor
    x_col = torch.from_numpy(x_col_np).to(device=device, dtype=torch.float32)
    if x_col.ndim == 1:
        x_col = x_col.view(-1, 1)

    # baseline cancer
    C0 = total_cancer_from_xcol_torch(x_col)

    # positive theta
    theta_pos = theta_raw_to_positive(theta_raw)

    # fixed params for this sample
    p_fixed = copy.deepcopy(p_default)
    p_fixed.rows = rows
    p_fixed.cols = cols

    # fresh stateful dosing func per simulation
    u_func = eval_u_keytruda_input(w=w, dose=dose, interval=interval)

    # forward solve
    X, t_vec = SimpleSolver_torch(
        eval_f_torch,
        x_start=x_col,
        theta_pos=theta_pos,
        p_fixed=p_fixed,
        eval_u=u_func,
        NumIter=NumIter,
        w=w,
        device=device
    )

    # final cancer
    x_final = X[:, -1]
    Cf = total_cancer_from_xcol_torch(x_final)

    pct = pct_change_torch(Cf, C0)

    # more negative pct -> higher responder prob
    logit = - pct / temp

    return pct, logit


# ============================================================
# 8) Dataset utilities
# ============================================================

def load_sorted_files(CSV_PATH, clinical_data_path):
    df = pd.read_csv(CSV_PATH)
    df_sorted = df.sort_values(by="first_two_product", ascending=True)

    files = [os.path.basename(fp) for fp in df_sorted["full_path"].tolist()]
    file_paths = [os.path.join(clinical_data_path, f) for f in files]
    return df_sorted, files, file_paths

def load_labels(slide_response_path):
    df = pd.read_csv(slide_response_path)
    label_map = {}
    for _, row in df.iterrows():
        label_map[str(row["Slide.id.Ab"])] = str(row["Response"])
    return label_map

def response_to_y(resp):
    return 1.0 if resp == "R" else 0.0

def load_sample_xcol(file_path):
    x_arr = np.load(file_path)
    rows, cols, channels = x_arr.shape
    assert channels == 3, "Expected 3 channels [C,T,A]."

    # keep your orientation rule if desired
    if rows < cols:
        x_arr = x_arr.transpose(1, 0, 2)
        rows, cols, channels = x_arr.shape

    x_col = x_arr.reshape(rows * cols * channels, 1)
    return x_col, rows, cols


# ============================================================
# 9) Dataset loss (Torch)
# ============================================================

def dataset_loss_torch(
    theta_raw,
    samples,
    p_default_base,
    device,
    NumIter=8400,
    w=0.01,
    dose=200.0,
    interval=21.0,
    temp=10.0,
    pos_weight=2.333,
):
    logits = []
    ys = []

    for s in samples:
        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        _, logit = simulate_pct_change_cancer_torch(
            x_col_np=s["x_col"],
            p_default=p_def,
            theta_raw=theta_raw,
            rows=s["rows"],
            cols=s["cols"],
            NumIter=NumIter,
            w=w,
            dose=dose,
            interval=interval,
            temp=temp,
            device=device,
        )
        logits.append(logit)
        ys.append(torch.tensor(s["y"], dtype=torch.float32, device=device))

    z_vec = torch.stack(logits)   # (B,)
    y_vec = torch.stack(ys)       # (B,)

    pos_w = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    loss = F.binary_cross_entropy_with_logits(z_vec, y_vec, pos_weight=pos_w)

    return loss


# ============================================================
# 10) Accuracy (Torch, no grad)
# ============================================================

def predict_label_from_logit_torch(logit):
    return 1 if float(logit.item()) >= 0 else 0

@torch.no_grad()
def evaluate_accuracy_torch(
    theta_raw,
    samples,
    p_default_base,
    device,
    NumIter=8400,
    w=0.01,
    dose=200.0,
    interval=21.0,
    temp=10.0,
):
    correct = 0
    total = 0

    for s in samples:
        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        _, logit = simulate_pct_change_cancer_torch(
            x_col_np=s["x_col"],
            p_default=p_def,
            theta_raw=theta_raw,
            rows=s["rows"],
            cols=s["cols"],
            NumIter=NumIter,
            w=w,
            dose=dose,
            interval=interval,
            temp=temp,
            device=device,
        )

        pred = predict_label_from_logit_torch(logit)
        if pred == int(s["y"]):
            correct += 1
        total += 1

    return correct / max(total, 1)


# ============================================================
# 11) GD training (validate ONCE at end)
# ============================================================

def run_gd_torch(
    train_samples,
    p_default,
    device,
    NumIter=8400,
    w=0.01,
    dose=200.0,
    interval=21.0,
    temp=10.0,
    lr=1e-2,
    epochs=10,
    pos_weight=2.333,
    verbose=True,
):
    theta_raw = pack_from_default_torch(p_default, device=device)

    optimizer = torch.optim.Adam([theta_raw], lr=lr)

    history = {
        "train_loss": []
    }

    for ep in range(1, epochs + 1):
        optimizer.zero_grad()

        train_loss = dataset_loss_torch(
            theta_raw, train_samples, p_default, device,
            NumIter=NumIter, w=w, dose=dose, interval=interval, temp=temp,
            pos_weight=pos_weight
        )

        train_loss.backward()
        optimizer.step()

        history["train_loss"].append(float(train_loss.detach().cpu()))

        if verbose:
            print(f"[GD] Epoch {ep:03d} | train loss {float(train_loss):.4f}", flush=True)

    return theta_raw, history


# ============================================================
# 12) Main
# ============================================================

if __name__ == "__main__":
    device = get_device()
    print("Using device:", device)

    # ---------- Paths ----------
    clinical_data_path = "data/nature_immune_processed"
    CSV_PATH = "data_preprocessing_notebooks/npy_dimensions_sorted.csv"
    slide_response_path = "data/nature_immune_processed/slide_responses.csv"

    # ---------- Debug hyperparams ----------
    TOP_TRAIN = 20
    NumIter = 8400     # start small, then scale up
    w = 0.01
    temp = 10.0
    pos_weight = 2.333
    lr = 0.3
    epochs = 10

    # ---------- Defaults ----------
    p_default = Params(
        lambda_C=0.7, K_C=28, d_C=0.01, k_T=4, K_K=25, D_C=0.0005,
        lambda_T=0.05, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
        d_A=0.0315, rows=1, cols=1
    )

    # ---------- Load ordering ----------
    df_sorted, files, file_paths = load_sorted_files(CSV_PATH, clinical_data_path)
    label_map = load_labels(slide_response_path)

    # ---------- Build samples ----------
    samples = []
    for file_name, fp in zip(files, file_paths):
        sample_id = file_name.replace(".npy", "")
        if sample_id not in label_map:
            continue

        resp = label_map[sample_id]
        y = response_to_y(resp)

        x_col, rows, cols = load_sample_xcol(fp)

        samples.append({
            "sample_id": sample_id,
            "x_col": x_col,
            "rows": rows,
            "cols": cols,
            "y": y,
        })

        if len(samples) >= (TOP_TRAIN + 5):
            break

    train_samples = samples[:TOP_TRAIN]
    val_samples = samples[TOP_TRAIN:TOP_TRAIN+5]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    # ---------- Train ----------
    theta_raw_star, history = run_gd_torch(
        train_samples=train_samples,
        p_default=p_default,
        device=device,
        NumIter=NumIter,
        w=w,
        temp=temp,
        lr=lr,
        epochs=epochs,
        pos_weight=pos_weight,
        verbose=True
    )

    # ---------- Validate ONCE ----------
    train_acc = evaluate_accuracy_torch(
        theta_raw_star, train_samples, p_default, device,
        NumIter=NumIter, w=w, temp=temp
    )
    val_acc = evaluate_accuracy_torch(
        theta_raw_star, val_samples, p_default, device,
        NumIter=NumIter, w=w, temp=temp
    )

    print("\nFinal accuracy:")
    print(f"  Train acc: {train_acc:.3f}")
    print(f"  Val acc:   {val_acc:.3f}")

    # ---------- Report learned params ----------
    theta_pos = theta_raw_to_positive(theta_raw_star).detach().cpu().numpy()
    learned = {k: float(v) for k, v in zip(OPT_NAMES, theta_pos)}

    print("\nLearned parameters:")
    for k in OPT_NAMES:
        print(f"  {k}: {learned[k]:.6g}")

    # ---------- Save history ----------
    out_dir = "test_clinical_data_visualization_torch"
    os.makedirs(out_dir, exist_ok=True)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    print(f"\nSaved training history to {os.path.join(out_dir, 'training_history.csv')}")
