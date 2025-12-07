"""
Goal (explicit, enforced in the loss):
1) NO-DRUG condition (input always 0):
      -> ALL samples should be predicted NR (y=0).
2) WITH-DRUG condition (Keytruda dosing):
      -> R should be predicted R, NR should be predicted NR
         using the true labels from slide_responses.csv.
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from eval_u_keytruda_input import eval_u_keytruda_input

# ============================================================
# 0) Reproducibility + clamps
# ============================================================

torch.set_default_dtype(torch.float32)
TORCH_SEED = 123
np.random.seed(42)
torch.manual_seed(TORCH_SEED)

CLAMP_MIN_STATE = 0.0
CLAMP_MAX_STATE = 1e6
CLAMP_PCT_MIN = -1000.0
CLAMP_PCT_MAX = 1000.0

# ============================================================
# 1) Device helper
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ============================================================
# 2) Params
# ============================================================

class Params:
    """
    p_default carries baseline values for non-optimized params.
    Optimized params are provided by theta_pos in OPT_NAMES order.
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
# 3) Optimized subset
# ============================================================

# Optimize all model parameters except K_C, K_A, and d_A.
OPT_NAMES = [
    "lambda_C", "K_C", "d_C", "k_T", "K_K", "D_C",
    "lambda_T", "K_T", "K_R", "d_T", "k_A", "D_T",
]

# Parameters that are allowed to be signed and constrained to [-1, 1]
SIGNED_PARAMS = ["D_C", "D_T"]

def softplus_torch(x):
    return F.softplus(x)

def inv_softplus_torch(y, eps=1e-8):
    return torch.log(torch.exp(y) - 1.0 + eps)


def inv_tanh(y, eps=1e-6):
    y_clamped = torch.clamp(y, -1.0 + eps, 1.0 - eps)
    return 0.5 * torch.log((1.0 + y_clamped) / (1.0 - y_clamped))

def pack_from_default_torch(p_default, device):
    defaults = [getattr(p_default, k) for k in OPT_NAMES]
    theta_raw_list = []
    for name, val in zip(OPT_NAMES, defaults):
        if name in SIGNED_PARAMS:
            v = float(np.clip(val, 0, 0.999999))
            theta_raw_list.append(inv_tanh(torch.tensor(v, dtype=torch.float32, device=device)))
        else:
            theta_raw_list.append(inv_softplus_torch(torch.tensor(float(val), dtype=torch.float32, device=device)))

    theta_raw = torch.stack(theta_raw_list)
    theta_raw.requires_grad_(True)
    return theta_raw

def theta_raw_to_positive(theta_raw):
    parts = []
    for i, name in enumerate(OPT_NAMES):
        val_raw = theta_raw[i]
        if name in SIGNED_PARAMS:
            parts.append(torch.tanh(val_raw))
        else:
            p = softplus_torch(val_raw)
            p = torch.clamp(p, 1e-6, 1e3)
            parts.append(p)
    return torch.stack(parts)


# ============================================================
# 4) Autograd-safe 4-neighbor Laplacian (vectorized)
# ============================================================

def laplacian4_vec(X):
    rows, cols = X.shape
    device = X.device
    dtype = X.dtype

    sum_nb = torch.zeros((rows, cols), device=device, dtype=dtype)
    nnb = torch.zeros((rows, cols), device=device, dtype=dtype)

    if rows > 1:
        sum_nb[1:, :] = sum_nb[1:, :] + X[:-1, :]
        nnb[1:, :] = nnb[1:, :] + 1.0

        sum_nb[:-1, :] = sum_nb[:-1, :] + X[1:, :]
        nnb[:-1, :] = nnb[:-1, :] + 1.0

    if cols > 1:
        sum_nb[:, 1:] = sum_nb[:, 1:] + X[:, :-1]
        nnb[:, 1:] = nnb[:, 1:] + 1.0

        sum_nb[:, :-1] = sum_nb[:, :-1] + X[:, 1:]
        nnb[:, :-1] = nnb[:, :-1] + 1.0

    return sum_nb - nnb * X


# ============================================================
# 5) eval_f_torch
# ============================================================

def eval_f_torch(x_col, theta_pos, p_fixed, r_A):
    rows = p_fixed.rows
    cols = p_fixed.cols

    # fixed (not optimized)
    K_A = p_fixed.K_A
    d_A = p_fixed.d_A

    # learned (OPT_NAMES order)
    lambda_C = theta_pos[0]
    K_C      = theta_pos[1]
    d_C      = theta_pos[2]
    k_T      = theta_pos[3]
    K_K      = theta_pos[4]
    D_C      = theta_pos[5]
    lambda_T = theta_pos[6]
    K_T      = theta_pos[7]
    K_R      = theta_pos[8]
    d_T      = theta_pos[9]
    k_A      = theta_pos[10]
    D_T      = theta_pos[11]
    
    eps = 1e-12

    x_flat = x_col.view(-1)
    N = x_flat.numel()
    assert N == rows * cols * 3, f"State length {N} != rows*cols*3 ({rows*cols*3})"

    x_grid = x_flat.view(rows, cols, 3)
    C = x_grid[:, :, 0]
    T = x_grid[:, :, 1]
    A = x_grid[:, :, 2]

    C = torch.clamp(C, CLAMP_MIN_STATE, CLAMP_MAX_STATE)
    T = torch.clamp(T, CLAMP_MIN_STATE, CLAMP_MAX_STATE)
    A = torch.clamp(A, CLAMP_MIN_STATE, CLAMP_MAX_STATE)

    lapC = laplacian4_vec(C)
    lapT = laplacian4_vec(T)

    dCdt = (
        lambda_C * C * (1.0 - C / (K_C + eps))
        - d_C * C
        - (k_T * C * T) / (C + K_K + eps)
        + D_C * lapC
    )

    drug_boost = (k_A * A) / (A + K_A + eps)
    dTdt = (
        lambda_T * (C / (C + K_R + eps)) * (1.0 - T / (K_T + eps))
        - d_T * T
        + drug_boost * T
        + D_T * lapT
    )

    dAdt = r_A - d_A * A

    f_grid = torch.stack([dCdt, dTdt, dAdt], dim=2)
    f_flat = f_grid.reshape(-1, 1)
    return f_flat


# ============================================================
# 6) SimpleSolver_torch
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
    NumIter = int(NumIter)

    x0 = torch.as_tensor(x_start, dtype=torch.float32, device=device)
    if x0.ndim == 1:
        x0 = x0.view(-1, 1)

    x0 = torch.nan_to_num(x0, nan=0.0, posinf=CLAMP_MAX_STATE, neginf=CLAMP_MIN_STATE)
    x0 = torch.clamp(x0, CLAMP_MIN_STATE, CLAMP_MAX_STATE)

    X_list = [x0]
    t_list = [torch.tensor(0.0, dtype=torch.float32, device=device)]

    for _ in range(NumIter):
        x_curr = X_list[-1]
        t_curr = t_list[-1]

        t_float = float(t_curr.item())
        u_val = eval_u(t_float)

        r_A = torch.tensor(u_val, dtype=torch.float32, device=device)

        f_val = eval_f(x_curr, theta_pos, p_fixed, r_A)
        x_next = x_curr + w * f_val

        x_next = torch.nan_to_num(
            x_next, nan=0.0, posinf=CLAMP_MAX_STATE, neginf=CLAMP_MIN_STATE
        )
        x_next = torch.clamp(x_next, CLAMP_MIN_STATE, CLAMP_MAX_STATE)

        X_list.append(x_next)
        t_list.append(t_curr + w)

    X = torch.stack([xi.squeeze(1) for xi in X_list], dim=1)
    t = torch.stack(t_list)
    return X, t


# ============================================================
# 7) Cancer summary helpers
# ============================================================

def total_cancer_from_xcol_torch(x_col):
    if x_col.ndim == 2:
        C_vals = x_col[0::3, 0]
    else:
        C_vals = x_col[0::3]
    return C_vals.sum()

def pct_change_torch(final, initial, eps=1e-12):
    return (final - initial) / (initial + eps) * 100.0


# ============================================================
# 8) Simulation with injectable eval_u
# ============================================================

def simulate_pct_change_cancer_torch_custom_u(
    x_col_np,
    p_default,
    theta_raw,
    rows,
    cols,
    eval_u_func,
    NumIter=8400,
    w=0.01,
    temp=10.0,
    device=None,
):
    if device is None:
        device = get_device()

    x_col = torch.from_numpy(x_col_np).to(device=device, dtype=torch.float32)
    if x_col.ndim == 1:
        x_col = x_col.view(-1, 1)

    C0 = total_cancer_from_xcol_torch(x_col)
    if not torch.isfinite(C0):
        C0 = torch.nan_to_num(C0, nan=0.0, posinf=CLAMP_MAX_STATE, neginf=CLAMP_MIN_STATE)

    theta_pos = theta_raw_to_positive(theta_raw)

    p_fixed = copy.deepcopy(p_default)
    p_fixed.rows = rows
    p_fixed.cols = cols

    X, _ = SimpleSolver_torch(
        eval_f_torch,
        x_start=x_col,
        theta_pos=theta_pos,
        p_fixed=p_fixed,
        eval_u=eval_u_func,
        NumIter=NumIter,
        w=w,
        device=device
    )

    x_final = X[:, -1]
    Cf = total_cancer_from_xcol_torch(x_final)
    if not torch.isfinite(Cf):
        Cf = torch.nan_to_num(Cf, nan=C0, posinf=CLAMP_MAX_STATE, neginf=CLAMP_MIN_STATE)

    pct = pct_change_torch(Cf, C0)
    pct = torch.clamp(pct, CLAMP_PCT_MIN, CLAMP_PCT_MAX)

    logit = - pct / temp
    logit = torch.nan_to_num(
        logit,
        nan=0.0,
        posinf=CLAMP_PCT_MAX / temp,
        neginf=CLAMP_PCT_MIN / temp
    )

    return pct, logit


def make_eval_u_zero():
    # deterministic no-drug input
    return lambda t: 0.0


# ============================================================
# 9) Dataset utilities
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
    if rows < cols:
        x_arr = x_arr.transpose(1, 0, 2)
        rows, cols, channels = x_arr.shape
    x_col = x_arr.reshape(rows * cols * channels, 1)
    return x_col, rows, cols


# ============================================================
# 10) Dual-condition loss
# ============================================================

def dataset_loss_dual_condition_torch(
    theta_raw,
    samples,
    p_default_base,
    device,
    # simulation
    NumIter=8400,
    w=0.01,
    temp=10.0,
    # with-drug regimen
    dose=200.0,
    interval=21.0,
    # class balance for with-drug condition
    pos_weight=2.333,
    # weights for the two objectives
    alpha=1.0,   # with-drug classification vs true labels
    beta=1.0,    # no-drug -> all NR constraint
):
    """
    Enforces:
      - WITH-DRUG logits match true labels
      - NO-DRUG logits all behave like NR (target 0)

    Returns a single scalar loss.
    """
    if len(samples) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

    logits_drug = []
    logits_nodrug = []
    ys_true = []

    # build two eval_u functions
    u_drug = eval_u_keytruda_input(w=w, dose=dose, interval=interval)
    u_zero = make_eval_u_zero()

    for s in samples:
        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        # WITH DRUG
        _, logit_d = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"],
            p_default=p_def,
            theta_raw=theta_raw,
            rows=s["rows"],
            cols=s["cols"],
            eval_u_func=u_drug,
            NumIter=NumIter,
            w=w,
            temp=temp,
            device=device
        )

        # NO DRUG
        _, logit_0 = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"],
            p_default=p_def,
            theta_raw=theta_raw,
            rows=s["rows"],
            cols=s["cols"],
            eval_u_func=u_zero,
            NumIter=NumIter,
            w=w,
            temp=temp,
            device=device
        )

        logits_drug.append(logit_d)
        logits_nodrug.append(logit_0)
        ys_true.append(torch.tensor(s["y"], dtype=torch.float32, device=device))

    z_drug = torch.stack(logits_drug)
    z_0 = torch.stack(logits_nodrug)
    y_true = torch.stack(ys_true)

    # mask non-finite
    mask_d = torch.isfinite(z_drug)
    mask_0 = torch.isfinite(z_0)

    z_drug = z_drug[mask_d]
    y_true_d = y_true[mask_d]

    z_0 = z_0[mask_0]

    # if everything is bad, return neutral
    if z_drug.numel() == 0 and z_0.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

    losses = []

    # WITH-DRUG BCE vs true labels
    if z_drug.numel() > 0:
        pos_w = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        loss_drug = F.binary_cross_entropy_with_logits(z_drug, y_true_d, pos_weight=pos_w)
        losses.append(alpha * loss_drug)

    # NO-DRUG BCE vs ALL-ZEROS
    if z_0.numel() > 0:
        y0 = torch.zeros_like(z_0)
        loss_0 = F.binary_cross_entropy_with_logits(z_0, y0)
        losses.append(beta * loss_0)

    return torch.stack(losses).sum()


# ============================================================
# 11) Dual-condition accuracy reporting
# ============================================================

def predict_label_from_logit_torch(logit):
    return 1 if float(logit.item()) >= 0 else 0

@torch.no_grad()
def evaluate_accuracy_condition_torch(
    theta_raw,
    samples,
    p_default_base,
    device,
    eval_u_func,
    target_mode="true",  # "true" or "all_zero"
    NumIter=8400,
    w=0.01,
    temp=10.0,
):
    """
    If target_mode == "true":
        compare predictions to each sample's y
    If target_mode == "all_zero":
        compare predictions to 0 for all samples
    """
    correct = 0
    total = 0

    for s in samples:
        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        _, logit = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"],
            p_default=p_def,
            theta_raw=theta_raw,
            rows=s["rows"],
            cols=s["cols"],
            eval_u_func=eval_u_func,
            NumIter=NumIter,
            w=w,
            temp=temp,
            device=device
        )

        if not torch.isfinite(logit):
            continue

        pred = predict_label_from_logit_torch(logit)

        if target_mode == "true":
            true = int(s["y"])
        else:
            true = 0

        correct += int(pred == true)
        total += 1

    return correct / max(total, 1)


def print_confusion_matrix(cm_dict, dataset_name=""):
    """Print confusion matrix in readable format"""
    TP, TN, FP, FN = cm_dict["TP"], cm_dict["TN"], cm_dict["FP"], cm_dict["FN"]
    print(f"\n{dataset_name} Confusion Matrix:")
    print("                 Predicted")
    print("                 Neg    Pos")
    print(f"Actual   Neg  |  {TN:3d}    {FP:3d}")
    print(f"         Pos  |  {FN:3d}    {TP:3d}")
    print(f"\nTP={TP}, TN={TN}, FP={FP}, FN={FN}")


def compute_confusion_matrix_condition(
    theta_raw,
    samples,
    p_default_base,
    device,
    eval_u_func,
    target_mode="true",
    NumIter=8400,
    w=0.01,
    temp=10.0,
):
    """
    Compute confusion matrix for given eval_u function and target_mode.
    target_mode: "true" compares to sample['y']; "all_zero" compares to 0.
    Returns dict with TP,TN,FP,FN
    """
    TP = TN = FP = FN = 0
    for s in samples:
        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        _, logit = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"],
            p_default=p_def,
            theta_raw=theta_raw,
            rows=s["rows"],
            cols=s["cols"],
            eval_u_func=eval_u_func,
            NumIter=NumIter,
            w=w,
            temp=temp,
            device=device,
        )

        if not torch.isfinite(logit):
            continue

        pred = predict_label_from_logit_torch(logit)
        if target_mode == "true":
            true = int(s["y"]) 
        else:
            true = 0

        if pred == 1 and true == 1:
            TP += 1
        elif pred == 0 and true == 0:
            TN += 1
        elif pred == 1 and true == 0:
            FP += 1
        else:
            FN += 1

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


# ============================================================
# 12) Optimizer loop (SGD/Adam)
# ============================================================

def run_sgd_dual_condition_torch(
    train_samples,
    val_samples,
    p_default,
    device,
    # simulation
    NumIter=8400,
    w=0.01,
    temp=10.0,
    # regimen
    dose=200.0,
    interval=21.0,
    # loss weights
    pos_weight=2.333,
    alpha=1.0,
    beta=1.0,
    # optimizer
    lr=1e-2,
    epochs=50,
    batch_size=None,
    verbose=True,
    early_stopping=True,
    patience=10,
    min_delta=1e-4,
    monitor="val_loss",
):
    if len(train_samples) == 0:
        raise ValueError("No training samples provided.")

    if batch_size is None or batch_size >= len(train_samples):
        batch_size = len(train_samples)
        is_full_batch = True
    else:
        is_full_batch = False

    theta_raw = pack_from_default_torch(p_default, device=device)
    optimizer = torch.optim.Adam([theta_raw], lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc_drug": [],
        "val_acc_drug": [],
        "train_acc_nodrug_all0": [],
        "val_acc_nodrug_all0": [],
    }

    best_metric = float("inf") if monitor.endswith("loss") else -float("inf")
    best_theta = theta_raw.detach().clone()
    epochs_no_improve = 0

    # eval_u funcs for accuracy
    u_drug = eval_u_keytruda_input(w=w, dose=dose, interval=interval)
    u_zero = make_eval_u_zero()

    for ep in range(1, epochs + 1):
        indices = np.random.permutation(len(train_samples))
        shuffled = [train_samples[i] for i in indices]

        epoch_train_losses = []

        for batch_start in range(0, len(train_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(train_samples))
            batch = shuffled[batch_start:batch_end]

            optimizer.zero_grad()

            batch_loss = dataset_loss_dual_condition_torch(
                theta_raw=theta_raw,
                samples=batch,
                p_default_base=p_default,
                device=device,
                NumIter=NumIter,
                w=w,
                temp=temp,
                dose=dose,
                interval=interval,
                pos_weight=pos_weight,
                alpha=alpha,
                beta=beta
            )

            batch_loss.backward()
            optimizer.step()

            epoch_train_losses.append(float(batch_loss.detach().cpu()))

        avg_train_loss = float(np.mean(epoch_train_losses))

        # val loss
        if len(val_samples) > 0:
            with torch.no_grad():
                val_loss = dataset_loss_dual_condition_torch(
                    theta_raw=theta_raw,
                    samples=val_samples,
                    p_default_base=p_default,
                    device=device,
                    NumIter=NumIter,
                    w=w,
                    temp=temp,
                    dose=dose,
                    interval=interval,
                    pos_weight=pos_weight,
                    alpha=alpha,
                    beta=beta
                )
                avg_val_loss = float(val_loss.detach().cpu())
        else:
            avg_val_loss = float("nan")

        # accuracies:
        train_acc_drug = evaluate_accuracy_condition_torch(
            theta_raw, train_samples, p_default, device,
            eval_u_func=u_drug, target_mode="true",
            NumIter=NumIter, w=w, temp=temp
        )
        val_acc_drug = evaluate_accuracy_condition_torch(
            theta_raw, val_samples, p_default, device,
            eval_u_func=u_drug, target_mode="true",
            NumIter=NumIter, w=w, temp=temp
        ) if len(val_samples) > 0 else float("nan")

        train_acc_nodrug = evaluate_accuracy_condition_torch(
            theta_raw, train_samples, p_default, device,
            eval_u_func=u_zero, target_mode="all_zero",
            NumIter=NumIter, w=w, temp=temp
        )
        val_acc_nodrug = evaluate_accuracy_condition_torch(
            theta_raw, val_samples, p_default, device,
            eval_u_func=u_zero, target_mode="all_zero",
            NumIter=NumIter, w=w, temp=temp
        ) if len(val_samples) > 0 else float("nan")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc_drug"].append(train_acc_drug)
        history["val_acc_drug"].append(val_acc_drug)
        history["train_acc_nodrug_all0"].append(train_acc_nodrug)
        history["val_acc_nodrug_all0"].append(val_acc_nodrug)

        mode_str = "[GD]" if is_full_batch else f"[SGD bs={batch_size}]"

        if verbose:
            print(
                f"{mode_str} Epoch {ep:03d} | "
                f"train_loss {avg_train_loss:.4f} | val_loss {avg_val_loss:.4f} | "
                f"drug_acc tr {train_acc_drug:.3f} va {val_acc_drug:.3f} | "
                f"noDrug(allNR)_acc tr {train_acc_nodrug:.3f} va {val_acc_nodrug:.3f}",
                flush=True
            )

        # early stopping metric
        if monitor == "val_loss" and not np.isnan(avg_val_loss):
            current_metric = avg_val_loss
            improved = current_metric < (best_metric - min_delta)
        elif monitor == "train_loss":
            current_metric = avg_train_loss
            improved = current_metric < (best_metric - min_delta)
        else:
            # default
            current_metric = avg_val_loss if not np.isnan(avg_val_loss) else avg_train_loss
            improved = current_metric < (best_metric - min_delta)

        if improved:
            best_metric = current_metric
            best_theta = theta_raw.detach().clone()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if early_stopping and epochs_no_improve >= patience:
            if verbose:
                print(
                    f"Early stopping at epoch {ep} "
                    f"(no improvement in {patience} epochs)."
                )
            return best_theta, history

    return theta_raw, history


# ============================================================
# 13) Main
# ============================================================

if __name__ == "__main__":
    device = get_device()
    print("Using device:", device)

    # ---------- Paths ----------
    clinical_data_path = "data/nature_immune_processed"
    CSV_PATH = "data_preprocessing_notebooks/npy_dimensions_sorted.csv"
    slide_response_path = "data/nature_immune_processed/slide_responses.csv"

    # ---------- Hyperparams (keep small for debug; scale later) ----------
    NumIter = 100
    w = 0.1
    temp = 10.0

    dose = 200.0
    interval = 21.0

    pos_weight = 2.333

    # dual-objective weights
    alpha = 2.0   # with-drug classification importance
    beta = 0.2    # enforce all-NR under no-drug

    lr = 0.01
    epochs = 100
    batch_size = 4
    train_val_ratio = 0.7

    # ---------- Defaults ----------
    p_default = Params(
        lambda_C=1.5, K_C=40, d_C=0.01, k_T=0.01, K_K=25, D_C=0.01,
        lambda_T=0.0001, K_T=10, K_R=10, d_T=0.3, k_A=0.50, K_A=100, D_T=0.1,
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

    # ---------- Train/Val split ----------
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    n_train = int(len(samples) * train_val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    # ---------- Optimize with dual-condition objective ----------
    theta_raw_star, history = run_sgd_dual_condition_torch(
        train_samples=train_samples,
        val_samples=val_samples,
        p_default=p_default,
        device=device,
        NumIter=NumIter,
        w=w,
        temp=temp,
        dose=dose,
        interval=interval,
        pos_weight=pos_weight,
        alpha=alpha,
        beta=beta,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    # ---------- Final evaluation for both conditions ----------
    u_drug = eval_u_keytruda_input(w=w, dose=dose, interval=interval)
    u_zero = make_eval_u_zero()

    train_acc_drug = evaluate_accuracy_condition_torch(
        theta_raw_star, train_samples, p_default, device,
        eval_u_func=u_drug, target_mode="true",
        NumIter=NumIter, w=w, temp=temp
    )
    val_acc_drug = evaluate_accuracy_condition_torch(
        theta_raw_star, val_samples, p_default, device,
        eval_u_func=u_drug, target_mode="true",
        NumIter=NumIter, w=w, temp=temp
    )

    train_acc_nodrug = evaluate_accuracy_condition_torch(
        theta_raw_star, train_samples, p_default, device,
        eval_u_func=u_zero, target_mode="all_zero",
        NumIter=NumIter, w=w, temp=temp
    )
    val_acc_nodrug = evaluate_accuracy_condition_torch(
        theta_raw_star, val_samples, p_default, device,
        eval_u_func=u_zero, target_mode="all_zero",
        NumIter=NumIter, w=w, temp=temp
    )

    print("\nFinal accuracy (WITH DRUG, true labels):")
    print(f"  Train acc: {train_acc_drug:.3f}")
    print(f"  Val acc:   {val_acc_drug:.3f}")

    print("\nFinal accuracy (NO DRUG, target ALL NR):")
    print(f"  Train acc: {train_acc_nodrug:.3f}")
    print(f"  Val acc:   {val_acc_nodrug:.3f}")

    # ---------- Confusion matrices for both conditions ----------
    train_cm_drug = compute_confusion_matrix_condition(
        theta_raw_star, train_samples, p_default, device,
        eval_u_func=u_drug, target_mode="true",
        NumIter=NumIter, w=w, temp=temp
    )
    val_cm_drug = compute_confusion_matrix_condition(
        theta_raw_star, val_samples, p_default, device,
        eval_u_func=u_drug, target_mode="true",
        NumIter=NumIter, w=w, temp=temp
    )

    train_cm_nodrug = compute_confusion_matrix_condition(
        theta_raw_star, train_samples, p_default, device,
        eval_u_func=u_zero, target_mode="all_zero",
        NumIter=NumIter, w=w, temp=temp
    )
    val_cm_nodrug = compute_confusion_matrix_condition(
        theta_raw_star, val_samples, p_default, device,
        eval_u_func=u_zero, target_mode="all_zero",
        NumIter=NumIter, w=w, temp=temp
    )

    print_confusion_matrix(train_cm_drug, "Train (WITH DRUG)")
    print_confusion_matrix(val_cm_drug, "Val   (WITH DRUG)")

    print_confusion_matrix(train_cm_nodrug, "Train (NO DRUG)")
    print_confusion_matrix(val_cm_nodrug, "Val   (NO DRUG)")

    # ---------- Report learned params ----------
    theta_pos = theta_raw_to_positive(theta_raw_star).detach().cpu().numpy()
    learned = {k: float(v) for k, v in zip(OPT_NAMES, theta_pos)}

    print("\nLearned parameters:")
    for k in OPT_NAMES:
        print(f"  {k}: {learned[k]:.6g}")

    # ---------- Save learned parameters to out_dir ----------
    out_dir = "final_report_drug_modeling_dual_objective"
    os.makedirs(out_dir, exist_ok=True)
    params_path = os.path.join(out_dir, "learned_parameters.csv")
    pd.DataFrame([learned]).to_csv(params_path, index=False)
    print(f"Saved learned parameters to: {params_path}")

    # ---------- Per-sample predictions under both conditions ----------
    per_sample = []
    for s in train_samples + val_samples:
        sid = s.get("sample_id", "")
        y_true = int(s["y"])

        # drug prediction
        _, logit_drug = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"], p_default=p_default, theta_raw=theta_raw_star,
            rows=s["rows"], cols=s["cols"], eval_u_func=u_drug,
            NumIter=NumIter, w=w, temp=temp, device=device
        )
        pred_drug = predict_label_from_logit_torch(logit_drug) if torch.isfinite(logit_drug) else None

        # no-drug prediction
        _, logit_nodrug = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"], p_default=p_default, theta_raw=theta_raw_star,
            rows=s["rows"], cols=s["cols"], eval_u_func=u_zero,
            NumIter=NumIter, w=w, temp=temp, device=device
        )
        pred_nodrug = predict_label_from_logit_torch(logit_nodrug) if torch.isfinite(logit_nodrug) else None

        correct_drug = (pred_drug is not None) and (pred_drug == y_true)
        correct_nodrug = (pred_nodrug is not None) and (pred_nodrug == 0)

        per_sample.append({
            "sample_id": sid,
            "y_true": y_true,
            "pred_drug": pred_drug,
            "pred_nodrug": pred_nodrug,
            "correct_drug": correct_drug,
            "correct_nodrug": correct_nodrug,
        })

    per_sample_df = pd.DataFrame(per_sample)
    per_sample_csv = os.path.join(out_dir, "per_sample_predictions.csv")
    per_sample_df.to_csv(per_sample_csv, index=False)
    print(f"Saved per-sample predictions to: {per_sample_csv}")

    # Individuals that satisfy three conditions simultaneously:
    #   - true label == R (y_true == 1)
    #   - correctly predicted under the drug condition
    #   - correctly predicted under the no-drug condition (predicted NR)
    triple_correct = per_sample_df[
        (per_sample_df["y_true"] == 1) &
        (per_sample_df["correct_drug"]) &
        (per_sample_df["correct_nodrug"])
    ]

    print(f"\nCount samples that are R and correctly predicted in BOTH conditions: {len(triple_correct)}")
    if len(triple_correct) > 0:
        print("Samples (sample_id) that are R and correct under drug and no-drug:")
        for sid in triple_correct["sample_id"].tolist():
            print(f"  {sid}")

    # ---------- Plot training curves ----------
    out_dir = "final_report_drug_modeling_dual_objective"
    os.makedirs(out_dir, exist_ok=True)

    epochs_list = list(range(1, len(history["train_loss"]) + 1))

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_list, history["train_loss"], label="Train Loss")
    plt.plot(epochs_list, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Dual-Objective Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)

    # Acc (with drug)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_list, history["train_acc_drug"], label="Train Acc (Drug)")
    plt.plot(epochs_list, history["val_acc_drug"], label="Val Acc (Drug)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with Drug (True Labels)")
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_drug_curves.png"), dpi=150)

    # Acc (no drug all NR)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_list, history["train_acc_nodrug_all0"], label="Train Acc (NoDrug->AllNR)")
    plt.plot(epochs_list, history["val_acc_nodrug_all0"], label="Val Acc (NoDrug->AllNR)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy without Drug (Target All NR)")
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_nodrug_curves.png"), dpi=150)

    # Save history CSV
    history_df = pd.DataFrame({
        "epoch": epochs_list,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_acc_drug": history["train_acc_drug"],
        "val_acc_drug": history["val_acc_drug"],
        "train_acc_nodrug_all0": history["train_acc_nodrug_all0"],
        "val_acc_nodrug_all0": history["val_acc_nodrug_all0"],
    })
    history_df.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    print(f"\nSaved outputs to: {out_dir}")
