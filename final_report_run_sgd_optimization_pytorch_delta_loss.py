"""
Responder-benefit objective (ASYMMETRIC) — full runnable script (UPDATED + CLASS METRICS)

Key fixes:
1) FRESH eval_u_keytruda_input for EVERY drug simulation (avoid state leakage).
2) Balanced WITH-DRUG classification loss to discourage all-NR collapse.
3) Prints per-class metrics (TPR/TNR/etc.) for WITH-DRUG predictions.

Run:
    python final_report_run_sgd_optimization_pytorch_responder_benefit.py
"""

import os
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from eval_u_keytruda_input import eval_u_keytruda_input
from VisualizeNetwork import create_network_evolution_gif

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

OPT_NAMES = [
    "lambda_C", "K_C", "d_C", "k_T", "K_K", "D_C",
    "lambda_T", "K_T", "K_R", "d_T", "k_A", "D_T",
]
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
            v = float(np.clip(val, 0, 0.9))
            theta_raw_list.append(inv_tanh(torch.tensor(v, dtype=torch.float32, device=device)))
        else:
            theta_raw_list.append(inv_softplus_torch(torch.tensor(float(val), dtype=torch.float32, device=device)))

    theta_raw = torch.stack(theta_raw_list)
    theta_raw.requires_grad_(True)
    return theta_raw

def theta_raw_to_params(theta_raw):
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
# 4) Autograd-safe 4-neighbor Laplacian
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
    assert N == rows * cols * 3

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
    return f_grid.reshape(-1, 1)

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

        u_val = eval_u(float(t_curr.item()))
        r_A = torch.tensor(u_val, dtype=torch.float32, device=device)

        f_val = eval_f(x_curr, theta_pos, p_fixed, r_A)
        x_next = x_curr + w * f_val

        x_next = torch.nan_to_num(x_next, nan=0.0, posinf=CLAMP_MAX_STATE, neginf=CLAMP_MIN_STATE)
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
# 8) Fresh eval_u factories
# ============================================================

def make_eval_u_zero():
    return lambda t: 0.0

def make_fresh_eval_u_keytruda(w, dose=200.0, interval=21.0, t_end=365.0):
    return eval_u_keytruda_input(w=w, dose=dose, interval=interval, t_end=t_end)

# ============================================================
# 9) Simulation with injectable eval_u
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

    theta_pos = theta_raw_to_params(theta_raw)

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

# ============================================================
# 10) Dataset utilities
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
    assert channels == 3
    if rows < cols:
        x_arr = x_arr.transpose(1, 0, 2)
        rows, cols, channels = x_arr.shape
    x_col = x_arr.reshape(rows * cols * channels, 1)
    return x_col, rows, cols

# ============================================================
# 11) Loss: asymmetric benefit + balanced drug classification
# ============================================================

def dataset_loss_responder_benefit_torch(
    theta_raw,
    samples,
    p_default_base,
    device,
    NumIter=8400,
    w=0.01,
    temp=10.0,
    dose=200.0,
    interval=21.0,
    alpha_drug_cls=1.0,
    beta_baseline=0.0,
    gamma=1.0,
    pos_weight_drug=None,
    benefit_margin_pct=5.0,
):
    if len(samples) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

    loss_terms = []
    u_zero = make_eval_u_zero()

    for s in samples:
        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        # fresh stateful drug function
        u_drug = make_fresh_eval_u_keytruda(w=w, dose=dose, interval=interval)

        pct_d, logit_d = simulate_pct_change_cancer_torch_custom_u(
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

        pct_0, logit_0 = simulate_pct_change_cancer_torch_custom_u(
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

        if not (torch.isfinite(pct_d) and torch.isfinite(pct_0)):
            continue

        y = float(s["y"])
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        # balanced WITH-DRUG classification
        if alpha_drug_cls > 0 and torch.isfinite(logit_d):
            if pos_weight_drug is not None:
                pw = torch.tensor(float(pos_weight_drug), dtype=torch.float32, device=device)
                loss_cls = F.binary_cross_entropy_with_logits(logit_d, y_t, pos_weight=pw)
            else:
                loss_cls = F.binary_cross_entropy_with_logits(logit_d, y_t)
            loss_terms.append(alpha_drug_cls * loss_cls)

        # asymmetric responder benefit (only for true R)
        if y == 1.0:
            delta_pct = pct_0 - pct_d
            margin = torch.tensor(benefit_margin_pct, dtype=torch.float32, device=device)
            loss_R = torch.relu(margin - delta_pct)
            loss_terms.append(gamma * loss_R)

        # optional baseline constraint (NO-DRUG -> NR-ish)
        if beta_baseline > 0 and torch.isfinite(logit_0):
            y0 = torch.zeros_like(logit_0)
            loss_base = F.binary_cross_entropy_with_logits(logit_0, y0)
            loss_terms.append(beta_baseline * loss_base)

    if len(loss_terms) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

    return torch.stack(loss_terms).sum()

# ============================================================
# 12) Prediction helpers + metrics
# ============================================================

def predict_label_from_logit_torch(logit):
    return 1 if float(logit.item()) >= 0 else 0

@torch.no_grad()
def get_logits_and_pcts_for_sample(
    theta_raw,
    sample,
    p_default_base,
    device,
    NumIter=8400,
    w=0.01,
    temp=10.0,
    dose=200.0,
    interval=21.0,
):
    p_def = copy.deepcopy(p_default_base)
    p_def.rows = sample["rows"]
    p_def.cols = sample["cols"]

    u_drug = make_fresh_eval_u_keytruda(w=w, dose=dose, interval=interval)
    u_zero = make_eval_u_zero()

    pct_d, z_d = simulate_pct_change_cancer_torch_custom_u(
        x_col_np=sample["x_col"], p_default=p_def, theta_raw=theta_raw,
        rows=sample["rows"], cols=sample["cols"], eval_u_func=u_drug,
        NumIter=NumIter, w=w, temp=temp, device=device
    )
    pct_0, z_0 = simulate_pct_change_cancer_torch_custom_u(
        x_col_np=sample["x_col"], p_default=p_def, theta_raw=theta_raw,
        rows=sample["rows"], cols=sample["cols"], eval_u_func=u_zero,
        NumIter=NumIter, w=w, temp=temp, device=device
    )
    return pct_d, pct_0, z_d, z_0

@torch.no_grad()
def evaluate_accuracy_drug(
    theta_raw,
    samples,
    p_default_base,
    device,
    NumIter=8400,
    w=0.01,
    temp=10.0,
    dose=200.0,
    interval=21.0,
):
    correct = 0
    total = 0
    for s in samples:
        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        u_drug = make_fresh_eval_u_keytruda(w=w, dose=dose, interval=interval)

        _, z = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"], p_default=p_def, theta_raw=theta_raw,
            rows=s["rows"], cols=s["cols"], eval_u_func=u_drug,
            NumIter=NumIter, w=w, temp=temp, device=device
        )
        if not torch.isfinite(z):
            continue
        pred = predict_label_from_logit_torch(z)
        true = int(s["y"])
        correct += int(pred == true)
        total += 1
    return correct / max(total, 1)

@torch.no_grad()
def evaluate_accuracy_nodrug_allNR(
    theta_raw,
    samples,
    p_default_base,
    device,
    NumIter=8400,
    w=0.01,
    temp=10.0,
):
    correct = 0
    total = 0
    u_zero = make_eval_u_zero()

    for s in samples:
        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        _, z = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"], p_default=p_def, theta_raw=theta_raw,
            rows=s["rows"], cols=s["cols"], eval_u_func=u_zero,
            NumIter=NumIter, w=w, temp=temp, device=device
        )
        if not torch.isfinite(z):
            continue
        pred = predict_label_from_logit_torch(z)
        correct += int(pred == 0)
        total += 1
    return correct / max(total, 1)

@torch.no_grad()
def evaluate_responder_benefit_rate(
    theta_raw,
    samples,
    p_default_base,
    device,
    NumIter=8400,
    w=0.01,
    temp=10.0,
    dose=200.0,
    interval=21.0,
    benefit_margin_pct=5.0,
):
    satisfied = 0
    total_R = 0
    u_zero = make_eval_u_zero()

    for s in samples:
        if int(s["y"]) != 1:
            continue

        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        u_drug = make_fresh_eval_u_keytruda(w=w, dose=dose, interval=interval)

        pct_d, _ = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"], p_default=p_def, theta_raw=theta_raw,
            rows=s["rows"], cols=s["cols"], eval_u_func=u_drug,
            NumIter=NumIter, w=w, temp=temp, device=device
        )
        pct_0, _ = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"], p_default=p_def, theta_raw=theta_raw,
            rows=s["rows"], cols=s["cols"], eval_u_func=u_zero,
            NumIter=NumIter, w=w, temp=temp, device=device
        )

        if not (torch.isfinite(pct_d) and torch.isfinite(pct_0)):
            continue

        delta_pct = pct_0 - pct_d
        total_R += 1
        if float(delta_pct.item()) >= benefit_margin_pct:
            satisfied += 1

    return satisfied / max(total_R, 1)

# ---------------- NEW: confusion + per-class metrics (WITH DRUG) ----------------

@torch.no_grad()
def compute_confusion_drug(
    theta_raw,
    samples,
    p_default_base,
    device,
    NumIter=8400,
    w=0.01,
    temp=10.0,
    dose=200.0,
    interval=21.0,
):
    TP = TN = FP = FN = 0
    used = 0

    for s in samples:
        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        u_drug = make_fresh_eval_u_keytruda(w=w, dose=dose, interval=interval)

        _, z = simulate_pct_change_cancer_torch_custom_u(
            x_col_np=s["x_col"], p_default=p_def, theta_raw=theta_raw,
            rows=s["rows"], cols=s["cols"], eval_u_func=u_drug,
            NumIter=NumIter, w=w, temp=temp, device=device
        )
        if not torch.isfinite(z):
            continue

        pred = predict_label_from_logit_torch(z)
        true = int(s["y"])

        used += 1
        if pred == 1 and true == 1:
            TP += 1
        elif pred == 0 and true == 0:
            TN += 1
        elif pred == 1 and true == 0:
            FP += 1
        else:
            FN += 1

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "used": used}

def confusion_to_rates(cm):
    TP = cm["TP"]; TN = cm["TN"]; FP = cm["FP"]; FN = cm["FN"]
    eps = 1e-12

    TPR = TP / (TP + FN + eps)  # sensitivity / recall+
    TNR = TN / (TN + FP + eps)  # specificity / recall-
    PPV = TP / (TP + FP + eps)  # precision+
    NPV = TN / (TN + FN + eps)
    F1  = (2 * PPV * TPR) / (PPV + TPR + eps)
    ACC = (TP + TN) / (TP + TN + FP + FN + eps)
    BA  = 0.5 * (TPR + TNR)

    return {
        "TPR": TPR, "TNR": TNR,
        "PPV": PPV, "NPV": NPV,
        "F1": F1, "ACC": ACC,
        "BalancedAcc": BA
    }

def print_drug_class_metrics(title, cm):
    rates = confusion_to_rates(cm)
    print(f"\n{title} (WITH DRUG) classification metrics:")
    print(f"  Used samples: {cm['used']}")
    print(f"  TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}")
    print(f"  TPR (R recall) : {rates['TPR']:.3f}")
    print(f"  TNR (NR recall): {rates['TNR']:.3f}")
    print(f"  PPV (precision): {rates['PPV']:.3f}")
    print(f"  NPV            : {rates['NPV']:.3f}")
    print(f"  F1             : {rates['F1']:.3f}")
    print(f"  ACC            : {rates['ACC']:.3f}")
    print(f"  Balanced Acc   : {rates['BalancedAcc']:.3f}")

# ============================================================
# 13) Optimizer loop
# ============================================================

# def get_phase_weights(ep, epochs):
#     e1 = int(0.4 * epochs)
#     e2 = int(0.8 * epochs)

#     if ep <= e1:
#         return dict(alpha_drug_cls=1.0, beta_baseline=0.0, gamma=0.2)
#     elif ep <= e2:
#         return dict(alpha_drug_cls=1.0, beta_baseline=0.0, gamma=1.0)
#     else:
#         return dict(alpha_drug_cls=1.0, beta_baseline=0.1, gamma=1.0)

def get_phase_weights(ep, epochs):
    return dict(alpha_drug_cls=1.0, beta_baseline=0.1, gamma=0.1)

def run_sgd_responder_benefit_torch(
    train_samples,
    val_samples,
    p_default,
    device,
    NumIter=8400,
    w=0.01,
    temp=10.0,
    dose=200.0,
    interval=21.0,
    lr=1e-2,
    epochs=80,
    batch_size=4,
    verbose=True,
    early_stopping=True,
    patience=12,
    min_delta=1e-4,
    benefit_margin_pct=5.0,
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

    # class balance
    n_pos = sum(1 for s in train_samples if float(s["y"]) == 1.0)
    n_neg = sum(1 for s in train_samples if float(s["y"]) == 0.0)
    pos_weight_drug = (n_neg / max(n_pos, 1))
    if verbose:
        print(f"[INFO] pos_weight_drug = {pos_weight_drug:.3f} (neg={n_neg}, pos={n_pos})")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc_drug": [],
        "val_acc_drug": [],
        "train_acc_nodrug": [],
        "val_acc_nodrug": [],
        "train_benefit_R": [],
        "val_benefit_R": [],
        "alpha_drug_cls": [],
        "beta_baseline": [],
        "gamma": [],
    }

    best_metric = float("inf")
    best_theta = theta_raw.detach().clone()
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        weights = get_phase_weights(ep, epochs)
        alpha_drug_cls = weights.get("alpha_drug_cls", 1.0)
        beta_baseline = weights.get("beta_baseline", 0.0)
        gamma = weights.get("gamma", 1.0)

        indices = np.random.permutation(len(train_samples))
        shuffled = [train_samples[i] for i in indices]

        epoch_train_losses = []

        for batch_start in range(0, len(train_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(train_samples))
            batch = shuffled[batch_start:batch_end]

            optimizer.zero_grad()

            batch_loss = dataset_loss_responder_benefit_torch(
                theta_raw=theta_raw,
                samples=batch,
                p_default_base=p_default,
                device=device,
                NumIter=NumIter,
                w=w,
                temp=temp,
                dose=dose,
                interval=interval,
                alpha_drug_cls=alpha_drug_cls,
                pos_weight_drug=pos_weight_drug,
                beta_baseline=beta_baseline,
                gamma=gamma,
                benefit_margin_pct=benefit_margin_pct,
            )

            batch_loss.backward()
            optimizer.step()

            epoch_train_losses.append(float(batch_loss.detach().cpu()))

        avg_train_loss = float(np.mean(epoch_train_losses)) if epoch_train_losses else 0.0

        # val loss
        if len(val_samples) > 0:
            with torch.no_grad():
                val_loss = dataset_loss_responder_benefit_torch(
                    theta_raw=theta_raw,
                    samples=val_samples,
                    p_default_base=p_default,
                    device=device,
                    NumIter=NumIter,
                    w=w,
                    temp=temp,
                    dose=dose,
                    interval=interval,
                    alpha_drug_cls=alpha_drug_cls,
                    pos_weight_drug=pos_weight_drug,
                    beta_baseline=beta_baseline,
                    gamma=gamma,
                    benefit_margin_pct=benefit_margin_pct,
                )
                avg_val_loss = float(val_loss.detach().cpu())
        else:
            avg_val_loss = float("nan")

        # monitoring
        train_acc_drug = evaluate_accuracy_drug(
            theta_raw, train_samples, p_default, device,
            NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval
        )
        val_acc_drug = evaluate_accuracy_drug(
            theta_raw, val_samples, p_default, device,
            NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval
        ) if len(val_samples) > 0 else float("nan")

        train_acc_nodrug = evaluate_accuracy_nodrug_allNR(
            theta_raw, train_samples, p_default, device,
            NumIter=NumIter, w=w, temp=temp
        )
        val_acc_nodrug = evaluate_accuracy_nodrug_allNR(
            theta_raw, val_samples, p_default, device,
            NumIter=NumIter, w=w, temp=temp
        ) if len(val_samples) > 0 else float("nan")

        train_benefit_R = evaluate_responder_benefit_rate(
            theta_raw, train_samples, p_default, device,
            NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval,
            benefit_margin_pct=benefit_margin_pct
        )
        val_benefit_R = evaluate_responder_benefit_rate(
            theta_raw, val_samples, p_default, device,
            NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval,
            benefit_margin_pct=benefit_margin_pct
        ) if len(val_samples) > 0 else float("nan")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc_drug"].append(train_acc_drug)
        history["val_acc_drug"].append(val_acc_drug)
        history["train_acc_nodrug"].append(train_acc_nodrug)
        history["val_acc_nodrug"].append(val_acc_nodrug)
        history["train_benefit_R"].append(train_benefit_R)
        history["val_benefit_R"].append(val_benefit_R)
        history["alpha_drug_cls"].append(alpha_drug_cls)
        history["beta_baseline"].append(beta_baseline)
        history["gamma"].append(gamma)

        mode_str = "[GD]" if is_full_batch else f"[SGD bs={batch_size}]"
        if verbose:
            print(
                f"{mode_str} Epoch {ep:03d} | "
                f"train_loss {avg_train_loss:.4f} | val_loss {avg_val_loss:.4f} | "
                f"alpha {alpha_drug_cls:.2f} beta {beta_baseline:.2f} gamma {gamma:.2f} | "
                f"drug_acc tr {train_acc_drug:.3f} va {val_acc_drug:.3f} | "
                f"noDrug->NR tr {train_acc_nodrug:.3f} va {val_acc_nodrug:.3f} | "
                f"R-benefit tr {train_benefit_R:.3f} va {val_benefit_R:.3f}",
                flush=True
            )

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
                print(f"Early stopping at epoch {ep} (no improvement in {patience} epochs).")
            return best_theta, history

    return theta_raw, history

if __name__ == "__main__":
    # ============================================================
    # Helper: nice confusion matrix print
    # ============================================================
    def print_confusion_matrix(title, cm):
        TP = cm.get("TP", 0)
        TN = cm.get("TN", 0)
        FP = cm.get("FP", 0)
        FN = cm.get("FN", 0)

        print(f"\n{title} Confusion Matrix (WITH DRUG)")
        print("                 Predicted")
        print("                 NR     R")
        print(f"Actual   NR  |  {TN:4d}  {FP:4d}")
        print(f"         R   |  {FN:4d}  {TP:4d}")
        print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")

    # ============================================================
    # 1) Device
    # ============================================================
    device = get_device()
    print("Using device:", device)

    # ============================================================
    # 2) Paths
    # ============================================================
    clinical_data_path = "data/nature_immune_processed"
    CSV_PATH = "data_preprocessing_notebooks/npy_dimensions_sorted.csv"
    slide_response_path = "data/nature_immune_processed/slide_responses.csv"

    # ============================================================
    # 3) Hyperparams (debug-friendly)
    # ============================================================
    NumIter = 100
    w = 0.1
    temp = 10.0

    dose = 200.0
    interval = 21.0

    lr = 0.01
    epochs = 20
    batch_size = 4
    train_val_ratio = 0.7

    benefit_margin_pct = 5.0

    # ============================================================
    # 4) Defaults
    # ============================================================
    p_default = Params(
        lambda_C=1.5, K_C=40, d_C=0.01, k_T=0.01, K_K=25, D_C=0.01,
        lambda_T=0.0001, K_T=10, K_R=10, d_T=0.3, k_A=10, K_A=100, D_T=0.1,
        d_A=0.0315, rows=1, cols=1
    )

    # ============================================================
    # 5) Load files + labels
    # ============================================================
    _, files, file_paths = load_sorted_files(CSV_PATH, clinical_data_path)
    label_map = load_labels(slide_response_path)

    # ============================================================
    # 6) Build samples
    # ============================================================
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

    if len(samples) == 0:
        raise RuntimeError("No samples loaded. Check paths and label mapping.")

    # Uncomment for quick debugging
    # samples = samples[:5]

    # ============================================================
    # 7) Train/Val split
    # ============================================================
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    n_train = int(len(samples) * train_val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    # ============================================================
    # 8) Optimize
    # ============================================================
    theta_raw_star, history = run_sgd_responder_benefit_torch(
        train_samples=train_samples,
        val_samples=val_samples,
        p_default=p_default,
        device=device,
        NumIter=NumIter,
        w=w,
        temp=temp,
        dose=dose,
        interval=interval,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        benefit_margin_pct=benefit_margin_pct,
    )

    # ============================================================
    # 9) Final scalar monitoring
    # ============================================================
    train_acc_drug = evaluate_accuracy_drug(
        theta_raw_star, train_samples, p_default, device,
        NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval
    )
    val_acc_drug = evaluate_accuracy_drug(
        theta_raw_star, val_samples, p_default, device,
        NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval
    )

    train_acc_nodrug = evaluate_accuracy_nodrug_allNR(
        theta_raw_star, train_samples, p_default, device,
        NumIter=NumIter, w=w, temp=temp
    )
    val_acc_nodrug = evaluate_accuracy_nodrug_allNR(
        theta_raw_star, val_samples, p_default, device,
        NumIter=NumIter, w=w, temp=temp
    )

    train_benefit_R = evaluate_responder_benefit_rate(
        theta_raw_star, train_samples, p_default, device,
        NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval,
        benefit_margin_pct=benefit_margin_pct
    )
    val_benefit_R = evaluate_responder_benefit_rate(
        theta_raw_star, val_samples, p_default, device,
        NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval,
        benefit_margin_pct=benefit_margin_pct
    )

    print("\nFinal monitoring:")
    print(f"  WITH-DRUG accuracy (train/val): {train_acc_drug:.3f} / {val_acc_drug:.3f}")
    print(f"  NO-DRUG predicted-NR rate (train/val): {train_acc_nodrug:.3f} / {val_acc_nodrug:.3f}")
    print(f"  R-benefit satisfaction (train/val): {train_benefit_R:.3f} / {val_benefit_R:.3f}")

    # ============================================================
    # 10) WITH-DRUG confusion matrices + class metrics
    # ============================================================
    cm_train = compute_confusion_drug(
        theta_raw_star, train_samples, p_default, device,
        NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval
    )
    cm_val = compute_confusion_drug(
        theta_raw_star, val_samples, p_default, device,
        NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval
    )
    cm_all = compute_confusion_drug(
        theta_raw_star, (train_samples + val_samples), p_default, device,
        NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval
    )

    print_drug_class_metrics("TRAIN", cm_train)
    print_drug_class_metrics("VAL", cm_val)
    print_drug_class_metrics("ALL", cm_all)

    print_confusion_matrix("TRAIN", cm_train)
    print_confusion_matrix("VAL", cm_val)
    print_confusion_matrix("ALL", cm_all)

    # ============================================================
    # 11) Count TRUE-R correct in BOTH drug + no-drug
    # Definition used:
    #   - TRUE responder (y_true == 1)
    #   - Predicted R WITH DRUG  (pred_drug == 1)
    #   - Predicted NR NO DRUG  (pred_nodrug == 0)
    # This matches your dual-criterion setup.
    # ============================================================
    both_correct_R = 0
    both_correct_R_ids = []

    for s in (train_samples + val_samples):
        if int(s["y"]) != 1:
            continue

        pct_d, pct_0, z_d, z_0 = get_logits_and_pcts_for_sample(
            theta_raw_star, s, p_default, device,
            NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval
        )

        if not (torch.isfinite(z_d) and torch.isfinite(z_0)):
            continue

        pred_drug = predict_label_from_logit_torch(z_d)
        pred_nodrug = predict_label_from_logit_torch(z_0)

        if pred_drug == 1 and pred_nodrug == 0:
            both_correct_R += 1
            both_correct_R_ids.append(s["sample_id"])

    print(
        f"\nCount of TRUE-R correctly predicted in BOTH contexts "
        f"(WITH-DRUG=R AND NO-DRUG=NR): {both_correct_R}"
    )
    if len(both_correct_R_ids) > 0:
        print("IDs:", ", ".join(both_correct_R_ids))

    # ============================================================
    # 12) Save learned params
    # ============================================================
    theta_pos_np = theta_raw_to_params(theta_raw_star).detach().cpu().numpy()
    learned = {k: float(v) for k, v in zip(OPT_NAMES, theta_pos_np)}

    out_dir = "final_report_drug_modeling_counterfactual_minpath"
    os.makedirs(out_dir, exist_ok=True)

    params_path = os.path.join(out_dir, "learned_parameters.csv")
    pd.DataFrame([learned]).to_csv(params_path, index=False)
    print(f"\nSaved learned parameters to: {params_path}")

    # ============================================================
    # 13) Per-sample predictions CSV
    # ============================================================
    per_sample = []
    all_samples = train_samples + val_samples

    for s in all_samples:
        sid = s.get("sample_id", "")
        y_true = int(s["y"])

        pct_d, pct_0, z_d, z_0 = get_logits_and_pcts_for_sample(
            theta_raw_star, s, p_default, device,
            NumIter=NumIter, w=w, temp=temp, dose=dose, interval=interval
        )

        pred_drug = predict_label_from_logit_torch(z_d) if torch.isfinite(z_d) else None
        pred_nodrug = predict_label_from_logit_torch(z_0) if torch.isfinite(z_0) else None

        delta_pct = (pct_0 - pct_d) if (torch.isfinite(pct_d) and torch.isfinite(pct_0)) else None

        benefit_satisfied = None
        if delta_pct is not None and y_true == 1:
            benefit_satisfied = bool(float(delta_pct.item()) >= benefit_margin_pct)

        per_sample.append({
            "sample_id": sid,
            "y_true": y_true,
            "pct_drug": float(pct_d.item()) if torch.isfinite(pct_d) else np.nan,
            "pct_nodrug": float(pct_0.item()) if torch.isfinite(pct_0) else np.nan,
            "delta_pct_nodrug_minus_drug": float(delta_pct.item()) if delta_pct is not None else np.nan,
            "logit_drug": float(z_d.item()) if torch.isfinite(z_d) else np.nan,
            "logit_nodrug": float(z_0.item()) if torch.isfinite(z_0) else np.nan,
            "pred_drug": pred_drug,
            "pred_nodrug": pred_nodrug,
            "benefit_margin_pct": benefit_margin_pct,
            "benefit_satisfied_if_R": benefit_satisfied,
        })

    per_sample_df = pd.DataFrame(per_sample)
    per_sample_csv = os.path.join(out_dir, "per_sample_predictions.csv")
    per_sample_df.to_csv(per_sample_csv, index=False)
    print(f"Saved per-sample predictions to: {per_sample_csv}")

    # Save confusion matrices JSON (WITH DRUG)
    cms_path = os.path.join(out_dir, "confusion_matrices_drug.json")
    with open(cms_path, "w") as fh:
        json.dump({"train": cm_train, "val": cm_val, "all": cm_all}, fh, indent=2)
    print(f"Saved drug confusion matrices to: {cms_path}")

    # ============================================================
    # 14) Final visualizations per sample (WITH + NO DRUG)
    # ============================================================
    viz_base = os.path.join(out_dir, "final_visualization")
    viz_with = os.path.join(viz_base, "with_drug")
    viz_without = os.path.join(viz_base, "no_drug")
    os.makedirs(viz_with, exist_ok=True)
    os.makedirs(viz_without, exist_ok=True)

    # theta_pos for simulation (torch tensor)
    theta_pos = theta_raw_to_params(theta_raw_star)

    create_visualizations = False
    if create_visualizations:
        print(f"\nCreating per-sample visualizations ({len(all_samples)} samples)...")

        for s in all_samples:
            sid = s.get("sample_id", "")

            p_def = copy.deepcopy(p_default)
            p_def.rows = s["rows"]
            p_def.cols = s["cols"]

            # Initial totals (channel stride = 3)
            xcol = s["x_col"]
            total_C0 = float(np.sum(xcol[0::3]))
            total_T0 = float(np.sum(xcol[1::3]))
            total_A0 = float(np.sum(xcol[2::3]))

            print(
                f"Sample ID: {sid}, "
                f"Initial Cancer: {total_C0:.1f}, "
                f"Initial T: {total_T0:.1f}, "
                f"Initial A: {total_A0:.1f}"
            )

            # -------- WITH DRUG --------
            try:
                u_drug = make_fresh_eval_u_keytruda(w=w, dose=dose, interval=interval)

                X_drug, _ = SimpleSolver_torch(
                    eval_f_torch,
                    x_start=torch.from_numpy(s["x_col"]).to(device=device, dtype=torch.float32),
                    theta_pos=theta_pos,
                    p_fixed=p_def,
                    eval_u=u_drug,
                    NumIter=NumIter,
                    w=w,
                    device=device,
                )

                Xd_np = X_drug.detach().cpu().numpy()

                gif_path = create_network_evolution_gif(
                    Xd_np,
                    p_def,
                    output_dir=viz_with,
                    title_prefix=f"{sid}_with_drug",
                    fps=10,
                    dpi=120,
                    save=True,
                    show=False
                )
                print(f"Saved WITH-DRUG GIF for {sid}: {gif_path}")

            except Exception as e:
                print(f"Failed to create WITH-DRUG GIF for {sid}: {e}")

            # -------- NO DRUG --------
            try:
                u_zero = make_eval_u_zero()

                X_0, _ = SimpleSolver_torch(
                    eval_f_torch,
                    x_start=torch.from_numpy(s["x_col"]).to(device=device, dtype=torch.float32),
                    theta_pos=theta_pos,
                    p_fixed=p_def,
                    eval_u=u_zero,
                    NumIter=NumIter,
                    w=w,
                    device=device,
                )

                X0_np = X_0.detach().cpu().numpy()

                gif_path0 = create_network_evolution_gif(
                    X0_np,
                    p_def,
                    output_dir=viz_without,
                    title_prefix=f"{sid}_no_drug",
                    fps=10,
                    dpi=120,
                    save=True,
                    show=False
                )
                print(f"Saved NO-DRUG GIF for {sid}: {gif_path0}")

            except Exception as e:
                print(f"Failed to create NO-DRUG GIF for {sid}: {e}")

        print(f"Finished visualizations. GIFs saved under: {viz_base}")

