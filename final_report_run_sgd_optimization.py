import os
import copy
import numpy as np
import pandas as pd

import autograd.numpy as anp
from autograd import grad

from eval_f import eval_f, Params
from eval_f_output import eval_f_output  # uses numpy; ok for initial totals
from eval_u_keytruda_input import eval_u_keytruda_input
from SimpleSolver_autograd import SimpleSolver_autograd


# ============================================================
# 1) Weighted BCE with logits (your function)
# ============================================================

def weighted_bce_with_logits(z, y, w_pos=2.333, w_neg=1.0):
    """
    z: logit (scalar or vector)
    y: label in {0,1}
    default weights correspond to NR:R ratio of 7:3
    """
    # stable BCE with logits:
    # y*log(1+exp(-z)) + (1-y)*log(1+exp(z))
    return (w_pos * y) * anp.log1p(anp.exp(-z)) + (w_neg * (1 - y)) * anp.log1p(anp.exp(z))


def mean_weighted_bce_with_logits(z_vec, y_vec, w_pos=2.333, w_neg=1.0):
    losses = weighted_bce_with_logits(z_vec, y_vec, w_pos=w_pos, w_neg=w_neg)
    return anp.mean(losses)


# ============================================================
# 2) Parameter handling (optimize subset)
# ============================================================

OPT_NAMES = ["K_K", "D_C", "lambda_T", "K_R", "k_A", "K_A", "D_T"]


def softplus(x):
    # stable softplus
    return anp.log1p(anp.exp(-anp.abs(x))) + anp.maximum(x, 0)


def pack_from_default(p_default):
    """Return theta_raw init so that softplus(theta_raw) ~= default value."""
    vals = anp.array([getattr(p_default, k) for k in OPT_NAMES], dtype=float)
    # inverse softplus approx: x ≈ log(exp(y)-1)
    eps = 1e-8
    theta_raw = anp.log(anp.exp(vals) - 1 + eps)
    return theta_raw


def inject_theta_into_params(theta_raw, p_default, rows, cols):
    """
    Build a fresh Params with optimized subset replaced.
    Enforce positivity via softplus.
    """
    theta = softplus(theta_raw)

    p = copy.deepcopy(p_default)
    p.rows = rows
    p.cols = cols

    for name, value in zip(OPT_NAMES, theta):
        setattr(p, name, value)  # Params class likely expects python floats

    return p


# ============================================================
# 3) Cancer summary helpers (autograd-friendly)
# ============================================================

def total_cancer_from_xcol_anp(x_col):
    """
    x_col shape: (N,1) in [C, T, A] repeating order.
    Autograd-safe sum of cancer channel.
    """
    C_vals = x_col[0::3, 0]
    return anp.sum(C_vals)


def pct_change_anp(final, initial):
    eps = 1e-12
    return (final - initial) / (initial + eps) * 100.0


# ============================================================
# 4) Forward simulation → logit
# ============================================================

def simulate_pct_change_cancer(
    x_col_np,
    p_default,
    theta_raw,
    NumIter=8400,
    w=0.01,
    dose=200.0,
    interval=21.0,
    temp=10.0,
):
    """
    Returns:
      pct_change_cancer (anp scalar)
      logit (anp scalar), where higher logit => more likely Responder (y=1)

    Important:
    - x_col_np is numpy input from file (not part of grad).
    - We convert to anp.
    - We create a fresh stateful u_func each call.
    """
    # infer grid from x length if needed
    # But you already know rows/cols from x_arr shape upstream.
    # Here we assume x_col_np already matches rows, cols in default later injection.

    x_col = anp.array(x_col_np)

    # initial total cancer
    C0 = total_cancer_from_xcol_anp(x_col)

    # We need rows/cols to build Params correctly.
    # Your x_col corresponds to rows*cols*3.
    N = x_col.shape[0]
    assert N % 3 == 0
    n_cells = N // 3

    # We don't know rows/cols from flattened vector alone.
    # So this function assumes caller will pass rows/cols via p_default fields,
    # which we will trust here. We'll read them:
    rows = p_default.rows
    cols = p_default.cols
    assert rows * cols == n_cells, \
        f"Mismatch: rows*cols={rows*cols} but flattened cells={n_cells}"

    # build params with optimized subset
    p = inject_theta_into_params(theta_raw, p_default, rows, cols)

    print(f"Simulating {p.rows}x{p.cols} | NumIter={NumIter}", flush=True)

    # stateful drug input - create fresh each call
    u_func = eval_u_keytruda_input(w=w, dose=dose, interval=interval)

    # run autograd-safe solver
    X, t = SimpleSolver_autograd(
        eval_f,
        x_start=x_col,
        p=p,
        eval_u=u_func,
        NumIter=NumIter,
        w=w,
        visualize=False
    )

    x_final = X[:, -1].reshape((-1, 1))
    Cf = total_cancer_from_xcol_anp(x_final)

    pct = pct_change_anp(Cf, C0)

    # Convert predicted % change to a logit
    # more negative % change -> higher probability of R
    # temp controls sharpness
    logit = - pct / temp

    return pct, logit


# ============================================================
# 5) Dataset construction
# ============================================================

def load_sorted_files(CSV_PATH, clinical_data_path):
    """
    Your CSV has 'full_path' and 'first_two_product'.
    We'll use basename to open in clinical_data_path.
    """
    df = pd.read_csv(CSV_PATH)
    df_sorted = df.sort_values(by="first_two_product", ascending=True)

    files = [os.path.basename(fp) for fp in df_sorted["full_path"].tolist()]
    file_paths = [os.path.join(clinical_data_path, f) for f in files]

    return df_sorted, files, file_paths


def load_labels(slide_response_path):
    df = pd.read_csv(slide_response_path)
    # Expect Slide.id.Ab and Response in {"R", "NR"}
    label_map = {}
    for _, row in df.iterrows():
        label_map[str(row["Slide.id.Ab"])] = str(row["Response"])
    return label_map


def response_to_y(resp):
    # y=1 for R, y=0 for NR
    return 1.0 if resp == "R" else 0.0


def load_sample_xcol(file_path):
    """
    Load .npy which is (rows, cols, channels) with channels=3.
    Ensure rows >= cols convention if you want.
    """
    x_arr = np.load(file_path)
    rows, cols, channels = x_arr.shape
    assert channels == 3, "Expected 3 channels [C,T,A]."

    if rows < cols:
        x_arr = x_arr.transpose(1, 0, 2)
        rows, cols, channels = x_arr.shape

    x_col = x_arr.reshape(rows * cols * channels, 1)
    return x_col, rows, cols


# ============================================================
# 6) Loss over a set of samples
# ============================================================

def dataset_loss(
    theta_raw,
    samples,
    p_default_base,
    NumIter=8400,
    w=0.01,
    dose=200.0,
    interval=21.0,
    temp=10.0,
    w_pos=2.333,
    w_neg=1.0,
):
    """
    samples: list of dicts:
      {
        "x_col": np array (N,1),
        "rows": int,
        "cols": int,
        "y": float 0/1,
        "sample_id": str
      }
    """
    logits = []
    ys = []

    for s in samples:
        # set rows/cols on a copy of default for this sample
        p_def = copy.deepcopy(p_default_base)
        p_def.rows = s["rows"]
        p_def.cols = s["cols"]

        _, logit = simulate_pct_change_cancer(
            x_col_np=s["x_col"],
            p_default=p_def,
            theta_raw=theta_raw,
            NumIter=NumIter,
            w=w,
            dose=dose,
            interval=interval,
            temp=temp,
        )
        logits.append(logit)
        ys.append(s["y"])

    z_vec = anp.array(logits)
    y_vec = anp.array(ys)

    return mean_weighted_bce_with_logits(z_vec, y_vec, w_pos=w_pos, w_neg=w_neg)


# ============================================================
# 7) Accuracy evaluation (no grad needed)
# ============================================================

def predict_label_from_logit(logit):
    # sigmoid(logit) >= 0.5 iff logit >= 0
    return 1 if logit >= 0 else 0


def evaluate_accuracy(
    theta_raw,
    samples,
    p_default_base,
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

        _, logit = simulate_pct_change_cancer(
            x_col_np=s["x_col"],
            p_default=p_def,
            theta_raw=theta_raw,
            NumIter=NumIter,
            w=w,
            dose=dose,
            interval=interval,
            temp=temp,
        )

        pred = predict_label_from_logit(float(logit))
        if pred == int(s["y"]):
            correct += 1
        total += 1

    return correct / max(total, 1)


# ============================================================
# 8) GD / SGD training loops
# ============================================================

def run_gd(
    train_samples,
    val_samples,      # kept in signature but unused inside (you validate later)
    p_default,
    NumIter=8400,
    w=0.01,
    dose=200.0,
    interval=21.0,
    temp=10.0,
    lr=1e-2,
    epochs=50,
    w_pos=2.333,
    w_neg=1.0,
    verbose=True,
):
    # init theta from defaults
    theta_raw = pack_from_default(p_default)

    # gradient of full-batch loss
    loss_grad = grad(lambda th: dataset_loss(
        th, train_samples, p_default,
        NumIter=NumIter, w=w, dose=dose, interval=interval, temp=temp,
        w_pos=w_pos, w_neg=w_neg
    ))

    history = {
        "train_loss": [],
        "train_acc": [],
    }

    for ep in range(1, epochs + 1):
        # forward pass on train
        train_loss = dataset_loss(
            theta_raw, train_samples, p_default,
            NumIter=NumIter, w=w, dose=dose, interval=interval, temp=temp,
            w_pos=w_pos, w_neg=w_neg
        )

        # gradient
        g = loss_grad(theta_raw)

        # update
        theta_raw = theta_raw - lr * g

        # train accuracy (optional but cheap-ish)
        train_acc = evaluate_accuracy(
            theta_raw, train_samples, p_default,
            NumIter=NumIter, w=w, dose=dose, interval=interval, temp=temp
        )

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))

        if verbose:
            print(f"[GD] Epoch {ep:03d} | "
                  f"train loss {float(train_loss):.4f} | "
                  f"train acc {train_acc:.3f}")

    return theta_raw, history



def run_sgd(
    train_samples,
    val_samples,      # kept for symmetry, not used here
    p_default,
    NumIter=8400,
    w=0.01,
    dose=200.0,
    interval=21.0,
    temp=10.0,
    lr=1e-2,
    epochs=50,
    batch_size=1,
    w_pos=2.333,
    w_neg=1.0,
    seed=0,
    verbose=True,
):
    rng = np.random.default_rng(seed)

    theta_raw = pack_from_default(p_default)

    # gradient of minibatch loss
    def minibatch_loss(th, batch):
        return dataset_loss(
            th, batch, p_default,
            NumIter=NumIter, w=w, dose=dose, interval=interval, temp=temp,
            w_pos=w_pos, w_neg=w_neg
        )

    mb_grad = grad(minibatch_loss)

    history = {
        "train_loss": [],
        "train_acc": [],
    }

    n = len(train_samples)

    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        shuffled = [train_samples[i] for i in idx]

        # iterate minibatches
        for start in range(0, n, batch_size):
            batch = shuffled[start:start + batch_size]
            g = mb_grad(theta_raw, batch)
            theta_raw = theta_raw - lr * g

        # end of epoch metrics on full train set
        train_loss = dataset_loss(
            theta_raw, train_samples, p_default,
            NumIter=NumIter, w=w, dose=dose, interval=interval, temp=temp,
            w_pos=w_pos, w_neg=w_neg
        )
        train_acc = evaluate_accuracy(
            theta_raw, train_samples, p_default,
            NumIter=NumIter, w=w, dose=dose, interval=interval, temp=temp
        )

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))

        if verbose:
            print(f"[SGD] Epoch {ep:03d} | "
                  f"train loss {float(train_loss):.4f} | "
                  f"train acc {train_acc:.3f}")

    return theta_raw, history


# ============================================================
# 9) Main script
# ============================================================

if __name__ == "__main__":

    # ---------- Paths ----------
    clinical_data_path = "data/nature_immune_processed"
    CSV_PATH = "data_preprocessing_notebooks/npy_dimensions_sorted.csv"
    slide_response_path = "data/nature_immune_processed/slide_responses.csv"

    # ---------- Hyperparams ----------
    TOP_TRAIN = 20
    NumIter = 10
    w = 0.01

    # Weighted BCE for 70% NR, 30% R
    # If y=1 is R, then:
    # pos_weight ~ N_neg / N_pos = 0.7 / 0.3 = 2.333
    w_pos = 2.333
    w_neg = 1.0

    # logit temperature
    temp = 10.0

    # Optimization settings
    mode = "gd"  # "gd" or "sgd"
    epochs = 30
    lr = 1e-2
    batch_size = 1

    # ---------- Defaults ----------
    p_default = Params(
        lambda_C=0.7, K_C=28, d_C=0.01, k_T=4, K_K=25, D_C=0.0005,
        lambda_T=0.05, K_T=10, K_R=10, d_T=0.01, k_A=0.16, K_A=100, D_T=0.1,
        d_A=0.0315, rows=1, cols=1
    )

    # ---------- Load file ordering ----------
    df_sorted, files, file_paths = load_sorted_files(CSV_PATH, clinical_data_path)

    # ---------- Load labels ----------
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

    # ---------- Split ----------
    train_samples = samples[:20]
    val_samples = samples[20:25]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    # ---------- Train ----------
    if mode == "gd":
        theta_raw_star, history = run_gd(
            train_samples, val_samples, p_default,
            NumIter=NumIter, w=w,
            temp=temp,
            lr=lr, epochs=epochs,
            w_pos=w_pos, w_neg=w_neg,
            verbose=True
        )
    else:
        theta_raw_star, history = run_sgd(
            train_samples, val_samples, p_default,
            NumIter=NumIter, w=w,
            temp=temp,
            lr=lr, epochs=epochs, batch_size=batch_size,
            w_pos=w_pos, w_neg=w_neg,
            seed=0,
            verbose=True
        )

    # ---------- Final metrics ----------
    train_acc = evaluate_accuracy(theta_raw_star, train_samples, p_default, NumIter=NumIter, w=w, temp=temp)
    val_acc = evaluate_accuracy(theta_raw_star, val_samples, p_default, NumIter=NumIter, w=w, temp=temp)

    print("\nFinal accuracy:")
    print(f"  Train acc: {train_acc:.3f}")
    print(f"  Val acc:   {val_acc:.3f}")

    final_val_loss = dataset_loss(
        theta_raw_star, val_samples, p_default,
        NumIter=NumIter, w=w, dose=200.0, interval=21.0, temp=temp,
        w_pos=w_pos, w_neg=w_neg
    )
    print(f"Final val loss: {float(final_val_loss):.4f}")

    # ---------- Report learned params ----------
    theta_pos = softplus(theta_raw_star)
    learned = {k: float(v) for k, v in zip(OPT_NAMES, theta_pos)}

    print("\nLearned parameters:")
    for k in OPT_NAMES:
        print(f"  {k}: {learned[k]:.6g}")

    # ---------- Save training curve ----------
    out_dir = "test_clinical_data_visualization_sgd"
    os.makedirs(out_dir, exist_ok=True)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    print(f"\nSaved training history to {os.path.join(out_dir, 'training_history.csv')}")
