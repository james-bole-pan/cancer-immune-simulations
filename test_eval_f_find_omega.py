# test_eval_f_find_omega.py
"""
Explore how different omega (time step) values affect convergence
for logistic tumor growth on a 3x3 grid using SimpleSolver + eval_f.

We compare the numerical total tumor count across the grid against
the analytical logistic solution summed over all 9 cells.

We want to find the omega that would give us numerical instability, 
as well as the omega that gives us a good balance of accuracy (given
the required accuracy for the clinical setting) and efficiency.
"""

import numpy as np
import copy
import math
import matplotlib.pyplot as plt

from SimpleSolver import SimpleSolver
from eval_f import eval_f, Params


def analytical_logistic_total(t, C0, lam, K, n_cells):
    """
    Analytical total tumor count across n_cells, each following:
      C(t) = K / (1 + ((K - C0)/C0) * exp(-lam * t))
    """
    C_single = K / (1.0 + ((K - C0) / C0) * np.exp(-lam * t))
    return n_cells * C_single


def run_case_for_omega(omega, total_days=84.0, visualize=False):
    """
    Run SimpleSolver with step size omega and return trajectory errors:
      - relative L2 error over the whole trajectory
      - final-value absolute and relative errors
    """
    # ----- Model setup: logistic growth only -----
    rows, cols = 3, 3
    n_cells = rows * cols

    # Params: only logistic tumor growth active
    p = Params(
        lambda_C=0.33, K_C=28.0, d_C=0.0, k_T=0.0, K_K=1.0, D_C=0.0,
        lambda_T=0.0, K_R=1.0, d_T=0.0, k_A=0.0, K_A=1.0, D_T=0.0,
        d_A=0.0, rows=rows, cols=cols
    )

    # Initial condition: same per cell; T=A=0
    C0 = 0.5
    x0 = np.zeros((n_cells * 3, 1))
    x0[0::3, 0] = C0
    x0[1::3, 0] = 0.0
    x0[2::3, 0] = 0.0

    # Zero input
    u_func = lambda t: 0.0

    # Time grid
    NumIter = int(round(total_days / omega))
    total_days = NumIter * omega  # align with integer number of steps

    # Run solver
    X, t = SimpleSolver(
        eval_f,
        x_start=x0,
        p=p,
        eval_u=u_func,
        NumIter=NumIter,
        w=omega,
        visualize=visualize,
        gif_file_name=f"test_evalf_output_figures/logistic_3x3_omega_{omega}.gif" if visualize else "State_visualization.gif",
    )

    # Numerical total tumor count across grid
    C_num_total = np.sum(X[0::3, :], axis=0)

    # Analytical total
    C_anal_total = analytical_logistic_total(
        t=np.asarray(t),
        C0=C0,
        lam=p.lambda_C,
        K=p.K_C,
        n_cells=n_cells
    )

    # Errors
    rel_L2 = np.linalg.norm(C_num_total - C_anal_total) / (np.linalg.norm(C_anal_total) + 1e-12)
    abs_final = float(abs(C_num_total[-1] - C_anal_total[-1]))
    rel_final = float(abs_final / (abs(C_anal_total[-1]) + 1e-12))

    return {
        "omega": omega,
        "NumIter": NumIter,
        "rel_L2": float(rel_L2),
        "abs_final": abs_final,
        "rel_final": rel_final,
        "t": t,
        "C_num_total": C_num_total,
        "C_anal_total": C_anal_total,
    }


def main():
    omegas = [7.0, 6.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.01]
    total_days = 84
    results = []

    print("Exploring convergence vs omega for 3x3 logistic growth...\n")
    print(f"{'omega':>8} | {'steps':>6} | {'rel_L2':>10} | {'rel_final':>10} | {'abs_final':>10}")
    print("-" * 56)

    for w in omegas:
        try:
            res = run_case_for_omega(w, total_days=total_days, visualize=False)
            results.append(res)
            print(f"{res['omega']:8.2g} | {res['NumIter']:6d} | {res['rel_L2']:10.3e} "
                  f"| {res['rel_final']:10.3e} | {res['abs_final']:10.3e}")
        except Exception as e:
            print(f"{w:8.2g} | {'ERR':>6} | {'-':>10} | {'-':>10} | {'-':>10}   ({e})")

    # --- Plot all trajectories together ---
    plt.figure(figsize=(8, 5))
    for res in results:
        t = res["t"]
        C_num = res["C_num_total"]
        plt.plot(t, C_num, label=f"Numerical (ω={res['omega']})")

    # One analytical curve for reference (they are all identical)
    t_ref = results[-1]["t"]
    C_anal_ref = results[-1]["C_anal_total"]
    plt.plot(t_ref, C_anal_ref, "k--", linewidth=2, label="Analytical")

    plt.xlabel("Time")
    plt.ylabel("Total tumor (9 cells)")
    plt.title(f"3x3 Logistic Growth: Numerical vs Analytical (varied ω); total_days={total_days}")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig("test_evalf_output_figures/logistic_3x3_different_omega_trajectory_plot.png")
    plt.show()

    # --- Error convergence plot ---
    ws = np.array([r["omega"] for r in results if np.isfinite(r["rel_L2"])])
    errs = np.array([r["rel_L2"] for r in results if np.isfinite(r["rel_L2"])])
    if len(ws) > 0:
        plt.figure(figsize=(6, 4))
        plt.loglog(ws, errs, marker="o")
        plt.gca().invert_xaxis()
        plt.xlabel("omega (time step)")
        plt.ylabel("Relative L2 error (trajectory)")
        plt.title(f"Convergence vs omega (3x3 logistic); total_days={total_days}")
        plt.grid(True, which="both", alpha=0.4)
        plt.tight_layout()
        plt.savefig("test_evalf_output_figures/logistic_3x3_different_omega_error_plot.png")
        plt.show()

if __name__ == "__main__":
    main()