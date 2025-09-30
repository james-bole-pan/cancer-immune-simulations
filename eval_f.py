import autograd.numpy as anp

class Params:
    def __init__(self, lambda_C, K_C, d_C, k_T, K_K, D_C,
                       lambda_T, K_R, d_T, k_A, K_A, D_T,
                       d_A, rows, cols, dxFD=None):
        self.lambda_C = lambda_C; self.K_C = K_C; self.d_C = d_C
        self.k_T = k_T; self.K_K = K_K; self.D_C = D_C
        self.lambda_T = lambda_T; self.K_R = K_R; self.d_T = d_T
        self.k_A = k_A; self.K_A = K_A; self.D_T = D_T
        self.d_A = d_A
        self.rows = rows; self.cols = cols
        self.dxFD = dxFD  # optional

    def tuple(self):
        return (self.lambda_C, self.K_C, self.d_C, self.k_T, self.K_K, self.D_C,
                self.lambda_T, self.K_R, self.d_T, self.k_A, self.K_A, self.D_T,
                self.d_A, self.rows, self.cols)

def evalf_autograd(x_col, p: Params, r_A):
    """
    x_col: (N, 1) column vector with N = rows*cols*3, storing [C, T, A] per grid cell in row-major order.
    returns: (N, 1) column vector of time-derivatives in the same layout.
    """
    (lambda_C, K_C, d_C, k_T, K_K, D_C,
     lambda_T, K_R, d_T, k_A, K_A, D_T,
     d_A, rows, cols) = p.tuple()

    # ---- reshape (N,1) -> (rows, cols, 3) ----
    x_flat = anp.ravel(x_col)                          # (N,)
    N = x_flat.size
    d = 3
    assert N == rows * cols * d, "State length must be rows*cols*3"
    x = anp.reshape(x_flat, (rows, cols, d))           # (rows, cols, 3) as [C,T,A]

    def laplacian4(X, i, j, comp):
        """4-neighbor (unnormalized) Laplacian: sum(U,D,L,R) - n_neighbors*center."""
        s = anp.array(0.0)
        nnb = 0
        if i-1 >= 0:   s = s + X[i-1, j, comp]; nnb += 1
        if i+1 < rows: s = s + X[i+1, j, comp]; nnb += 1
        if j-1 >= 0:   s = s + X[i, j-1, comp]; nnb += 1
        if j+1 < cols: s = s + X[i, j+1, comp]; nnb += 1
        return s - nnb * X[i, j, comp]

    eps = 1e-12
    f_rows = []
    for i in range(rows):
        row_vals = []
        for j in range(cols):
            C = x[i, j, 0]
            T = x[i, j, 1]
            A = x[i, j, 2]

            # Tumor: logistic - baseline death - saturating kill + diffusion
            dC = (lambda_C * C * (1.0 - C / (K_C + eps))
                  - d_C * C
                  - (k_T * C * T) / (C + K_K + eps)
                  + D_C * laplacian4(x, i, j, 0))

            # CD8: recruitment - decay + drug-boosted survival/prolif + diffusion
            drug_boost = (k_A * A) / (A + K_A + eps)  # per-day per-T
            dT = (lambda_T * (C / (C + K_R + eps))
                  - d_T * T
                  + drug_boost * T
                  + D_T * laplacian4(x, i, j, 1))

            # Drug: systemic PK (no diffusion here)
            dAdt = r_A - d_A * A

            row_vals.append(anp.stack([dC, dT, dAdt]))
        f_rows.append(anp.stack(row_vals))

    f = anp.stack(f_rows)              # (rows, cols, 3)
    f_flat = anp.ravel(f)              # (N,)
    return f_flat.reshape((-1, 1))     # (N,1) column vector
