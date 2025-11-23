
def eval_f_output(x_col):
    """
    Given flattened state vector x_col shaped (N,1) with channel order [C, T, A],
    return total cancer burden and total T-cell count.

    x_col = [C_1, T_1, A_1, C_2, T_2, A_2, ..., C_n, T_n, A_n]^T
    """

    assert x_col.ndim == 2 and x_col.shape[1] == 1, "x_col must be (N, 1)"
    N = x_col.shape[0]
    
    # Ensure N is divisible by 3 (C, T, A per pixel)
    assert N % 3 == 0, "Length of x_col must be divisible by 3"

    # Extract flattened channels
    C_vals = x_col[0::3, 0]   # C at indices 0,3,6,...
    T_vals = x_col[1::3, 0]   # T at indices 1,4,7,...

    total_cancer_burden = float(C_vals.sum())
    total_tcell_count   = float(T_vals.sum())

    return total_cancer_burden, total_tcell_count