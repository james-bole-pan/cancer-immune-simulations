import numpy as np
from eval_f import eval_f, Params

# function to save parameters to a .npz file
def save_params_npz(filename, params):
    np.savez(
        filename,
        **{
            k: np.array(v) if not isinstance(v, np.ndarray) else v
            for k, v in params.__dict__.items()
        }
    )

# function to load parameters from a .npz file
def load_params_npz(filename):
    data = np.load(filename, allow_pickle=True)

    # Extract each required parameter from the file
    p = Params(
        lambda_C = data["lambda_C"].item(),
        K_C      = data["K_C"].item(),
        d_C      = data["d_C"].item(),
        k_T      = data["k_T"].item(),
        K_K      = data["K_K"].item(),
        D_C      = data["D_C"].item(),
        lambda_T = data["lambda_T"].item(),
        K_T      = data["K_T"].item(),
        K_R      = data["K_R"].item(),
        d_T      = data["d_T"].item(),
        k_A      = data["k_A"].item(),
        K_A      = data["K_A"].item(),
        D_T      = data["D_T"].item(),
        d_A      = data["d_A"].item(),
        rows     = data["rows"].item(),
        cols     = data["cols"].item(),
        dxFD     = data["dxFD"].item() if "dxFD" in data else None
    )

    return p