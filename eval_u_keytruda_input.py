import numpy as np

def eval_u_keytruda_input(w, dose=200.0, interval=21.0, t_end=365.0):
    # Precompute nominal dose times (0, 21, 42, 63, ...)
    dose_times = np.arange(0.0, t_end + interval, interval)

    # Keep track of which dose index was last given
    state = {"last_k": None}

    # Tolerance: a couple of timesteps wide
    tol = max(2 * w, 1e-3)

    def r(t):
        # Which nominal dose index k does this time t correspond to?
        # e.g., t ~ 21 → k = 1, t ~ 42 → k = 2
        k = int(round(t / interval))

        # If k is outside our planned dose range, no dose
        if k < 0 or k >= len(dose_times):
            return 0.0

        t_target = dose_times[k]

        # Check if we're close enough to this scheduled dose time
        if abs(t - t_target) < tol and state["last_k"] != k:
            # Fire ONCE for this k
            state["last_k"] = k
            return dose
        else:
            return 0.0

    return r
