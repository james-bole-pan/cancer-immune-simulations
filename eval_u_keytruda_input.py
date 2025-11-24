import numpy as np

def eval_u_keytruda_input(dose=200.0, interval=21.0):
    def r(t):
        return dose if np.isclose(t % interval, 0.0, atol=1e-6) else 0.0
    return r