import numpy as np
from newtonNd import newtonNd
from trapezoidal import f_trap

def trapezoidal_adapt(eval_f, x_start, p, eval_u, t_start, t_stop, dt,
                      tol_f=5.0,
                      dt_min=1e-5,
                      dt_max=1.0,
                      errf=1e-9,
                      errDeltax=1e-9,
                      relDeltax=1e-9,
                      MaxIter=20):

    # Set starting values
    x = np.asarray(x_start).ravel().copy()
    t = t_start
    dt = dt

    Ts = [t]
    Xs = [x.copy()]

    while t < t_stop:

        # Avoid overshoot
        dt = min(dt, t_stop - t)

        # Evaluate f at current point
        u_n = eval_u(t)
        f_n = np.asarray(eval_f(x, p, u_n)).ravel()

        # Forward Euler prediction with current dt
        x_FE = x + dt * f_n

        # Evaluate slope at predicted state
        u_np1_predict = eval_u(t + dt)
        f_pred = np.asarray(eval_f(x_FE, p, u_np1_predict)).ravel()

        # Relative slope based on ratio between next slope and current slope
        speed_ratio = np.linalg.norm(f_pred, ord=np.inf) / (np.linalg.norm(f_n, ord=np.inf) + 1e-14)

        # adjust dt
        dt_new = dt
        if speed_ratio > tol_f:
            dt_new = max(dt_min, 0.75 * dt)
        elif speed_ratio < 1.0 / tol_f:
            dt_new = min(dt_max, 2 * dt)

        # if dt changes a TON then redo the predictor step
        if abs(dt_new - dt) / dt > 0.2:
            dt = dt_new
            # repeat predictor at new dt
            x_FE = x + dt * f_n
            u_np1_predict = eval_u(t + dt)
            f_pred = np.asarray(eval_f(x_FE, p, u_np1_predict)).ravel()
        else:
            dt = dt_new

        # Now run trapezoidal with new dt
        u_np1 = eval_u(t + dt)
        f_l = f_n  # consistent with trapezoidal

        params = {
            "x_n": x,
            "f_l": f_l,
            "dt": dt,
            "p": p,
            "u_np1": u_np1,
            "eval_f": eval_f
        }

        x0_guess = x_FE  # better predition that x0

        x_next, converged, *_ = newtonNd(
            f_trap,
            x0_guess,
            params,
            u_np1,
            errf, errDeltax, relDeltax,
            MaxIter,
            visualize=False,
            FiniteDifference=1,
            Jfhand=None
        )

        # If newton doesnt converge, reduce dt and try again
        if not converged:
            dt = max(dt_min, dt * 0.25)
            continue

        # Increment time
        t += dt
        x = x_next.ravel()

        Ts.append(t)
        Xs.append(x.copy())

    return np.array(Xs).T, np.array(Ts)
