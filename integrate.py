import numpy as np
import scipy as sp


def integrate_eom(f, y0, knots, t_eval, events=None):
    """
    Integrates the equation of motions through knots.
    t_eval are not necessarily in knots.

    Parameters
    ----------
    f (Callable): the dynamics. f(t, y) = dy /dt.
    knots (ndarray): Knots of reference trajectory.
    t_eval (ndarray): the evaluation time points.
    """
    atol = 1e-9
    t_eval_i = 0

    xs, ys = [], []

    last_x = knots[0]
    # The initial state.
    last_y = y0

    # Integrate the equations of motion piecewisely (segment by segment).
    for a, b in zip(knots[:-1], knots[1:]):
        te = []
        incl_b_flag = False

        # Collect only relevant t_eval points for this segment.
        while True:
            if np.isclose(t_eval[t_eval_i], a, atol=atol):
                te.append(a)
                t_eval_i += 1
                continue

            if np.isclose(t_eval[t_eval_i], b, atol=atol):
                te.append(b)
                t_eval_i += 1
                incl_b_flag = True
                break
        
            if t_eval[t_eval_i] > b:
                break

            te.append(t_eval[t_eval_i])
            t_eval_i += 1

        # b is added to `te` for the initial condition for the next segment.
        # If b is close to one of evaluation points, `incl_b_flag` must be set.
        if not incl_b_flag:
            te.append(b)

        traj = sp.integrate.solve_ivp(f, (a, b), last_y, t_eval=te, events=events)

        if traj.status == 0:
            if incl_b_flag:
                xs.append(traj.t)
                ys.append(traj.y)
            elif traj.t is not None and traj.t.shape[0] > 1:
                xs.append(traj.t[:-1])
                ys.append(traj.y[:, :-1])

            last_x = traj.t[-1]
            last_y = traj.y[:, -1]
        elif traj.status == 1:
            if (isinstance(traj.t, list) and traj.t) or (isinstance(traj.t, np.ndarray) and traj.t.shape[0] > 0):
                xs.append(traj.t)
                ys.append(traj.y)

                last_x = traj.t[-1]
                last_y = traj.y[:, -1]

            traj = sp.integrate.solve_ivp(f, (last_x, b), last_y, events=events)
            xs.append(traj.t[-1:])
            ys.append(traj.y[:, -1:])
            print(f'terminal @ {xs[-1][-1] / 1e-3:.3f} (ms).')
            break

    xs = np.concatenate(xs)
    ys = np.concatenate(ys, axis=1)

    return xs, ys
