"""
Trajectory generations as convex optimizations.
Also trajectory evaluations.
"""
import cvxpy as cp
import numpy as np


def path_jerk(knots):
    """
    Variables x = (s_0, v_0, a_0, j_0, j_1, ..., j_(n - 1)),
    where s, v, a, and j stand for position, velocity, acceleration,
    and jerk respectively.

    Parameters
    ----------
    tf (float): time duration
    n (int): the number segment. It will have (n + 1) knots.

    Returns
    -------
    list[Expression]: CVXPY expressions for physical quantities at knots,
                      and the jerks on segments.
    """
    dts = knots[1:] - knots[:-1]
    n = knots.shape[0] - 1

    x = cp.Variable(n + 3)
    jerks = x[3:]

    a_mat, v_mat, s_mat = np.zeros((3, n + 1, n + 3))

    s_mat[0, 0] = v_mat[0, 1] =  a_mat[0, 2] = 1

    # For the end of each segment,
    for i, dt in enumerate(dts):
        # a_(i + 1) = a_i + j_i * dt,
        # where j_i = x[i + 3].
        a_mat[i + 1] += a_mat[i]
        a_mat[i + 1, i + 3] += dt

        v_mat[i + 1] += v_mat[i]
        v_mat[i + 1] += a_mat[i] * dt
        v_mat[i + 1, i + 3] += dt**2 / 2

        # s_i+1 = s_i + v_i * dt + a_i * dt**2 / 2 +
        #         j_i * dt**3 / 6.
        s_mat[i + 1] += s_mat[i]
        s_mat[i + 1] += v_mat[i] * dt
        s_mat[i + 1] += a_mat[i] * dt**2 / 2
        s_mat[i + 1, i + 3] += dt**3 / 6

    accs = a_mat @ x
    vels = v_mat @ x
    ss = s_mat @ x

    return ss, vels, accs, jerks


def path(knots):
    """
    For variables x = (s_0, v_0, a_0, a_1, ..., a_(n-1)),
    where s_0 and v_0 are the position and the velocity at the beginning of the first knot.

    Parameters
    ----------
    tf (float): the time duration.
    n (int): the number of the equally spaced segments.

    Returns
    -------
    list[Expression]: physical quantities at the ends of knots.
    """
    dts = knots[1:] - knots[:-1]
    n = knots.shape[0] - 1

    x = cp.Variable(n + 2)
    accs = x[2:]

    # Total n + 1 knots from 0, 1, 2, ..., to n.
    v_mat = np.zeros((n + 1, n + 2))
    s_mat = np.zeros((n + 1, n + 2))

    # Initial values.
    s_mat[0, 0] = 1
    v_mat[0, 1] = 1

    for i, dt in enumerate(dts):
        # Note that v_{i + 1} = v_i + a_i * dt,
        # where a_i = x[i + 2].
        v_mat[i + 1] += v_mat[i]
        v_mat[i + 1, i + 2] += dt

        # Note that s_{i + 1} = s_i + v_i * dt + (a_i / 2) * dt**2,
        # where a_i = x[i + 2].
        s_mat[i + 1] += s_mat[i]
        s_mat[i + 1] += v_mat[i] * dt
        s_mat[i + 1, i + 2] += dt**2 / 2

    vels = v_mat @ x
    ss = s_mat @ x

    return ss, vels, accs


def pos(t, ts, ss, vels, accs, jerks=None):
    """
    Parameters
    ----------
    t (float): The time.
    ts (ndarray): (n + 1,)
    ss (ndarray): (n + 1,)
    vels (ndarray): (n + 1,)
    accs (ndarray): (n,)
    """
    idx = np.digitize(t, ts)

    if idx >= ts.shape[0]:
        # Assumes that the acceleration vanishes after t_f.
        return ss.value[-1] + (t - ts[-1]) * vels.value[-1]

    # Note that ts[idx - 1] <= t < ts[idx].
    # v = vels.value[idx - 2] at t = ts[idx - 1].
    # And it accelerate for dt with a rate of accs.value[idx - 1].
    dt = t - ts[idx - 1]

    s, v = (ss.value[idx - 1], vels.value[idx - 1])
    a = accs.value[idx - 1]
    j = jerks.value[idx - 1] if jerks else 0.

    return s + v * dt + a * dt**2 / 2 + j * dt**3 / 6


def vel(t, ts, ss, vels, accs, jerks=None):
    idx = np.digitize(t, ts)

    if idx >= ts.shape[0]:
        return vels.value[-1]

    dt = t - ts[idx - 1]
    v = vels.value[idx - 1]
    a = accs.value[idx - 1]
    j = jerks.value[idx - 1] if jerks else 0.

    return v + a * dt + j * dt**2 / 2


def acc(t, ts, ss, vels, accs, jerks=None):
    idx = np.digitize(t, ts)

    if idx >= ts.shape[0]:
        return 0.

    dt = t - ts[idx - 1]
    a = accs.value[idx - 1]
    j = jerks.value[idx - 1] if jerks else 0.

    return a + j * dt
