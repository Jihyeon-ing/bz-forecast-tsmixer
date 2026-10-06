import numpy as np

def burton_dst(
        time,
        v_sw,
        bz,
        n_sw,
        dst0=0.0
):
    """
    Burton-type Dst model
    dDst*/dt = Q(t) - Dst*/tau
    :param time: time array in hours
    :param v_sw: solar wind speed [km/s]
    :param bz: IMF Bz [nT]
    :param n_sw: proton density [cm^-3]
    :param dst0: Initial Dst* value [nT]
    :return:
        dst: Modeled dst [nT]
    """

    v_sw = np.asarray(v_sw, dtype=float)
    bz = np.asarray(bz, dtype=float)
    n_sw = np.asarray(n_sw, dtype=float)
    dst0 = np.asarray(dst0, dtype=float)

    b = 15.8
    c = 20.0
    tau = 7.7
    Ec = 0.5

    # P = 1e-2 * n_sw * v_sw**2
    P = 1.67276e-6 * n_sw * v_sw ** 2
    Bs = np.maximum(-bz, 0.0)
    Ey = 1e-3 * v_sw * Bs
    Q = np.where(Ey > Ec, -4.4 * (Ey-Ec), 0.0)

    dst = np.zeros_like(time)
    dst_star = np.zeros_like(time)

    dst[0] = dst0
    dst_star[0] = dst0 - b * np.sqrt(P[0]) + c

    for i in range(1, len(time)):
        dt = time[i] - time[i-1]

        dst_star[i] = dst_star[i-1] + dt * (Q[i-1] - dst_star[i-1] / tau)

        dst[i] = dst_star[i] + b * np.sqrt(P[i]) - c

    return dst

def obrien_mcpherron_dst(time, v_sw, n_sw, bz, dst0=0.0, prediction=True):
    v_sw = np.asarray(v_sw, dtype=float)
    bz = np.asarray(bz, dtype=float)
    n_sw = np.asarray(n_sw, dtype=float)
    dst0 = np.asarray(dst0, dtype=float)

    Ec = 0.49
    b = 7.26
    c = 11.0

    P = 1.67276e-6 * n_sw * v_sw ** 2
    Bs = np.maximum(-bz, 0.0)
    Ey = 1e-3 * v_sw * Bs
    Q = np.where(Ey > Ec, -4.4 * (Ey - Ec), 0.0)
    tau = 2.4 * np.exp(9.74 / (4.69 + Ey))

    dst = np.zeros_like(time, dtype=float)
    dst_star = np.zeros_like(time, dtype=float)

    if prediction:
        dst[0] = dst0
        dst_star[0] = dst0 - b * np.sqrt(P[0]) + c

        for i in range(1, len(time)):
            dt = time[i] - time[i-1]

            dst_star[i] = dst_star[i-1] + dt * (Q[i-1] - dst_star[i-1] / tau[i-1])
            dst[i] = dst_star[i] + b * np.sqrt(P[i]) - c

    else:
        dst_obs = np.asarray(dst0, dtype=float)
        dst[0] = dst_obs[0]
        dst_star[0] = dst_obs[0] - b * np.sqrt(P[0]) + c

        for i in range(1, len(time)):
            dt = time[i] - time[i-1]
            dst_star_prev = dst_obs[i-1] - b * np.sqrt(P[i-1]) + c
            dst_star_i = dst_star_prev + dt * (Q[i-1] - dst_star_prev / tau[i-1])
            dst_star[i] = dst_star_i
            dst[i] = dst_star_i + b * np.sqrt(P[i]) - c

    return dst

if __name__ == "__main__":
    time = np.arange(0, 24, 1)
    v = np.full_like(time, 339)
    n = np.full_like(time, 38)
    bz = np.full_like(time, -10)
    dst = burton_dst(time, v, bz, n)
    print(dst)