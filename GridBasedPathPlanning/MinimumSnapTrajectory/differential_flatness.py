import pickle
import numpy as np
import scipy.interpolate as si
from scipy.spatial.transform import Rotation
import casadi as ca
import spatial_casadi as sca
from MinimumSnapTrajectory.casadi_traj import CasadiTraj


def my_dot(a: np.ndarray, b: np.ndarray):
    return np.expand_dims(np.sum(a*b, axis=1), axis=1)


def my_cross(a: np.ndarray, b: np.ndarray):
    return np.vstack([np.cross(a[i, :], b[i, :]) for i in range(a.shape[0])])

def compute_quad_state_trajectory_casadi(ref: CasadiTraj, m=0.605, g=9.81,
                                         J=ca.diag(ca.vertcat(1.5e-3, 1.45e-3, 2.66e-3))):
    # ref.t: casadi symbolic time
    x = ref.x  # derivatives of payload position
    y = ref.y  # derivatives of payload position
    z = ref.z  # derivatives of payload position
    yaw = ref.yaw  # derivatives of yaw angle
    r = [ca.vertcat(x_, y_, z_) for x_, y_, z_ in zip(x, y, z)]

    F = 3 * [ref.t]
    e3 = ca.vertcat(0, 0, 1)
    FRe3 = m*(r[2] + g * e3)
    F[0] = ca.norm_2(FRe3)
    Re3 = FRe3 / F[0]
    c1 = ca.vertcat(ca.cos(yaw[0]), ca.sin(yaw[0]), 0)
    Re2 = ca.cross(Re3, c1) / ca.norm_2(ca.cross(Re3, c1))
    Re1 = ca.cross(Re2, Re3)
    R = ca.horzcat(Re1, Re2, Re3)

    hw = m / F[0] * (r[3] - ca.dot(Re3, r[3]) * Re3)
    wx = -ca.dot(hw, Re2)
    wy = ca.dot(hw, Re1)
    wz = yaw[1] * ca.dot(e3, Re3)
    w = ca.vertcat(wx, wy, wz)

    F[1] = m * (ca.dot(r[3], Re3))
    F[2] = m * ca.dot(r[4], Re3) - 2 * F[1] * ca.dot(ca.cross(w, Re3), Re3) - \
           F[0] * ca.dot(ca.cross(w, ca.cross(w, Re3)), Re3)

    hdw = 1 / F[0] * (m * r[4] - F[2] * Re3 - 2 * F[1] * ca.cross(w, Re3)) - ca.cross(w, ca.cross(w, Re3))
    dw = ca.vertcat(-ca.dot(hdw, Re2), ca.dot(hdw, Re1), yaw[2] * ca.dot(e3, Re3))

    # dw = ca.jacobian(w, ref.t)
    tau = J @ dw + ca.cross(w, J @ w)

    eul = sca.Rotation.from_matrix(R).as_euler('xyz')

    states = ca.vertcat(r[0], r[1], eul, w)
    inputs = ca.vertcat(F[0], tau)
    states = ca.Function('x', [ref.t], [states])
    inputs = ca.Function('u', [ref.t], [inputs])
    return states, inputs

def compute_state_trajectory_casadi(ref: CasadiTraj, m=0.605, payload_mass=0.01, L=0.4, g=9.81,
                                    J=ca.diag(ca.vertcat(1.5e-3, 1.45e-3, 2.66e-3))):
    # ref.t: casadi symbolic time
    xL = ref.x  # derivatives of payload position
    yL = ref.y  # derivatives of payload position
    zL = ref.z  # derivatives of payload position
    yaw = ref.yaw  # derivatives of yaw angle
    mL = payload_mass
    rL = [ca.vertcat(x, y, z) for x, y, z in zip(xL, yL, zL)]
    # for dim in range(3):
    #     rL[dim][6] = [0 * elem for elem in rL[dim][5]]  # sixth derivative is zero
    # yaw = [np.expand_dims(np.array(yaw_), axis=1) for yaw_ in yaw]  # convert to numpy arrays
    # rL = [np.asarray(x).T for x in zip(*rL)]  # convert to numpy arrays

    q = 5 * [ref.t]
    T = 5 * [ref.t]
    e3 = ca.vertcat(0, 0, 1)
    q[0] = -mL * (rL[2] + g * e3)
    T[0] = ca.norm_2(q[0])
    q[0] = q[0] / T[0]
    T[1] = -mL * (ca.dot(q[0], rL[3]))
    q[1] = -(mL * rL[3] + T[1] * q[0]) / T[0]
    T[2] = -mL * (ca.dot(q[0], rL[4]) + ca.dot(q[1], rL[3]))
    q[2] = -(mL * rL[4] + 2 * T[1] * q[1] + T[2] * q[0]) / T[0]
    T[3] = -mL * (ca.dot(q[0], rL[5]) + 2 * ca.dot(q[1], rL[4]) + ca.dot(q[2], rL[3]))
    q[3] = -(mL * rL[5] + 3 * T[1] * q[2] + 3 * T[2] * q[1] + T[3] * q[0]) / T[0]
    T[4] = -mL * (ca.dot(q[0], rL[6]) + 3 * ca.dot(q[1], rL[5]) + 3 * ca.dot(q[2], rL[4]) + ca.dot(q[3], rL[3]))
    q[4] = -(mL * rL[6] + 4 * T[1] * q[3] + 6 * T[2] * q[2] + 4 * T[3] * q[1] + T[4] * q[0]) / T[0]

    r = rL.copy()
    for der in range(5):
        r[der] = rL[der] - L * q[der]

    F = 3 * [ref.t]
    F[0] = ca.norm_2(m * (r[2] + g * e3) + mL * (rL[2] + g * e3))

    Re3 = (m * (r[2] + g * e3) + mL * (rL[2] + g * e3)) / F[0]
    Re2 = ca.cross(Re3, ca.vertcat(ca.cos(yaw[0]), ca.sin(yaw[0]), 0*yaw[0]))
    Re2 = Re2 / ca.norm_2(Re2)
    Re1 = ca.cross(Re2, Re3)

    F[1] = ca.dot(m * r[3] + mL * rL[3], Re3)
    h_w = ((m * r[3] + mL * rL[3]) - F[1] * Re3) / F[0]
    w = ca.vertcat(-ca.dot(h_w, Re2), ca.dot(h_w, Re1), yaw[1] * Re3[2])

    temp = ca.cross(w, Re3)
    F[2] = ca.dot(m * r[4] + mL * rL[4], Re3) - F[1] * ca.dot(ca.cross(w, temp), Re3)
    h_dw = 1 / F[0] * (m * r[4] + mL * rL[4]) - ca.cross(w, ca.cross(w, Re3)) - 2 * F[1] / F[0] * ca.cross(w, Re3) - \
           1 / F[0] * F[2] * Re3
    dw = ca.vertcat(-ca.dot(h_dw, Re2), ca.dot(h_dw, Re1), yaw[2] * Re3[2])

    tau = J @ dw + ca.cross(w, J @ w)

    beta = ca.arcsin(-q[0][0])
    alpha = ca.arcsin(q[0][1]/ca.cos(beta))
    dbeta = -q[1][0]/ca.cos(beta)
    dalpha = (q[1][1] + ca.sin(alpha)*ca.sin(beta)*dbeta) / ca.cos(alpha) / ca.cos(beta)

    R = ca.horzcat(Re1, Re2, Re3)
    eul = sca.Rotation.from_matrix(R).as_euler('xyz')
    # eul = ca.vertcat(0, 0, 0)
    # eul = np.asarray([Rotation.from_matrix(np.vstack((Re1[i, :], Re2[i, :], Re3[i, :])).T).as_euler('xyz')
    #                   for i in range(Re1.shape[0])])
    pole_ang = ca.vertcat(alpha, beta, dalpha, dbeta)
    states = ca.vertcat(r[0], r[1], eul, w, pole_ang)
    inputs = ca.vertcat(F[0], tau)
    states = ca.Function('x', [ref.t], [states])
    inputs = ca.Function('u', [ref.t], [inputs])
    return states, inputs


def compute_state_trajectory_from_splines(spl, m, hook_mass, payload_mass, L, g, J, dt):
    x_spl = spl[0]
    y_spl = spl[1]
    z_spl = spl[2]
    yaw_spl = spl[3]
    xL = 7 * [[]]  # derivatives of payload position
    yL = 7 * [[]]  # derivatives of payload position
    zL = 7 * [[]]  # derivatives of payload position
    yaw = 3 * [[]]  # derivatives of yaw angle
    t = []
    mL = []
    for phase in range(6):
        for der in range(6):
            t_cur = np.arange(0, x_spl[phase][0][-1], dt)
            if len(t) < 1:
                t = t_cur.tolist()
            else:
                t = t + [t_ + t[-1] for t_ in t_cur]
            xL[der] = xL[der] + si.splev(t_cur, x_spl[phase], der=der).tolist()
            yL[der] = yL[der] + si.splev(t_cur, y_spl[phase], der=der).tolist()
            zL[der] = zL[der] + si.splev(t_cur, z_spl[phase], der=der).tolist()
            if der < 3:
                yaw[der] = yaw[der] + si.splev(t_cur, yaw_spl[phase], der=der).tolist()

        mass = hook_mass if phase in [0, 1, 5] else hook_mass + payload_mass
        mL = mL + [0 * elem + mass for elem in t_cur]
    mL = np.expand_dims(np.array(mL), axis=1)
    rL = [xL, yL, zL]
    for dim in range(3):
        rL[dim][6] = [0 * elem for elem in rL[dim][5]]  # sixth derivative is zero
    yaw = [np.expand_dims(np.array(yaw_), axis=1) for yaw_ in yaw]  # convert to numpy arrays
    rL = [np.asarray(x).T for x in zip(*rL)]  # convert to numpy arrays

    q = 5 * [np.array([])]
    T = 5 * [np.array([])]
    e3 = np.array([[0, 0, 1]])
    q[0] = -mL * (rL[2] + g * e3)
    T[0] = np.expand_dims(np.linalg.norm(q[0], axis=1), 1)
    q[0] = q[0] / T[0]
    T[1] = -mL * (my_dot(q[0], rL[3]))
    q[1] = -(mL * rL[3] + T[1] * q[0]) / T[0]
    T[2] = -mL * (my_dot(q[0], rL[4]) + my_dot(q[1], rL[3]))
    q[2] = -(mL * rL[4] + 2 * T[1] * q[1] + T[2] * q[0]) / T[0]
    T[3] = -mL * (my_dot(q[0], rL[5]) + 2 * my_dot(q[1], rL[4]) + my_dot(q[2], rL[3]))
    q[3] = -(mL * rL[5] + 3 * T[1] * q[2] + 3 * T[2] * q[1] + T[3] * q[0]) / T[0]
    T[4] = -mL * (my_dot(q[0], rL[6]) + 3 * my_dot(q[1], rL[5]) + 3 * my_dot(q[2], rL[4]) + my_dot(q[3], rL[3]))
    q[4] = -(mL * rL[6] + 4 * T[1] * q[3] + 6 * T[2] * q[2] + 4 * T[3] * q[1] + T[4] * q[0]) / T[0]

    r = rL.copy()
    for der in range(5):
        r[der] = rL[der] - L * q[der]

    F = 3 * [np.array([])]
    F[0] = np.expand_dims(np.linalg.norm(m * (r[2] + g * e3) + mL * (rL[2] + g * e3), axis=1), 1)

    Re3 = (m * (r[2] + g * e3) + mL * (rL[2] + g * e3)) / F[0]
    Re2 = my_cross(Re3, np.hstack((np.cos(yaw[0]), np.sin(yaw[0]), np.zeros_like(yaw[0]))))
    Re2 = Re2 / np.expand_dims(np.linalg.norm(Re2, axis=1), 1)
    Re1 = my_cross(Re2, Re3)

    F[1] = my_dot(m * r[3] + mL * rL[3], Re3)
    h_w = ((m * r[3] + mL * rL[3]) - F[1] * Re3) / F[0]
    w = np.hstack((-my_dot(h_w, Re2), my_dot(h_w, Re1), yaw[1] * np.expand_dims(Re3[:, 2], axis=1)))

    temp = my_cross(w, Re3)
    F[2] = my_dot(m * r[4] + mL * rL[4], Re3) - F[1] * my_dot(my_cross(w, temp), Re3)
    h_dw = 1 / F[0] * (m * r[4] + mL * rL[4]) - my_cross(w, my_cross(w, Re3)) - 2 * F[1] / F[0] * my_cross(w, Re3) - \
           1 / F[0] * F[2] * Re3
    dw = np.hstack((-my_dot(h_dw, Re2), my_dot(h_dw, Re1), yaw[2] * np.expand_dims(Re3[:, 2], axis=1)))

    tau = dw * J + my_cross(w, w * J)

    beta = np.arcsin(-q[0][:, 0])
    alpha = np.arcsin(q[0][:, 1]/np.cos(beta))
    dbeta = -q[1][:, 0]/np.cos(beta)
    dalpha = (q[1][:, 1] + np.sin(alpha)*np.sin(beta)*dbeta) / np.cos(alpha) / np.cos(beta)

    eul = np.asarray([Rotation.from_matrix(np.vstack((Re1[i, :], Re2[i, :], Re3[i, :])).T).as_euler('xyz')
                      for i in range(Re1.shape[0])])
    pole_ang = np.vstack((alpha, beta, dalpha, dbeta)).T
    states = np.hstack((r[0], r[1], eul, w, pole_ang))
    inputs = np.hstack((F[0], tau))
    return states, inputs, mL[:, 0]


if __name__ == "__main__":
    m = 0.605
    payload_mass = 0.1
    hook_mass = 0.01
    g = 9.81
    L = 0.4
    J = np.array([1.5e-3, 1.45e-3, 2.66e-3])

    with open("../../pickle/hook_up_spline_13_50_48.pickle", 'rb') as fp:
        spl = pickle.load(fp)
    compute_state_trajectory_from_splines(spl, m, hook_mass, payload_mass, L, g, J)













    # x_points = []
    # x_der = []
    # t = []
    # for phase in range(len(x_spl)):
    #     t_cur = np.arange(0, x_spl[phase][0][-1], 0.01)
    #     if len(t) < 1:
    #         t = t_cur.tolist()
    #     else:
    #         t = t + [t_ + t[-1] for t_ in t_cur]
    #     x_points = x_points + si.splev(t_cur, x_spl[phase], der=0).tolist()
    #     x_der = x_der + si.splev(t_cur, x_spl[phase], der=2).tolist()
    #
    # spl_fit = si.splrep(t, x_points, k=5, s=1e-2)
    # x_points_fit = si.splev(t, spl_fit, der=2)
    #
    # plt.figure()
    # plt.plot(t, x_der, t, x_points_fit)
    # plt.show()
