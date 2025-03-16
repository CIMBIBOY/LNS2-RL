import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


class CasadiTraj:
    def __init__(self):
        self.t = ca.MX.sym("t")
        self.num_der = 7
        self.x, self.y, self.z, self.yaw = self.define_traj()

    def define_traj(self, *args, **kwargs):
        x = self.num_der * [None]
        y = self.num_der * [None]
        z = self.num_der * [None]
        yaw = self.num_der * [None]
        return x, y, z, yaw

    @staticmethod
    def casadi_ppoly(v, breakpoints, coefficients, der, shift=False):
        bp = ca.MX(ca.DM(breakpoints))
        y = ca.MX(ca.DM(coefficients))
        n = y.shape[0]
        L = ca.low(bp, v)
        co = y[:, L]
        if shift:
            res = ca.dot(co, (v-bp[L]) ** ca.DM(range(n)))
        else:
            res = ca.dot(co, v ** ca.DM(range(n)))
        for _ in range(der):
            res = ca.gradient(res, v)
        return res


class ZeroTraj(CasadiTraj):
    def __init__(self):
        super().__init__()
        self.segment_times = np.array([5, 10, 15, 20, 25])

    def define_traj(self):
        x_f = 0 * ca.sin(self.t)
        y_f = 0 * ca.cos(self.t)
        z_f = 0.5 + 0 * self.t
        yaw_f = 0 * ca.sin(self.t)
        x = self.num_der * [self.t]
        x[0] = x_f
        y = self.num_der * [self.t]
        y[0] = y_f
        z = self.num_der * [self.t]
        z[0] = z_f
        yaw = self.num_der * [self.t]
        yaw[0] = yaw_f
        for d in range(1, self.num_der):
            x[d] = ca.gradient(x[d - 1], self.t)
            y[d] = ca.gradient(y[d - 1], self.t)
            z[d] = ca.gradient(z[d - 1], self.t)
            yaw[d] = ca.gradient(yaw[d - 1], self.t)
        return x, y, z, yaw


class SpiralTraj(CasadiTraj):
    def __init__(self):
        super().__init__()

    def define_traj(self):
        x_f = 0.5 * ca.sin(self.t)
        y_f = 0.5 * ca.cos(self.t)
        z_f = 0.5 + 0.05 * self.t
        yaw_f = 0.7 * ca.sin(self.t)
        x = self.num_der * [self.t]
        x[0] = x_f
        y = self.num_der * [self.t]
        y[0] = y_f
        z = self.num_der * [self.t]
        z[0] = z_f
        yaw = self.num_der * [self.t]
        yaw[0] = yaw_f
        for d in range(1, self.num_der):
            x[d] = ca.gradient(x[d - 1], self.t)
            y[d] = ca.gradient(y[d - 1], self.t)
            z[d] = ca.gradient(z[d - 1], self.t)
            yaw[d] = ca.gradient(yaw[d - 1], self.t)
        return x, y, z, yaw


class LissajousTraj(CasadiTraj):
    def __init__(self):
        super().__init__()

    def define_traj(self):
        self.freq = 1.
        x_f = 0.5 * ca.sin(self.freq * self.t)
        y_f = 0.6 * ca.sin(2 * self.freq * (self.t - 3.141592 / 2))
        z_f = 0.7 + 0. * self.t
        yaw_f = 0.7 * ca.sin(self.t)
        x = self.num_der * [self.t]
        x[0] = x_f
        y = self.num_der * [self.t]
        y[0] = y_f
        z = self.num_der * [self.t]
        z[0] = z_f
        yaw = self.num_der * [self.t]
        yaw[0] = yaw_f
        for d in range(1, self.num_der):
            x[d] = ca.gradient(x[d - 1], self.t)
            y[d] = ca.gradient(y[d - 1], self.t)
            z[d] = ca.gradient(z[d - 1], self.t)
            yaw[d] = ca.gradient(yaw[d - 1], self.t)
        return x, y, z, yaw


class OptimalPolyTraj(CasadiTraj):
    def __init__(self):
        self.segment_times = np.zeros(5)
        super().__init__()

    def define_traj(self, drone_init_pos=np.array([-1.5, -1, 1.1]), drone_init_yaw=np.pi / 6,
                    load_init_pos=np.array([0, 0, 0.25]), load_init_yaw=0,
                    load_target_pos=np.array([0.5, -1.2, 0.25]), load_target_yaw=np.pi/3, grasp_speed=0.3):
        constraints = self.get_hook_waypoints(drone_init_pos, drone_init_yaw, load_init_pos, load_init_yaw,
                                              load_target_pos, load_target_yaw, grasp_speed)
        poly_deg = 7
        optim_order = 6  # minimum pop

        from quadcopter_hook_twodof.planning.optimal_poly_traj import generate_trajectory
        coeff = generate_trajectory(poly_deg=poly_deg, optim_order=optim_order, constraints=constraints,
                                    continuity_order=3)

        T_lst = [pos[0] for pos in constraints["pos"]]
        T_arr = np.array(T_lst)

        x = self.num_der * [self.t]
        y = self.num_der * [self.t]
        z = self.num_der * [self.t]
        yaw = self.num_der * [self.t]
        for d in range(self.num_der):
            x[d] = self.casadi_ppoly(v=self.t, breakpoints=T_lst,
                                     coefficients=coeff[0:poly_deg + 1, :].tolist(), der=d)
            y[d] = self.casadi_ppoly(v=self.t, breakpoints=T_lst,
                                     coefficients=coeff[poly_deg + 1:2 * (poly_deg + 1), :].tolist(), der=d)
            z[d] = self.casadi_ppoly(v=self.t, breakpoints=T_lst,
                                     coefficients=coeff[2 * (poly_deg + 1):3 * (poly_deg + 1), :].tolist(), der=d)
            yaw[d] = self.casadi_ppoly(v=self.t, breakpoints=T_lst,
                                       coefficients=coeff[3 * (poly_deg + 1):4 * (poly_deg + 1), :].tolist(), der=d)

        plot = True
        if plot:
            from quadcopter_hook_twodof.planning.traj_opt_min_time import plot_3d_trajectory
            t_num = np.linspace(0, T_arr[-1], 500)
            x_num = ca.Function('x', [self.t], [x[0]])(t_num)
            y_num = ca.Function('y', [self.t], [y[0]])(t_num)
            z_num = ca.Function('z', [self.t], [z[0]])(t_num)
            v_num = np.linalg.norm(np.hstack((np.array(ca.Function('vx', [self.t], [x[1]])(t_num)),
                                              np.array(ca.Function('vy', [self.t], [y[1]])(t_num)),
                                              np.array(ca.Function('vz', [self.t], [z[1]])(t_num)))), axis=1)
            v_num[0] = 0
            plot_3d_trajectory(np.array(x_num), np.array(y_num), np.array(z_num), v_num, "", 0)
        self.segment_times = T_arr
        return x, y, z, yaw

    @staticmethod
    def get_hook_waypoints(drone_init_pos, drone_init_yaw, load_init_pos, load_init_yaw, load_target_pos,
                           load_target_yaw, grasp_speed):
        init_pos_drone = drone_init_pos  # Drone position after takeoff
        init_yaw_drone = drone_init_yaw  # Drone yaw at takeoff
        init_pos_load = load_init_pos  # Load initial position, only z coordinate is offset
        init_yaw_load = load_init_yaw  # Load initial orientation
        target_pos_load = load_target_pos  # Load target position
        target_yaw_load = load_target_yaw  # Load target yaw

        while init_yaw_load - init_yaw_drone > np.pi:
            init_yaw_load -= 2 * np.pi
        while init_yaw_load - init_yaw_drone < -np.pi:
            init_yaw_load += 2 * np.pi
        while target_yaw_load - init_yaw_load > np.pi:
            target_yaw_load -= 2 * np.pi
        while target_yaw_load - init_yaw_load < -np.pi:
            target_yaw_load += 2 * np.pi

        # Rotation matrices
        R_attach = np.array([[np.cos(init_yaw_load), -np.sin(init_yaw_load), 0],
                             [np.sin(init_yaw_load), np.cos(init_yaw_load), 0], [0, 0, 1]])
        R_detach = np.array([[np.cos(target_yaw_load), -np.sin(target_yaw_load), 0],
                             [np.sin(target_yaw_load), np.cos(target_yaw_load), 0], [0, 0, 1]])

        t_scale = 3
        T1 = t_scale * np.linalg.norm(init_pos_load - init_pos_drone)
        T2 = t_scale * np.linalg.norm(target_pos_load + np.array([0, 0, 0.45]) - init_pos_load)
        T3 = 2 * t_scale * 0.45
        T4 = 3 * t_scale * 0.3

        print(f"Duration of trajectory: {T1 + T2 + T3 + T4}")

        T5 = 20

        v_grasp = grasp_speed

        constraints = {"pos": [[0] + (init_pos_drone - np.array([0, 0, 0])).tolist() + [init_yaw_drone],
                               [T1] + (init_pos_load - np.array([0, 0, 0])).tolist() + [init_yaw_load],
                               [T1 + T2] + (target_pos_load + np.array([0, 0, 0.45])).tolist() + [target_yaw_load],
                               [T1 + T2 + T3] + (target_pos_load - np.array([0, 0, 0])).tolist() + [target_yaw_load],
                               [T1 + T2 + T3 + T4] + (target_pos_load - np.array([0, 0, 0]) + R_detach @ np.array(
                                   [-0.3, 0, 0])).tolist() + [target_yaw_load],
                               [T1 + T2 + T3 + T4 + T5] + (target_pos_load - np.array([0, 0, 0]) + R_detach @ np.array(
                                   [-0.3, 0, 0])).tolist() + [target_yaw_load]
                               ],  # syntax: [t, x, y, z]
                       "vel": [[0, 0, 0, 0, 0],
                               [T1] + (R_attach @ np.array([v_grasp, 0, 0])).tolist() + [0],
                               [T1 + T2, 0, 0, 0, 0],
                               [T1 + T2 + T3, 0, 0, 0, 0],
                               [T1 + T2 + T3 + T4, 0, 0, 0, 0],
                               [T1 + T2 + T3 + T4 + T5, 0, 0, 0, 0]
                               ],
                       "acc": [[0, 0, 0, 0, 0],
                               [T1, 0, 0, 0, 0],
                               [T1 + T2, 0, 0, 0, 0],
                               [T1 + T2 + T3, 0, 0, 0, 0],
                               [T1 + T2 + T3 + T4, 0, 0, 0, 0],
                               [T1 + T2 + T3 + T4 + T5, 0, 0, 0, 0]
                               ]}
        return constraints
