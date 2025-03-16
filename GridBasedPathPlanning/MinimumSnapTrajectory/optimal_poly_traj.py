import sys
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import cvxpy as cp
from pathlib import Path
from MinimumSnapTrajectory.casadi_traj import CasadiTraj
from MinimumSnapTrajectory.differential_flatness import compute_quad_state_trajectory_casadi

# Polynomial trajectory optimization

def generate_trajectory(poly_deg, optim_order, constraints, continuity_order):
    n = poly_deg + 1  # no. of coefficients (polynomial degree + 1)
    m = len(constraints["pos"])  # no. of waypoints
    # Compute hessian of objective function
    t = ca.SX.sym('t')
    c = ca.SX.sym('c', n)
    basis = ca.vertcat(*[t**i for i in range(n)])
    basis_ders = [basis]
    for der in range(optim_order-1):
        basis_ders = basis_ders + [ca.jacobian(basis_ders[der], t)]
    poly = c.T @ basis

    poly_der_sqr = poly
    for _ in range(optim_order):
        poly_der_sqr = ca.gradient(poly_der_sqr, t)
    poly_der_sqr = poly_der_sqr**2

    poly_der_sqr_2 = ca.gradient(poly, t)**2
    c_poly_der_2 = ca.poly_coeff(poly_der_sqr_2, t)[::-1]
    c_poly_der_int_2 = ca.vertcat(0, ca.vertcat(*[c_poly_der_2[i] / (i + 1) for i in range(c_poly_der_2.shape[0])]))
    basis_int_2 = ca.vertcat(*[t ** i for i in range(c_poly_der_int_2.shape[0])])

    c_poly_der = ca.poly_coeff(poly_der_sqr, t)[::-1]
    c_poly_der_int = ca.vertcat(0, ca.vertcat(*[c_poly_der[i]/(i+1) for i in range(c_poly_der.shape[0])]))
    basis_int = ca.vertcat(*[t**i for i in range(c_poly_der_int.shape[0])])
    int_exp = c_poly_der_int.T @ basis_int  # + c_poly_der_int_2.T @ basis_int_2
    Q = ca.hessian(int_exp, c)[0]
    Q_f = ca.Function('Q', [t], [Q])
    basis_f = [ca.Function(f'basis{i}', [t], [basis_ders[i]]) for i in range(len(basis_ders))]

    # Put together the optimization
    x = cp.Variable((4*n, m-1))  # 3 position coordinates + yaw, m-1 sections
    Q_lst = [np.array(Q_f(con[0])) for con in constraints["pos"]]
    Q_obj_lst = [Q_lst[i+1] - Q_lst[i] for i in range(len(Q_lst)-1)]
    Q_blkdiag_lst = [np.kron(np.eye(4, dtype=int), Q) for Q in Q_obj_lst]
    Q_max = [np.max(Q_) for Q_ in Q_blkdiag_lst]
    # obj_lst = [cp.quad_form(x[:, i], cp.Parameter(shape=Q_blkdiag_lst[i].shape, value=Q_blkdiag_lst[i], PSD=True))
    #            for i in range(m-1)]
    # print([a/b for a, b in zip(Q_blkdiag_lst, Q_max)])
    obj_lst = [cp.quad_form(x[:, i], Q_blkdiag_lst[i]/Q_max[i])
               for i in range(m-1)]
    obj = cp.Minimize(cp.sum(cp.hstack(obj_lst)))
    const = []

    # continuity constraints
    for i in range(4):  # x, y, z, yaw
        for j in range(m-2):  # joints between sections
            for der in range(continuity_order):
                const += [x[i*n:(i+1)*n, j].T @ np.array(basis_f[der](constraints["pos"][j+1][0])).flatten() ==
                          x[i*n:(i+1)*n, j+1].T @ np.array(basis_f[der](constraints["pos"][j+1][0])).flatten()]

    # waypoint constraints
    for i in range(4):  # x, y, z, yaw
        for j in range(m-1):
            const += [x[i*n:(i+1)*n, j].T @ np.array(basis_f[0](pos[0])).flatten() == pos[i+1]
                      for pos in constraints["pos"][j:j+2]]
            const += [x[i*n:(i+1)*n, j].T @ np.array(basis_f[1](vel[0])).flatten() == vel[i+1]
                      for vel in constraints["vel"][j:j+2] if vel is not None]
            const += [x[i*n:(i+1)*n, j].T @ np.array(basis_f[2](acc[0])).flatten() == acc[i+1]
                      for acc in constraints["acc"][j:j+2] if acc is not None]

    prob = cp.Problem(obj, const)

    prob.solve(solver='MOSEK')

    return x.value

def eval_pw_poly(t, pw_poly, der):
    res = np.zeros_like(t)
    for i in range(len(t)):
        idx = (np.abs(T_arr - t[i])).argmin()
        if t[i] <= T_arr[idx] and i != 0:
            idx -= 1
        res[i] = pw_poly[idx].deriv(der)(t[i])
    return res


def getPathData(path):
        time, pos, vel, acc = [], [], [], []
        for i in range(len(path)):
            time.append(i)
            pos.append([i, path[i,0], path[i,1], path[i,2], 0])   # pos: [t, x, y, z]
            if i == 0 or i == path.shape[0]-1:
                vel.append([(i * 1/(path.shape[0]-1)), 0, 0, 0, 0])
            else:
                vel.append(None)
            acc.append(None)
        return time, pos, vel, acc

def plotGraph2D(x): #(t, poly):
        #show_der = 0
        plt.figure()
        #plt.plot(t, eval_pw_poly(t, poly[0], show_der))
        #plt.plot(t, eval_pw_poly(t, poly[1], show_der))
        #plt.plot(t, eval_pw_poly(t, poly[2], show_der))
        plt.plot(x[0], x[4], 'r')
        plt.plot(x[0], x[5], 'g')
        plt.plot(x[0], x[6], 'b')
        plt.show()

def plotGraph3D(path_extracted, path_corrected, x):
    # Plot the original points and the fitted curve
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path_extracted[:,0], path_extracted[:,1], path_extracted[:,2], 'ro', label='TJPS Extracted Path Points')
    ax.plot(path_corrected[:,0], path_corrected[:,1], path_corrected[:,2], 'go', label='TJPS Corrected Path Points')
    ax.plot(x[1], x[2], x[3], 'b-', label='Minimum Snap Trajectory')
    max = np.max(path_extracted,axis=0)
    ax.set_box_aspect(max)
    ax.legend()
    plt.show()

def getMinimumSnapTrajectory(path, poly_deg = 5, optim_order = 3):
    time, pos, vel, acc = getPathData(path)
    constraints = { "pos": pos, # syntax: [t, x, y, z, yaw]
                    "vel": vel, # syntax: [t, (x, y, z, yaw) or None] 
                    "acc": acc}

    coeff = generate_trajectory(poly_deg=poly_deg,
                                optim_order=optim_order,
                                constraints=constraints,
                                continuity_order=optim_order)
    
    poly = [None]*3
    poly[0] = [np.polynomial.Polynomial(coeff[0:poly_deg+1, j]) for j in range(len(time)-1)]
    poly[1] = [np.polynomial.Polynomial(coeff[poly_deg+1:2*(poly_deg+1), j]) for j in range(len(time)-1)]
    poly[2] = [np.polynomial.Polynomial(coeff[2*(poly_deg+1):3*(poly_deg+1), j]) for j in range(len(time)-1)]

    return time, coeff, poly

def getStateFromDifferentialFlatness(time, coeff, poly_deg):
    # Calculate motion state from the xyz polynomials from the flatness property of the drone
    
    ref = CasadiTraj()
    for d in range(ref.num_der): # 7
        ref.x[d] = ref.casadi_ppoly(v=ref.t, breakpoints=time, coefficients=coeff[0:poly_deg + 1, :].tolist(), der=d)
        ref.y[d] = ref.casadi_ppoly(v=ref.t, breakpoints=time, coefficients=coeff[poly_deg + 1:2 * (poly_deg + 1), :].tolist(), der=d)
        ref.z[d] = ref.casadi_ppoly(v=ref.t, breakpoints=time, coefficients=coeff[2 * (poly_deg + 1):3 * (poly_deg + 1), :].tolist(), der=d)
        ref.yaw[d]=ref.casadi_ppoly(v=ref.t, breakpoints=time, coefficients=coeff[3 * (poly_deg + 1):4 * (poly_deg + 1), :].tolist(), der=d)

    # x_f : x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz
    x_f, u_f = compute_quad_state_trajectory_casadi(ref, m=0.02, J=ca.diag(ca.vertcat(*[1.4e-5, 1.4e-5, 2.16e-5]))) # returns a function

    return x_f


if __name__ == "__main__":
    
    path_extracted = np.load('Data/path_extracted.npy')
    path_corrected = np.load('Data/path_corrected.npy')

    poly_deg = 4
    optim_order = 3
    time_ori, coeff, poly = getMinimumSnapTrajectory(path_corrected, poly_deg, optim_order)
    x_f = getStateFromDifferentialFlatness(time_ori, coeff, poly_deg)

    time_interp = np.linspace(0, time_ori[-1], 500)             # set interpolation timestep
    x = np.array(x_f(np.expand_dims(time_interp, 0)))           # shape: num_states * t.shape[0]
    x = np.vstack([np.expand_dims(np.array(time_interp), 0),x]) # x: t, x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz

    plotGraph3D(path_extracted, path_corrected, x)
    #plotGraph2D(x)