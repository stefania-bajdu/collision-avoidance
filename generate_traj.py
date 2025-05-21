import casadi
import numpy as np
import Bspline_conversionMatrix as BsplineM
from scipy.interpolate import BSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
import copy


def get_ref(psi, Tsim, dt):
    k = 8  # d = k + 1, k: polynomial degree of the spline
    n_ctrl_pts = 28  # number of control points
    knot = [0, Tsim]
    g = 9.81
    # psi = 0 * np.pi / 180
    knot = BsplineM.knot_vector(k, n_ctrl_pts, knot)
    tt = np.arange(min(knot), max(knot), dt)
    bs_list = BsplineM.b_spline_basis_functions(n_ctrl_pts, k, knot)

    # Conversion matrix M
    M = BsplineM.bsplineConversionMatrices(n_ctrl_pts, k, knot)

    # Waypoints
    # W = np.array([[0, 0.2, 0.5, 0.4, 0, -0.6, -0.6, -0.6, -0.6],
    #               [0, 0, 0, 0.6, 0.7, 0.7, 0.4, -0.3, -0.6],
    #               [0.35, 0.7, 0.9, 1.1, 1.1, 1.0, 0.9, 0.7, 0.25]  # 3D test
    #               ])
    # W = np.array([[0, 0.4, 0.5, 0.4, 0, -0.6, -0.6, -0.6, -0.6],
    #               [0, 0, 0, 0.6, 0.7, 0.7, 0.4, -0.3, -0.6],
    #               [0.1, 0.25, 0.7, 1.1, 1.1, 1.0, 0.9, 0.7, 0.25]  # 3D test
    #               ])
    W = np.array([[0, 0.3, 0.6, 0.6, 0.3, 0, -0.3, -0.3, 0],
                  [0, -0.3, 0, 0.3, 0.6, 0.6, 0.3, 0, 0],
                  [0.15, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.35, 0.2]  # 3D test
                  ])
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1])
    ctrl_pts_timestamps = np.linspace(min(knot), max(knot), n_ctrl_pts)

    ############################### Optimization problem construction ###############################
    solver = casadi.Opti()
    # Control point as optimization variable
    P = solver.variable(W.shape[0], n_ctrl_pts)

    # Objective function
    objective = 0
    P1 = casadi.mtimes(P, M[0])
    for i in range(n_ctrl_pts + 1):
        for j in range(n_ctrl_pts + 1):
            def f_lamb(t, it=i, jt=j): return bs_list[1][it](t) * bs_list[1][jt](t)
            buff_int = quad(f_lamb, min(knot), max(knot))[0]
            objective = objective + casadi.mtimes(casadi.transpose(casadi.mtimes(buff_int, P1[:, i])), P1[:, j])

    # Implementing waypoint constraints
    for i in range(W.shape[1]):
        tmp_bs = np.zeros((len(bs_list[0]), 1))
        for j in range(len(bs_list[0])):
            tmp_bs[j] = bs_list[0][j](waypoint_time_stamps[i])
        # Mathematically, casadi.mtimes(P, tmp_bs) = P * tmp_bs
        solver.subject_to(casadi.mtimes(P, tmp_bs) == W[:, i])

    # Final velocity is zero
    for i in range(W.shape[1]):
        tmp_bs = np.zeros((len(bs_list[1]), 1))
        for j in range(len(bs_list[1])):
            tmp_bs[j] = bs_list[1][j](waypoint_time_stamps[-1] - dt)
        solver.subject_to(casadi.mtimes(P1, tmp_bs) == 0)

    # Final acceleration is zero
    P2 = casadi.mtimes(P, M[1])
    for i in range(W.shape[1]):
        tmp_bs = np.zeros((len(bs_list[2]), 1))
        for j in range(len(bs_list[2])):
            tmp_bs[j] = bs_list[2][j](waypoint_time_stamps[-1] - dt)
        solver.subject_to(casadi.mtimes(P2, tmp_bs) == 0)

    solver.minimize(objective)

    solver_options = {'ipopt': {'print_level': 0, 'sb': 'yes'}, 'print_time': 0}
    solver.solver('ipopt', solver_options)
    # ============================================================================================
    print('Generating reference ...')
    sol = solver.solve()  # Solve for the control points
    # ============================================================================================
    # Construct the result curve
    P = sol.value(P)
    print('Optimal control-points found')
    # Compute the Bspline with the solution of P
    z = []
    for i in range(P.shape[0]):
        z.append(BSpline(knot, P[i], k))

    # First derivative of the flat output
    P1 = np.array(P * M[0])
    z_d = []
    for i in range(P1.shape[0]):
        z_d.append(BSpline(knot, P1[i], k - 1))

    # Second derivative of the flat output
    P2 = np.array(P * M[1])
    z_dd = []
    for i in range(P2.shape[0]):
        z_dd.append(BSpline(knot, P2[i], k - 2))

    x = z[0](tt)
    y = z[1](tt)
    z = z[2](tt)
    dx = z_d[0](tt)
    dy = z_d[1](tt)
    dz = z_d[2](tt)
    ddx = z_dd[0](tt)
    ddy = z_dd[1](tt)
    ddz = z_dd[2](tt)

    # ref = np.stack([x, y, z, dx, dy, dz, ddx, ddy, ddz])
    v_ref = np.stack([ddx, ddy, ddz])
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * np.sin(psi) - ddy * np.cos(psi)) / thrust)
    theta = np.arctan((ddx * np.cos(psi) + ddy * np.sin(psi)) / (ddz + g))

    # print('Reference information:')
    # print('Max Thrust: {txta}g (m/s^2), Min Thrust: {txtb}g (m/s^2)'.format(txta=round(max(thrust) / g, 2),
    #                                                                         txtb=round(min(thrust) / g, 2)))
    # print('Max Roll  : {txta}  (deg),   Min Roll  : {txtb}  (deg)'.format(txta=round(max(phi) * 180 / np.pi, 2),
    #                                                                       txtb=round(min(phi) * 180 / np.pi, 2)))
    # print('Max Pitch : {txta}  (deg),   Min Pitch : {txtb}  (deg)'.format(txta=round(max(theta) * 180 / np.pi, 2),
    #                                                                       txtb=round(min(theta) * 180 / np.pi, 2)))
    ref = {"trajectory": (np.round(np.stack([x, y, z, dx, dy, dz]), 3)).transpose(),
           "time_step": tt,
           "thrust": thrust,
           "phi": phi,
           "theta": theta,
           "Nsim": tt.shape[0],
           "v_ref": v_ref.transpose()}

    return ref


def get_ref_setpoints(psi, Tsim, dt, version=1):
    knot = [0, Tsim]
    g = 9.81
    tt = np.arange(min(knot), max(knot), dt)
    if version == 1:
        W = np.array([[0, 0.3, 0.6, 0.6, 0.3, 0, -0.3, -0.3, 0],
                      [0, -0.3, 0, 0.3, 0.6, 0.6, 0.3, 0, 0],
                      [0.35, 0.4, 0.75, 0.8, 0.8, 0.8, 0.8, 0.5, 0.35]  # 3D test
                      ])
    elif version == 2:
        W = np.array([[0, 0.6, 0.3, -0.3, 0],
                      [0, 0, 0.6, 0.3, 0],
                      [0.35, 0.75, 0.8, 0.8, 0.35]  # 3D test
                      ])
    elif version == 3:
        W = np.array([[0.6, 0.6],
                      [0.6, 0.6],
                      [0.8, 0.8]  # 3D test
                      ])
    elif version == 7:
        W = np.array([[0, 0],
                      [0, 0],
                      [0.8, 0.8]  # 3D test
                      ])
    k_pass = 1
    ref_tmp = np.empty((0, 3))
    waypoint_time_stamps = np.linspace(min(knot), max(knot), W.shape[1] + 1)
    for i_tmp in range(waypoint_time_stamps.shape[0] - 1):
        cur = np.array(W[:, i_tmp])
        while dt * k_pass <= waypoint_time_stamps[i_tmp + 1]:
            ref_tmp = np.vstack((ref_tmp, cur))
            k_pass = k_pass + 1

    ref_full = np.block([
        [ref_tmp, ref_tmp * 0]
    ])
    v_ref = 0 * ref_tmp.transpose()
    ddx, ddy, ddz = v_ref[0, :], v_ref[1, :], v_ref[2, :]
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * np.sin(psi) - ddy * np.cos(psi)) / thrust)
    theta = np.arctan((ddx * np.cos(psi) + ddy * np.sin(psi)) / (ddz + g))
    ref = {
        "trajectory": ref_full,
        "time_step": tt,
        "thrust": thrust,
        "phi": phi,
        "theta": theta,
        "Nsim": tt.shape[0],
        "v_ref": v_ref.transpose()}

    return ref


rref = get_ref(0, 30, 0.1)

# # rref2 = copy.deepcopy(rref)
# # rref2["trajectory"] = np.copy(rref["trajectory"])
# # rref2["trajectory"][:, 0] += 1
# # rref2["trajectory"][:, 1] += 1
# # rref2["trajectory"][:, 2] += 1

# # fig2 = plt.figure()
# # for i in range(3):
# #     ax2 = fig2.add_subplot(3, 1, i + 1)
# #     it = i + 0
# #     ax2.plot(rref["time_step"], rref['trajectory'][:, it])
# #     ax2.grid(True)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection="3d")

# ax.plot(rref["trajectory"][:, 0], rref["trajectory"][:, 1], rref["trajectory"][:, 2], label="Trajectory 1")
# # ax.plot(rref2["trajectory"][:, 0], rref2["trajectory"][:, 1], rref2["trajectory"][:, 2], label="Trajectory 2")

# # # Labels and legend
# # ax.set_xlabel("X")
# # ax.set_ylabel("Y")
# # ax.set_zlabel("Z")
# # ax.legend()

# plt.show()
