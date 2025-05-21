import numpy as np
import casadi as cas
from scipy.linalg import expm
import time
from scipy.io import loadmat
from utils import *
from att_utils import *
from generate_traj import *
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Bspline_conversionMatrix import *

log_file = "./log.log"
file = open(log_file, "w")


class IMPC:
    def __init__(self, pos_init):
        self.Na = len(pos_init)
        self.psi_ref = {i: 0 for i in range(self.Na)}
        self.pos_ref = {i: 0 for i in range(self.Na)}
        self.vref = {i: 0 for i in range(self.Na)}

        data_Upos = np.load('./Upos.npy', allow_pickle=True).tolist()
        self.Vc = {}
        self.Vc['A_vc'] = np.round(data_Upos['A'][0, 0], 5)
        self.Vc['b_vc'] = np.round(data_Upos['b'][0, 0], 5)

        self.Tfin = 30
        self.Ts = 0.1
        self.t = np.arange(0, self.Tfin, self.Ts)

        self.pos_ref[0] = np.tile([4.0, 1.0, 0, 0, 0, 0], (len(self.t), 1))
        self.psi_ref[0] = np.zeros(len(self.t))

        self.initialize_parameters()
        self.initialize_states(pos_init)
        self.setup_solver()

    def initialize_parameters(self):
        self.g = 9.81
        self.m = 28e-3
        self.I_BF = np.diag([1.4, 1.4, 2.2]) * 1e-5

        self.Npred = 4

        self.A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 6))]])
        self.B = np.block([[np.zeros((3, 3))], [np.eye(3)]])
        self.A_d = np.eye(6) + self.Ts * self.A
        self.B_d = self.B * self.Ts

        self.Q = np.diag([10, 10, 10, 5, 5, 5])
        self.R = np.diag([1, 1, 1])

        self.r_min = 0.35
        self.THETA = 5 * np.eye(3)
        self.THETA_1 = np.linalg.inv(self.THETA)
        self.THETA_2 = self.THETA_1 @ self.THETA_1

    def initialize_states(self, pos_init):
        self.state_xi = {i: np.zeros((6, len(self.t))) for i in range(self.Na)}
        self.predicted_evolution = {i: np.zeros((6, self.Npred)) for i in range(self.Na)}
        self.state_eta = {i: np.zeros((6, len(self.t))) for i in range(self.Na)}
        self.angle_refs = {i: np.zeros((3, len(self.t))) for i in range(self.Na)}
        self.thrusts = {i: np.zeros(len(self.t)) for i in range(self.Na)}
        self.angular_speed_ref = {i: np.zeros((3, len(self.t))) for i in range(self.Na)}
        self.vsim = {i: np.zeros((3, len(self.t))) for i in range(self.Na)}
        self.e_int = {i: np.zeros((3, 1)) for i in range(self.Na)}
        self.formation_error = {i: np.zeros((3, len(self.t))) for i in range(self.Na)}

        self.acc = {i: np.zeros((3, len(self.t))) for i in range(self.Na)}
        self.ang_acc = {i: np.zeros((3, len(self.t))) for i in range(self.Na)}
        self.sigma = {i: np.zeros((3, len(self.t))) for i in range(self.Na)}

        vel_init = np.zeros(3)
        for i in range(self.Na):
            self.state_xi[i][:, 0] = np.hstack((pos_init[i, :], vel_init))

    def W(self, eta):
        phi, theta, _ = eta
        return np.array([[1, 0, -np.sin(theta)],
                         [0, np.cos(phi), np.sin(phi) * np.cos(theta)],
                         [0, -np.sin(phi), np.cos(phi) * np.cos(theta)]])

    def W_dot(self, eta, rates):
        phi, theta, _ = eta
        p, q, _ = rates
        return np.array([[1, 0, -np.cos(theta) * q],
                        [0, -np.sin(phi) * p, np.cos(phi) * p * np.cos(theta) - np.sin(phi) * np.sin(theta) * q],
                        [0, -np.cos(phi) * p, -np.sin(phi) * p * np.cos(theta) + np.cos(phi) * np.sin(theta) * q]])

    def eta_ddot(self, x, tau):
        eta, omega = x[6:9], x[9:12]
        W_eta = self.W(eta)
        W_dot_eta = self.W_dot(eta, omega)
        term1 = np.linalg.inv(self.I_BF @ W_eta) @ (tau - self.I_BF @ W_dot_eta @ omega - np.cross(W_eta @ omega, self.I_BF @ W_eta @ omega))
        return term1

    def eps_ddot(self, x, u):
        term1 = np.array([0, 0, -self.g])

        term2 = (1 / self.m) * np.array([
            np.cos(u[1]) * np.sin(u[2]) * np.cos(x[8]) + np.sin(u[1]) * np.sin(x[8]),
            np.cos(u[1]) * np.sin(u[2]) * np.sin(x[8]) - np.sin(u[1]) * np.cos(x[8]),
            np.cos(u[1]) * np.cos(u[2])
        ]) * u[0]

        return term1 + term2

    def setup_solver(self):
        n, du = 6, 3

        self.spline_degree = 3
        self.n_ctrl_pts = self.Npred + self.spline_degree - 1
        self.knot_vec = knot_vector(self.spline_degree, self.n_ctrl_pts, [0, self.Ts * self.Npred])
        self.basis_funcs = b_spline_basis_functions(self.n_ctrl_pts, self.spline_degree, self.knot_vec)

        solver = cas.Opti()
        x = solver.variable(n, self.Npred + 1)
        P_i = solver.variable(3, self.n_ctrl_pts)
        v = solver.variable(du, self.Npred)

        # xinit = solver.parameter(n, 1)
        vinit = solver.parameter(du, 1)
        xref_param = solver.parameter(n, self.Npred)
        psi_ref_param = solver.parameter(1, 1)

        # Set constraints
        # solver.subject_to(x[:, 0] == xinit)
        for k in range(self.Npred):
            # solver.subject_to(x[:, k+1] == self.A_d @ x[:, k] + self.B_d @ v[:, k])
            solver.subject_to(cas.mtimes(self.Vc['A_vc'], v[:, k]) <= self.Vc['b_vc'])

            solver.subject_to(x[0:3, k] == P_i * self.basis_funcs[k])
            solver.subject_to(x[3:6, k] == self.A_d @ x[:, k] + self.B_d @ v[:, k])
            solver.subject_to(v[:, k] == self.A_d @ x[:, k] + self.B_d @ v[:, k])

            # solver.subject_to(cas.mtimes(Acoll, x[:3, coll_step + 1]) + cas.mtimes(dij_at_coll, phi_i) / 2 <= bcoll)
            # solver.subject_to(phi_i <= cas.MX.zeros(self.Na, 1))

        objective = 0
        for k in range(self.Npred):
            state_error = x[:, k] - xref_param[:, k]
            # control_effort = v[:, k] - (v[:, k-1] if k > 0 else vinit)
            control_effort = v[:, k]
            objective += cas.mtimes([state_error.T, self.Q[:, :], state_error]) + cas.mtimes([control_effort.T, self.R, control_effort])

        solver.minimize(objective)
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}
        solver.solver('ipopt', opts)

        self.x = x
        self.v = v
        self.solver = solver
        # self.xinit = xinit
        self.vinit = vinit
        self.xref_param = xref_param
        self.psi_ref_param = psi_ref_param
        self.objective = objective

    def run(self):
        self.solver.set_value(self.vinit, np.zeros(3))
        start_time = time.time()

        for i in range(len(self.t) - 1):
            print(f"[ITERATION {i}]\n")
            tic = time.time()
            for id in range(self.Na):
                self.compute_control(i, id=id)
                self.update_states(i, id=id)
            file.write(f"[TIME] {time.time() - tic}\n")

        elapsed_time = time.time() - start_time

        print(f"Elapsed time for a {len(self.t)} simulation horizon is {elapsed_time}.")
        comment = 'less' if elapsed_time / len(self.t) < self.Ts else 'more'
        print(f"One control loop lasts {elapsed_time / len(self.t)} and is {comment} than the sample time {self.Ts}.")

    def compute_control(self, i, id=0):

        if i + self.Npred <= len(self.t):
            desired_pos = self.pos_ref[id][i:i+self.Npred, 0:3].T
            ref = np.vstack([desired_pos, self.pos_ref[id][i:i+self.Npred, 3:6].T])
        else:
            desired_pos = self.pos_ref[id][-self.Npred:, 0:3].T
            ref = np.vstack([desired_pos, self.pos_ref[id][-self.Npred:, 3:6].T])

        self.solver.set_value(self.xref_param, ref)
        self.solver.set_value(self.xinit, self.state_xi[id][:, i])
        self.solver.set_value(self.psi_ref_param, self.psi_ref[id][i])

        sol = self.solver.solve()
        vopt = sol.value(self.v)

        self.predicted_evolution[id] = sol.value(self.x)

        self.vsim[id][:, i] = vopt[:, 0]

    def update_states(self, i, id=0):
        v_ap = self.vsim[id][:, i]

        T, phi_d, theta_d = fl_laws(v_ap, self.state_eta[id][2, i])

        self.thrusts[id][i] = T
        self.angle_refs[id][:, i] = np.hstack([phi_d, theta_d, self.psi_ref[id][i]])

        if i == 0:
            self.angular_speed_ref[id][:, i] = 0
        else:
            self.angular_speed_ref[id][:, i] = (self.angle_refs[id][:, i] - self.angle_refs[id][:, i-1]) / self.Ts

        [tau, sigma] = run_att_PD(self.state_eta[id][:, i], self.angle_refs[id][:, i], self.angular_speed_ref[id][:, i], 0)
        ang_acc = self.eta_ddot(np.hstack((self.state_xi[id][:, i], self.state_eta[id][:, i])), tau)
        acc = self.eps_ddot(np.hstack((self.state_xi[id][:, i], self.state_eta[id][:, i])), np.array([T, phi_d, theta_d]))

        self.acc[id][:, i] = acc
        self.ang_acc[id][:, i] = ang_acc
        self.sigma[id][:, i] = sigma

        # self.state_eta[id][:, i+1] = self.A_d @ self.state_eta[id][:, i] + self.B_d @ sigma

        self.state_eta[id][3:6, i+1] = self.state_eta[id][3:6, i] + ang_acc * self.Ts
        self.state_eta[id][0:3, i+1] = self.state_eta[id][0:3, i] + self.state_eta[id][3:6, i] * self.Ts

        # self.state_xi[id][:, i + 1] = self.A_d @ self.state_xi[id][:, i] + self.B_d @ self.vsim[id][:, i]

        self.state_xi[id][3:6, i+1] = self.state_xi[id][3:6, i] + acc * self.Ts
        self.state_xi[id][0:3, i+1] = self.state_xi[id][0:3, i] + self.state_xi[id][3:6, i] * self.Ts


if __name__ == "__main__":

    # pos_init = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, -1, 0])])
    # pos_init = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0])])
    # pos_init = np.array([np.array([1, 1, 0]), np.array([4, 1, 0]), np.array([2, 2, 0]), np.array([4, 4, 0])])
    pos_init = np.array([np.array([1, 1, 0])])

    controller = IMPC(pos_init)
    controller.run()

    plot_positions(controller.t, controller.state_xi[0], controller.pos_ref[0], id=0)
    plot_velocities(controller.t, controller.state_xi[0], controller.pos_ref[0][:, 3:6], id=0)

    # plot_positions(controller.t, controller.state_xi[1], controller.pos_ref[1], id=1)

    plot_all_agent_distances(controller.t, controller.state_xi, d0=controller.r_min)
    # plot_velocities(controller.t, controller.state_xi[0], controller.pos_ref[0][:, 3:6], id=0)
    # plot_angles(controller.t, controller.state_eta[0], controller.angle_refs[0], 0.25, id=0)
    plot_real_u(controller.t, controller.thrusts[0], controller.angle_refs[0], id=0)
    # plot_real_u(controller.t, controller.thrusts[1], controller.angle_refs[1], id=1)
    # plot_real_u(controller.t, controller.thrusts[2], controller.angle_refs[2], id=2)
    # plot_real_u(controller.t, controller.thrusts[3], controller.angle_refs[3], id=3)

    # plot_formation_error(controller.t, controller.formation_error, controller.Na)

    plot_traj_animated(controller.t, controller.state_xi, controller.pos_ref)
    plt.show()

    file.close()
