import numpy as np
import casadi as cas
from scipy.linalg import expm
import time
from scipy.io import loadmat
from utils import *
from att_utils import *
from generate_traj import *

log_file = "./log.log"
file = open(log_file, "w")


class IMPC:
    def __init__(self, ref, pos_init):
        self.Na = len(pos_init)
        self.psi_ref = {i: 0 for i in range(self.Na)}
        self.pos_ref = {i: 0 for i in range(1)}
        self.vref = {i: 0 for i in range(self.Na)}

        self.formation_offsets = generate_formation_offsets(self.Na, d=0.3)
        # self.formation_offsets = np.array([[0, 0, 0] for _ in range(self.Na)])

        data_Upos = np.load('./Upos.npy', allow_pickle=True).tolist()
        self.Vc = {}
        self.Vc['A_vc'] = np.round(data_Upos['A'][0, 0], 5)
        self.Vc['b_vc'] = np.round(data_Upos['b'][0, 0], 5)

        self.load_trajectory(ref, id=0)

        self.initialize_parameters()  # add Q and R here?
        self.initialize_states(pos_init)
        self.setup_solver()

    def load_trajectory(self, ref, id=0):
        self.Ts = ref["time_step"][1] - ref["time_step"][0]
        self.t = ref["time_step"]
        self.Tfin = ref["time_step"][-1]

        self.pos_ref[id] = ref["trajectory"]
        self.vref[id] = ref["v_ref"]
        self.psi_ref[id] = np.zeros(ref["trajectory"].shape)

    def initialize_parameters(self):
        self.g = 9.81
        self.m = 28e-3
        self.I_BF = np.diag([1.4, 1.4, 2.2]) * 1e-5

        self.Npred = 15

        self.A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 6))]])
        self.B = np.block([[np.zeros((3, 3))], [np.eye(3)]])
        self.A_d = np.eye(6) + self.Ts * self.A
        self.B_d = self.B * self.Ts

        self.Q = np.diag([100, 100, 100, 5, 5, 5])
        self.R = np.diag([0.1, 0.1, 0.1])

        self.M = np.vstack([np.linalg.matrix_power(self.A_d, i) for i in range(1, self.Npred + 1)])
        n, m = self.B_d.shape
        self.H = np.zeros((self.Npred * n, self.Npred * m))

        for i in range(self.Npred):
            for j in range(i + 1):
                self.H[i*n: (i+1)*n, j*m: (j+1)*m] = np.linalg.matrix_power(self.A_d, i - j) @ self.B_d

    def initialize_states(self, pos_init):
        self.state_xi = {i: np.zeros((6, len(self.t))) for i in range(self.Na)}
        self.state_eta = {i: np.zeros((6, len(self.t))) for i in range(self.Na)}
        self.angle_refs = {i: np.zeros((3, len(self.t))) for i in range(self.Na)}
        self.thrusts = {i: np.zeros(len(self.t)) for i in range(self.Na)}
        self.angular_acc_ref = {i: np.zeros((3, len(self.t))) for i in range(self.Na)}
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

        solver = cas.Opti()
        x = solver.variable(n, self.Npred + 1)
        v = solver.variable(du, self.Npred)

        repulsion_forces_var = solver.variable(du, self.Npred)
        Q_rep = np.diag([1, 1, 1])

        xinit = solver.parameter(n, 1)
        vinit = solver.parameter(du, 1)
        xref_param = solver.parameter(n, self.Npred)
        psi_ref_param = solver.parameter(1, 1)

        # Set constraints
        solver.subject_to(x[:, 0] == xinit)
        for k in range(self.Npred):
            solver.subject_to(x[:, k+1] == self.A_d @ x[:, k] + self.B_d @ v[:, k])

            solver.subject_to(cas.mtimes(self.Vc['A_vc'], v[:, k]) <= self.Vc['b_vc'])

            # solver.subject_to(v[2, k] >= -self.g)
            # solver.subject_to(v[0, k]**2 + v[1, k]**2 <= (v[2, k] + self.g)**2 * np.tan(eps_max)**2)
            # solver.subject_to(v[0, k]**2 + v[1, k]**2 + (v[2, k] + self.g)**2 <= T_max**2)

        # Set objective
        objective = 0
        for k in range(self.Npred):
            state_error = x[:, k] - xref_param[:, k]
            control_effort = v[:, k] - (v[:, k-1] if k > 0 else vinit)
            objective += cas.mtimes([state_error.T, self.Q, state_error]) + cas.mtimes([control_effort.T, self.R, control_effort])
            # objective += cas.mtimes([repulsion_forces_var[:, k].T, Q_rep, repulsion_forces_var[:, k]])

        solver.minimize(objective)
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}
        solver.solver('ipopt', opts)

        self.x = x
        self.v = v
        self.solver = solver
        self.xinit = xinit
        self.vinit = vinit
        self.xref_param = xref_param
        self.psi_ref_param = psi_ref_param

        self.objective = objective
        self.repulsion_forces_var = repulsion_forces_var

    def run(self):
        self.solver.set_value(self.vinit, np.zeros(3))
        start_time = time.time()

        for i in range(len(self.t) - 1):
            for id in range(self.Na):
                self.compute_control(i, id=id)
                self.update_states(i, id=id)

        elapsed_time = time.time() - start_time

        print(f"Elapsed time for a {len(self.t)} simulation horizon is {elapsed_time}.")
        comment = 'less' if elapsed_time / len(self.t) < self.Ts else 'more'
        print(f"One control loop lasts {elapsed_time / len(self.t)} and is {comment} than the sample time {self.Ts}.")

    def compute_control(self, i, id=0):
        nearby_agents = []
        closest_agent = None
        min_distance = float("inf")
        d0 = 0.25

        # for j in range(self.Na):
        #     if id != j:
        #         diff = self.state_xi[id][0:3, i] - self.state_xi[j][0:3, i]
        #         distance = np.linalg.norm(diff)
        #         if distance < d0 and distance > 1e-3:  # If agent is too close, communication starts
        #             if distance < min_distance:
        #                 min_distance = distance
        #                 closest_agent = j

        # repulsion_forces = cas.MX.zeros(3, self.Npred)
        # Q_rep = np.diag([0.1, 0.1, 0.1])

        # if closest_agent is not None:
        #     print(f"[DETECTED CLOSEST AGENT] for {id}, detected that {closest_agent} is too close at time {i} with dist {min_distance}\n")
        #     file.write(f"[DETECTED CLOSEST AGENT] for {id}, detected that {closest_agent} is too close at time {i} with dist {min_distance}\n")

        #     if i + self.Npred <= len(self.t):
        #         agent_control = self.vsim[closest_agent][:, i:i+self.Npred]
        #     else:
        #         agent_control = self.vsim[closest_agent][:, -self.Npred:]

        #     agent_position = self.state_xi[closest_agent][0:6, i]

        #     k_rep = 1

        #     predicted_evolution = self.M @ agent_position.reshape(-1, 1) + self.H @ agent_control.flatten().reshape(-1, 1)
        #     predicted_evolution = predicted_evolution.reshape(self.Npred, 6).T
        #     predicted_diff = self.x[0:3, 1:self.Npred+1] - predicted_evolution[0:3, :]

        #     for idx in range(self.Npred):
        #         distance = cas.norm_2(predicted_diff[:, idx])
        #         threshold = cas.logic_and(distance < d0, distance > 1e-2)
        #         repulsion_forces[:, idx] += cas.if_else(threshold,
        #                                                 k_rep * (1 / distance - 1 / d0) * (predicted_diff[:, idx] / distance**3),
        #                                                 0)

        if i + self.Npred <= len(self.t):
            desired_pos = self.pos_ref[0][i:i+self.Npred, 0:3].T + np.tile(self.formation_offsets[id].reshape(-1, 1), self.Npred)
            # ref = np.vstack([desired_pos, self.vref[0][i:i+self.Npred, 0:3].T])
            ref = np.vstack([desired_pos, self.pos_ref[0][i:i+self.Npred, 3:6].T])
        else:
            desired_pos = self.pos_ref[0][-self.Npred:, 0:3].T + np.tile(self.formation_offsets[id].reshape(-1, 1), self.Npred)
            # ref = np.vstack([desired_pos, self.vref[0][-self.Npred:, 0:3].T])
            ref = np.vstack([desired_pos, self.pos_ref[0][-self.Npred:, 3:6].T])

        self.solver.set_value(self.xref_param, ref)
        self.solver.set_value(self.xinit, self.state_xi[id][:, i])
        self.solver.set_value(self.psi_ref_param, self.psi_ref[0][i, 0])

        # print(f"[ITERATION {i}]\n")
        # file.write(f"[ITERATION {i}]\n")

        # self.repulsion_forces_var = repulsion_forces

        # new_objective = self.objective
        # for k in range(self.Npred):
        #     new_objective += cas.mtimes([self.repulsion_forces_var[:, k].T, Q_rep, self.repulsion_forces_var[:, k]])

        # self.solver.minimize(new_objective)

        sol = self.solver.solve()
        vopt = sol.value(self.v)

        self.vsim[id][:, i] = vopt[:, 0]

        # if i+self.Npred <= len(self.t):
        #     self.vsim[id][:, i:i+self.Npred] = vopt
        # else:
        #     self.vsim[id][:, -self.Npred:] = vopt

    def update_states(self, i, id=0):
        # if i+self.Npred <= len(self.t):
        #     v_ap = self.vsim[id][:, i]
        # else:
        #     v_ap = self.vsim[id][:, -self.Npred]

        v_ap = self.vsim[id][:, i]

        T, phi_d, theta_d = fl_laws(v_ap, self.state_eta[id][2, i])

        self.thrusts[id][i] = T
        self.angle_refs[id][:, i] = np.hstack([phi_d, theta_d, self.psi_ref[0][i, 0]])

        if i == 0:
            self.angular_acc_ref[id][:, i] = 0
        else:
            self.angular_acc_ref[id][:, i] = (self.angle_refs[id][:, i] - self.angle_refs[id][:, i-1]) / self.Ts

        [tau, sigma] = run_att_PD(self.state_eta[id][:, i], self.angle_refs[id][:, i], self.angular_acc_ref[id][:, i], 0)
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

    def save_results(self, idx=0):
        save_to_mat("results.mat", self.t, self.state_xi, self.pos_ref)
        for id in range(self.Na):
            plot_positions(self.t, self.state_xi[id], self.pos_ref[id], id=id)
            plot_angles(self.t, self.state_eta[id], self.angle_refs[id], 0.25, id=id)


if __name__ == "__main__":

    rref = get_ref(0, 30, 0.1)

    rref2 = copy.deepcopy(rref)
    rref2["trajectory"] = np.copy(rref["trajectory"])
    rref2["trajectory"][:, 0] += 1
    rref2["trajectory"][:, 1] += 1
    rref2["trajectory"][:, 2] += 1

    Na = 1
    pos_init = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, -1, 0])])
    # pos_init = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0])])
    # pos_init = np.array([np.array([1, 0, 0]), np.array([0, 1, 0])])

    pos_init = np.array([np.array([2, 1, 0])])

    controller = IMPC(rref, pos_init)
    controller.run()
    
    plot_positions(controller.t, controller.state_xi[0], controller.pos_ref[0], id=0)
    # plot_positions(controller.t, controller.state_xi[1], controller.pos_ref[0], id=1)

    # plot_agent_distance(controller.t, controller.state_xi[0], controller.state_xi[1])
    # plot_velocities(controller.t, controller.state_xi[0], controller.pos_ref[0][:, 3:6], id=0)
    plot_angles(controller.t, controller.state_eta[0], controller.angle_refs[0], 0.25, id=0)

    plot_real_u(controller.t, controller.thrusts[0], controller.angle_refs[0], id=0)

    # plot_formation_error(controller.t, controller.formation_error, controller.Na)

    plot_traj_animated(controller.t, controller.state_xi, controller.pos_ref)

    plt.show()

    file.close()
