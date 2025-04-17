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

log_file = "./log.log"
file = open(log_file, "w")


class IMPC:
    def __init__(self, ref, pos_init):
        self.Na = len(pos_init)
        self.psi_ref = {i: 0 for i in range(self.Na)}
        self.pos_ref = {i: 0 for i in range(self.Na)}
        self.vref = {i: 0 for i in range(self.Na)}

        # self.formation_offsets = generate_formation_offsets(self.Na, d=0.3)
        # self.formation_offsets = np.array([[0, 0, 0] for _ in range(self.Na)])

        data_Upos = np.load('./Upos.npy', allow_pickle=True).tolist()
        self.Vc = {}
        self.Vc['A_vc'] = np.round(data_Upos['A'][0, 0], 5)
        self.Vc['b_vc'] = np.round(data_Upos['b'][0, 0], 5)

        self.Tfin = 30
        self.Ts = 0.1
        self.t = np.arange(0, self.Tfin, self.Ts)

        self.pos_ref[0] = np.tile([4, 0.99, 0, 0, 0, 0], (len(self.t), 1))
        self.psi_ref[0] = np.zeros(len(self.t))

        self.pos_ref[1] = np.tile([1, 1.01, 0, 0, 0, 0], (len(self.t), 1))
        self.psi_ref[1] = np.zeros(len(self.t))

        self.pos_ref[2] = np.tile([4, 3.99, 0, 0, 0, 0], (len(self.t), 1))
        self.psi_ref[2] = np.zeros(len(self.t))

        self.pos_ref[3] = np.tile([2, 2.01, 0, 0, 0, 0], (len(self.t), 1))
        self.psi_ref[3] = np.zeros(len(self.t))

        self.initialize_parameters()  # add Q and R here?
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

        self.Q = np.diag([10, 10, 10, 10, 10, 10])
        self.R = np.diag([1, 1, 1])

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

        solver = cas.Opti()
        x = solver.variable(n, self.Npred + 1)
        v = solver.variable(du, self.Npred)
        phi_i = solver.variable(self.Na, 1)

        xinit = solver.parameter(n, 1)
        vinit = solver.parameter(du, 1)
        xref_param = solver.parameter(n, self.Npred)
        psi_ref_param = solver.parameter(1, 1)

        Acoll = solver.parameter(self.Na, 3)
        bcoll = solver.parameter(self.Na, 1)
        coll_step = solver.parameter(1, 1)
        dij_at_coll = solver.parameter(self.Na, self.Na)
        # I think i have to make this a diagonal matrix with the di and the respective j at the collision step

        self.r_min = 0.35
        # self.eta1, self.eta2 = 0, 5000
        self.eta1 = solver.parameter(1, 1)
        self.eta2 = solver.parameter(1, 1)

        self.THETA = 2.5 * np.eye(3)
        self.THETA_1 = np.linalg.inv(self.THETA)
        self.THETA_2 = self.THETA_1 @ self.THETA_1

        # Set constraints
        solver.subject_to(x[:, 0] == xinit)
        for k in range(self.Npred):
            solver.subject_to(x[:, k+1] == self.A_d @ x[:, k] + self.B_d @ v[:, k])
            solver.subject_to(cas.mtimes(self.Vc['A_vc'], v[:, k]) <= self.Vc['b_vc'])

            solver.subject_to(cas.mtimes(Acoll, x[:3, coll_step + 1]) + cas.mtimes(dij_at_coll, phi_i) / 2 <= bcoll)

            # solver.subject_to(phi_i <= cas.MX.zeros(self.Na, 1))

        # Set objective
        objective = 0
        for k in range(self.Npred):
            state_error = x[:, k] - xref_param[:, k]
            control_effort = v[:, k] - (v[:, k-1] if k > 0 else vinit)
            objective += cas.mtimes([state_error.T, self.Q, state_error]) + cas.mtimes([control_effort.T, self.R, control_effort])

        for i in range(self.Na):
            objective += cas.mtimes(self.eta2, phi_i[i]**2) - cas.mtimes(self.eta1, phi_i[i])

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

        self.phi_i = phi_i
        self.Acoll = Acoll
        self.bcoll = bcoll
        self.coll_step = coll_step
        self.dij_at_coll = dij_at_coll

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
        Acoll = np.zeros((self.Na, 3))
        bcoll = np.zeros((self.Na, 1))
        dij_colls = np.zeros((self.Na, self.Na))
        collision_steps = {j: -1 for j in range(self.Na)}
        
        eta1, eta2 = 0, 0

        if i >= 1:
            W_il = []
            # new_objective = self.objective
            for j in range(self.Na):
                if id != j:
                    for k in range(self.Npred - 1):
                        pos_id = self.predicted_evolution[id][:3, k+1]
                        pos_j = self.predicted_evolution[j][:3, k+1]
                        dist = np.linalg.norm(pos_id - pos_j)

                        if dist < self.r_min:
                            collision_steps[j] = k
                            W_il.append(j)
                            file.write(f"[ITERATION {i}] Collision detected between agent {id} and agent {j} dist = {dist} at prediction step {k} and sim time {i+k}\n")
                            eta1, eta2 = 0, 1000
                            break

            for j in W_il:
                diff = self.predicted_evolution[id][:3, collision_steps[j]] - self.predicted_evolution[j][:3, collision_steps[j]]
                dij_l = self.THETA_1 * diff
                dij_l = np.linalg.norm(dij_l)
                dij_colls[j, j] = dij_l
                A = -(self.THETA_2 @ diff)
                bcoll[j] = A @ self.predicted_evolution[id][:3, collision_steps[j]] - (self.r_min - dij_l) * dij_l / 2
                Acoll[j] = A.reshape(1, -1)
                # new_objective = new_objective + self.eta2 * self.phi_i[j]**2 - self.eta1 * self.phi_i[j]

            # self.solver.minimize(new_objective)

        self.solver.set_value(self.Acoll, Acoll)
        self.solver.set_value(self.bcoll, bcoll)
        self.solver.set_value(self.dij_at_coll, dij_colls)
        
        self.solver.set_value(self.eta1, eta1)
        self.solver.set_value(self.eta2, eta2)

        valid_points = [l for l in collision_steps.values() if l >= 0]
        if valid_points:
            self.solver.set_value(self.coll_step, min(valid_points))
        else:
            self.solver.set_value(self.coll_step, 0)

        if i + self.Npred <= len(self.t):
            desired_pos = self.pos_ref[id][i:i+self.Npred, 0:3].T
            ref = np.vstack([desired_pos, self.pos_ref[0][i:i+self.Npred, 3:6].T])
        else:
            desired_pos = self.pos_ref[id][-self.Npred:, 0:3].T
            ref = np.vstack([desired_pos, self.pos_ref[0][-self.Npred:, 3:6].T])

        self.solver.set_value(self.xref_param, ref)
        self.solver.set_value(self.xinit, self.state_xi[id][:, i])
        self.solver.set_value(self.psi_ref_param, self.psi_ref[id][i])

        sol = self.solver.solve()
        vopt = sol.value(self.v)
        phi_i = sol.value(self.phi_i)

        # print(phi_i)

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

    rref = get_ref(0, 30, 0.1)

    rref2 = copy.deepcopy(rref)
    rref2["trajectory"] = np.copy(rref["trajectory"])
    rref2["trajectory"][:, 0] += 1
    rref2["trajectory"][:, 1] += 1
    rref2["trajectory"][:, 2] += 1

    Na = 2
    # pos_init = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, -1, 0])])
    # pos_init = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0])])
    pos_init = np.array([np.array([1, 1, 0]), np.array([4, 1, 0]), np.array([2, 2, 0]), np.array([4, 4, 0])])
    # pos_init = np.array([np.array([2, 1, 0])])

    controller = IMPC(rref, pos_init)
    controller.run()

    # plot_positions(controller.t, controller.state_xi[0], controller.pos_ref[0], id=0)
    # plot_positions(controller.t, controller.state_xi[1], controller.pos_ref[1], id=1)

    plot_all_agent_distances(controller.t, controller.state_xi, d0=controller.r_min)
    # plot_velocities(controller.t, controller.state_xi[0], controller.pos_ref[0][:, 3:6], id=0)
    # plot_angles(controller.t, controller.state_eta[0], controller.angle_refs[0], 0.25, id=0)
    plot_real_u(controller.t, controller.thrusts[0], controller.angle_refs[0], id=0)
    plot_real_u(controller.t, controller.thrusts[1], controller.angle_refs[1], id=1)
    plot_real_u(controller.t, controller.thrusts[2], controller.angle_refs[2], id=2)
    plot_real_u(controller.t, controller.thrusts[3], controller.angle_refs[3], id=3)

    # plot_formation_error(controller.t, controller.formation_error, controller.Na)

    plot_traj_animated(controller.t, controller.state_xi, controller.pos_ref)
    plt.show()

    file.close()
