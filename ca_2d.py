import numpy as np
import casadi as cas
import time
from utils_2d import *
from att_utils import *
from generate_traj import *
from Bspline_conversionMatrix import *
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


log_file = "./log.log"
file = open(log_file, "w")


class IMPC:
    def __init__(self, pos_init):
        self.Na = len(pos_init)
        self.pos_ref = {i: 0 for i in range(self.Na)}
        self.vref = {i: 0 for i in range(self.Na)}

        data_Upos = np.load('./Upos.npy', allow_pickle=True).tolist()
        self.Vc = {}
        self.Vc['A_vc'] = np.round(data_Upos['A'][0, 0], 5)
        self.Vc['b_vc'] = np.round(data_Upos['b'][0, 0], 5)

        self.Tfin = 40
        self.Ts = 0.1
        self.t = np.arange(0, self.Tfin, self.Ts)

        # temp = get_ref(0, self.Tfin, self.Ts)
        # self.pos_ref[0] = temp["trajectory"]
        # self.pos_ref[1] = self.pos_ref[0]

        self.pos_ref[0] = np.tile([3.0, 3.0, 0, 0], (len(self.t), 1))
        self.pos_ref[1] = np.tile([1.0, 3.0, 0, 0], (len(self.t), 1))
        self.pos_ref[2] = np.tile([1.0, 1.0, 0, 0], (len(self.t), 1))
        self.pos_ref[3] = np.tile([3.0, 1.0, 0, 0], (len(self.t), 1))
        self.pos_ref[4] = np.tile([3.0, 2.0, 0, 0], (len(self.t), 1))
        self.pos_ref[5] = np.tile([1.0, 2.0, 0, 0], (len(self.t), 1))

        self.initialize_parameters()
        self.initialize_states(pos_init)
        self.setup_solver()

    def initialize_parameters(self):
        self.Npred = 15

        self.A = np.block([[np.zeros((2, 2)), np.eye(2)], [np.zeros((2, 4))]])
        self.B = np.block([[np.zeros((2, 2))], [np.eye(2)]])
        self.A_d = np.eye(4) + self.Ts * self.A
        self.B_d = self.B * self.Ts

        self.Q = np.diag([1, 1, 1, 1])
        self.R = np.diag([1.5, 1.5])

        self.r_min = 0.2
        self.theta_scale_factor = 2
        self.THETA = self.theta_scale_factor * np.eye(2)
        self.THETA_1 = np.linalg.inv(self.THETA)
        self.THETA_2 = self.THETA_1 @ self.THETA_1

    def initialize_states(self, pos_init):
        self.state_xi = {i: np.zeros((4, len(self.t))) for i in range(self.Na)}
        self.predicted_evolution = {i: np.zeros((4, self.Npred + 1)) for i in range(self.Na)}
        self.vsim = {i: np.zeros((2, len(self.t))) for i in range(self.Na)}
        self.computations_times = []

        vel_init = np.zeros(2)
        for i in range(self.Na):
            self.state_xi[i][:, 0] = np.hstack((pos_init[i, :], vel_init))

    def setup_solver(self):
        n, du = 4, 2

        self.spline_degree = 3
        self.n_ctrl_pts = 5
        self.knot_vec = knot_vector(self.spline_degree, self.n_ctrl_pts, [0, self.Ts * self.Npred])
        self.basis_funcs = b_spline_basis_functions(self.n_ctrl_pts, self.spline_degree, self.knot_vec)
        self.conv_M = bsplineConversionMatrices(self.n_ctrl_pts, self.spline_degree, self.knot_vec)

        self.P_history = []

        solver = cas.Opti()
        P_i = solver.variable(2, self.n_ctrl_pts)
        phi_i = solver.variable(self.Na, 1)

        xinit = solver.parameter(n, 1)
        vinit = solver.parameter(du, 1)
        xref_param = solver.parameter(n, self.Npred)
        # vref_param = solver.parameter(3, self.Npred)

        self.eta1 = solver.parameter(1, 1)
        self.eta2 = solver.parameter(1, 1)

        Acoll = solver.parameter(self.Na, 2)
        bcoll = solver.parameter(self.Na, 1)
        coll_step = solver.parameter(1, 1)
        dij_at_coll = solver.parameter(self.Na, self.Na)
        self.b0_coll_param = solver.parameter(self.n_ctrl_pts, 1)

        # Set initial constraint: spline must start at current "measured" state (positions and velocity)
        b0 = np.array([f(0.0) for f in self.basis_funcs[0]]).reshape(-1, 1)
        b1 = np.array([f(0.0) for f in self.basis_funcs[1]]).reshape(-1, 1)
        pos = cas.mtimes(P_i, b0)
        vel = cas.mtimes(cas.mtimes(P_i, self.conv_M[0]), b1)
        solver.subject_to(cas.vertcat(pos, vel) == xinit)

        objective = 0

        for k in range(self.Npred):
            tk = self.Ts * k
            b0 = np.array([f(tk) for f in self.basis_funcs[0]]).reshape(-1, 1)
            b1 = np.array([f(tk) for f in self.basis_funcs[1]]).reshape(-1, 1)
            b2 = np.array([f(tk) for f in self.basis_funcs[2]]).reshape(-1, 1)

            pos = cas.mtimes(P_i, b0)
            vel = cas.mtimes(cas.mtimes(P_i, self.conv_M[0]), b1)
            acc = cas.mtimes(cas.mtimes(P_i, self.conv_M[1]), b2)

            pos_error = pos - xref_param[0:2, k]
            vel_error = vel - xref_param[2:4, k]
            acc_error = acc

            objective += cas.mtimes([pos_error.T, self.Q[0:2, 0:2], pos_error])
            objective += cas.mtimes([vel_error.T, self.Q[2:4, 2:4], vel_error])
            objective += cas.mtimes([acc_error.T, self.R, acc_error])

            # solver.subject_to(cas.mtimes(self.Vc['A_vc'][0:2, 0:2], acc) <= self.Vc['b_vc'][0:2])
            solver.subject_to(cas.mtimes(Acoll, cas.mtimes(P_i, self.b0_coll_param)) + cas.mtimes(dij_at_coll, phi_i) / 2 <= bcoll)
            solver.subject_to(phi_i <= cas.MX.zeros(self.Na, 1))

        for i in range(self.Na):
            objective += cas.mtimes(self.eta2, phi_i[i]**2) - cas.mtimes(self.eta1, phi_i[i])

        solver.minimize(objective)
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}
        solver.solver('ipopt', opts)

        self.P_i = P_i
        self.solver = solver
        self.xinit = xinit
        self.vinit = vinit
        self.xref_param = xref_param
        # self.vref_param = vref_param
        self.objective = objective

        self.phi_i = phi_i
        self.Acoll = Acoll
        self.bcoll = bcoll
        self.coll_step = coll_step
        self.dij_at_coll = dij_at_coll

    def run(self):
        self.solver.set_value(self.vinit, np.zeros(2))
        start_time = time.time()

        for i in range(len(self.t) - 1):
            print(f"[ITERATION {i}]\n")
            tic = time.time()
            for id in range(self.Na):
                self.compute_control(i, id=id)
                self.update_states(i, id=id)
            # file.write(f"[TIME] {time.time() - tic}\n")
            self.computations_times.append(time.time() - tic)
            file.write("\n")

        elapsed_time = time.time() - start_time

        print(f"Elapsed time for a {len(self.t)} simulation horizon is {elapsed_time}.")
        comment = 'less' if elapsed_time / len(self.t) < self.Ts else 'more'
        print(f"One control loop lasts {elapsed_time / len(self.t)} and is {comment} than the sample time {self.Ts}.")

    def detect_collision(self, i, id):
        """ Return the set of neighbours W_il that has all the agents i will collide with in the future 
            and the dicitonary of the predicted collision step with each one. """
        collision_steps = {j: -1 for j in range(self.Na)}
        # self.detected_iterations_for_collision = []
        W_il = []
        for j in range(self.Na):
            if id != j:
                for k in range(self.Npred - 1):
                    pos_id = self.predicted_evolution[id][:2, k+1]
                    pos_j = self.predicted_evolution[j][:2, k+1]
                    dist = np.linalg.norm(pos_id - pos_j)

                    if dist < self.r_min:
                        collision_steps[j] = k
                        W_il.append(j)
                        file.write(f"[ITERATION {i}] Collision detected between agent {id} and agent {j} dist = {dist} at prediction step {k} and sim time {i+k}\n")
                        break
        return W_il, collision_steps

    def compute_control(self, i, id=0):
        Acoll = np.zeros((self.Na, 2))
        bcoll = np.zeros((self.Na, 1))
        dij_colls = np.zeros((self.Na, self.Na))
        collision_steps = {j: -1 for j in range(self.Na)}

        eta1, eta2 = 0, 1000

        if i >= 1:
            W_il, collision_steps = self.detect_collision(i, id)
            for j in W_il:
                diff = self.predicted_evolution[id][:2, collision_steps[j]] - self.predicted_evolution[j][:2, collision_steps[j]]
                dij_l = np.linalg.norm(self.THETA_1 * diff)
                dij_colls[j, j] = dij_l
                A = -(self.THETA_2 @ diff)
                bcoll[j] = A @ self.predicted_evolution[id][:2, collision_steps[j]] - (self.r_min - dij_l) * dij_l / 2
                Acoll[j] = A.reshape(1, -1)

        valid_points = [l for l in collision_steps.values() if l >= 0]
        l = min(valid_points) if valid_points else 0
        b0_coll_val = np.array([f(self.Ts * (l + 1)) for f in self.basis_funcs[0]]).reshape(-1, 1)

        self.solver.set_value(self.coll_step, l)
        self.solver.set_value(self.b0_coll_param, b0_coll_val)

        self.solver.set_value(self.Acoll, Acoll)
        self.solver.set_value(self.bcoll, bcoll)
        self.solver.set_value(self.dij_at_coll, dij_colls)

        self.solver.set_value(self.eta1, eta1)
        self.solver.set_value(self.eta2, eta2)

        if i + self.Npred <= len(self.t):
            desired_pos = self.pos_ref[id][i:i+self.Npred, 0:2].T
            ref = np.vstack([desired_pos, self.pos_ref[id][i:i+self.Npred, 2:4].T])
        else:
            desired_pos = self.pos_ref[id][-self.Npred:, 0:2].T
            ref = np.vstack([desired_pos, self.pos_ref[id][-self.Npred:, 2:4].T])

        self.solver.set_value(self.xref_param, ref)
        # self.solver.set_value(self.vref_param, ref)
        self.solver.set_value(self.xinit, self.state_xi[id][:, i])

        sol = self.solver.solve()
        P_sol = sol.value(self.P_i)
        self.P_history.append(P_sol)

        b2 = np.array([f(0.0) for f in self.basis_funcs[2]]).reshape(-1, 1)
        acc_i = P_sol @ self.conv_M[1] @ b2
        self.vsim[id][:, i] = acc_i.flatten()

        tt_pred = np.linspace(0, self.Ts * self.Npred, self.Npred + 1)
        B_mat = np.array([[f(tk) for f in self.basis_funcs[0]] for tk in tt_pred])
        self.predicted_evolution[id] = P_sol @ B_mat.T

    def update_states(self, i, id=0):
        self.state_xi[id][:, i + 1] = self.A_d @ self.state_xi[id][:, i] + self.B_d @ self.vsim[id][:, i]

    def plot_computation_times(self):
        avg = np.mean(self.computations_times)
        timesteps = range(0, len(self.computations_times))
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, self.computations_times, marker='o', color='tab:blue', label="", markersize=4, linewidth=1, alpha=1)
        plt.axhline(avg, color='magenta', linestyle='--', linewidth=2, label=f"Avg = {avg:.3f} s")

        plt.xlabel("Time Index n")
        plt.ylabel("Computation Time (s)")
        plt.title("MPC Solve Time per Step")
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()


if __name__ == "__main__":

    pos_init = np.array([np.array([1, 1]), np.array([3, 1]), np.array([3, 3]), 
                         np.array([1, 3]), ])
    # np.array([1, 2]), np.array([3, 2])
    # pos_init = np.array([np.array([1, 1]),  np.array([3, 3])])

    controller = IMPC(pos_init)
    controller.run()

    # plot_positions(controller.t, controller.state_xi[0], controller.pos_ref[0], id=0)
    # plot_velocities(controller.t, controller.state_xi[0], controller.pos_ref[0][:, 2:4], id=0)

    # plot_positions(controller.t, controller.state_xi[1], controller.pos_ref[1], id=1)
    # plot_velocities(controller.t, controller.state_xi[1], controller.pos_ref[1][:, 2:4], id=1)

    plot_all_agent_distances(controller.t, controller.state_xi, d0=controller.r_min)
    for i in range(len(pos_init)):
        plot_virtual_input(controller.t, controller.vsim[i], id=i)
    # controller.plot_computation_times()

    plot_traj_animated(controller.t, controller.state_xi, controller.pos_ref)

    plt.show()

    file.close()
