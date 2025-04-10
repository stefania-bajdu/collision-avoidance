import numpy as np
import casadi as cas
import copy
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import *
from att_utils import *
from grape_3d import *
from rep_fields import *
import generate_traj as GT


class CFT:
    def __init__(self, num_targets=3, num_agents=10, comm_distance=130, deployment_type='circle'):
        self.num_targets = num_targets
        self.num_agents = num_agents
        self.comm_distance = comm_distance
        self.deployment_type = deployment_type
        self.controllers = {}
        self.state_xi_reconstructed = {}

        self.reference_trajectories = self.generate_references()

        self.grape = GrapeClustering3D(self.num_targets, self.num_agents, self.comm_distance, self.deployment_type)
        initial_locs = np.vstack([self.reference_trajectories[i]["trajectory"][0, 0:3] for i in range(self.num_targets)])
        self.scenario = self.grape.generate_scenario(target_initial_locations=initial_locs, gap_agent=1, gap_target=3)
        self.allocation_result = self.grape.grape_allocation(self.scenario)

        self.init_controllers()

    def generate_references(self):
        """ Generates reference trajectories for each cluster. """
        rref_base = GT.get_ref(0, 30, 0.1)
        rrefs = {
            0: rref_base,
            1: copy.deepcopy(rref_base),
            2: copy.deepcopy(rref_base),
        }

        rrefs[1]["trajectory"][:, 0] += 1
        rrefs[1]["trajectory"][:, 1] += 1
        rrefs[1]["trajectory"][:, 2] += 1

        rrefs[2]["trajectory"][:, 0] -= 1
        rrefs[2]["trajectory"][:, 1] -= 1
        rrefs[2]["trajectory"][:, 2] += 1

        return rrefs

    def init_controllers(self):
        """Initializes IMPC controllers for each cluster based on allocation."""
        self.controllers = {}

        for cluster_id in range(self.num_targets):
            cluster_indices = np.where(np.array(self.allocation_result['final_allocation']) == cluster_id)[0]
            pos_init = self.scenario["environment"]["agent_locations"][cluster_indices]

            self.controllers[cluster_id] = IMPC(self.reference_trajectories[cluster_id], pos_init)
            self.controllers[cluster_id].solver.set_value(self.controllers[cluster_id].vinit, np.zeros(3))

    def run_simulation(self):
        """Runs the simulation for all controllers."""
        start_time = time.time()

        for i in range(len(self.controllers[0].t) - 1):
            for cluster_id, controller in self.controllers.items():
                for agent_id in range(controller.Na):
                    controller.compute_control(i, id=agent_id)
                    controller.update_states(i, id=agent_id)

        elapsed_time = time.time() - start_time

        print(f"Elapsed time for a {len(self.controllers[0].t)} simulation horizon is {elapsed_time}.")
        comment = 'less' if elapsed_time / len(self.controllers[0].t) < self.controllers[0].Ts else 'more'
        print(f"One control loop lasts {elapsed_time / len(self.controllers[0].t)} and is {comment} than the sample time {self.controllers[0].Ts}.")

    def reconstruct_states(self):
        """ Reconstructs the state_xi dictionary by placing agents in correct positions. """
        self.state_xi_reconstructed = {}

        for cluster_id, controller in self.controllers.items():
            cluster_agents_indices = np.where(np.array(self.allocation_result['final_allocation']) == cluster_id)[0]

            for idx, original_idx in enumerate(cluster_agents_indices):
                self.state_xi_reconstructed[original_idx] = controller.state_xi[idx]

        for i in range(self.num_agents):
            cluster_id = self.allocation_result["final_allocation"][i]
            filename = f"agent_{i}_cluster_{cluster_id}.mat"
            save_to_mat(filename, self.controllers[cluster_id].t, self.state_xi_reconstructed[i], self.controllers[cluster_id].pos_ref[0])

    def visualize_trajectory(self):
        """Visualizes the animated trajectory."""
        cluster_refs = {i: self.controllers[i].pos_ref[0] for i in range(self.num_targets)}
        plot_traj_animated(self.controllers[0].t, self.state_xi_reconstructed, cluster_refs)
        plt.show()

    def plot_state_evolution_xyz(self, cluster_ids=None, save_path="state_evolution.png"):
        """Plots the evolution of X, Y, and Z positions over time in a 3-subfigure layout, ensuring unique colors per agent."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # If no specific clusters are provided, plot all
        if cluster_ids is None:
            cluster_ids = range(self.num_targets)

        for cluster_id in cluster_ids:
            if cluster_id not in self.controllers:
                continue

            controller = self.controllers[cluster_id]
            cluster_agents_indices = np.where(np.array(self.allocation_result['final_allocation']) == cluster_id)[0]

            cmap = plt.cm.get_cmap('tab10', len(cluster_agents_indices))

            for i, agent_id in enumerate(cluster_agents_indices):
                state_trajectory = np.array(self.state_xi_reconstructed[agent_id])
                time_steps = self.controllers[0].t
                agent_color = cmap(i)

                if state_trajectory.shape[0] > 0:
                    axes[0].plot(time_steps, state_trajectory[0, :], label=f'Agent {agent_id} (Cluster {cluster_id})', color=agent_color)
                    axes[1].plot(time_steps, state_trajectory[1, :], label=f'Agent {agent_id}', color=agent_color)
                    # axes[2].plot(time_steps, state_trajectory[2, :], label=f'Agent {agent_id}', color=agent_color)

        axes[0].set_title("X Position Evolution Over Time")
        axes[1].set_title("Y Position Evolution Over Time")
        # axes[2].set_title("Z Position Evolution Over Time")

        axes[1].set_xlabel("Time (s)")
        axes[0].set_ylabel("X Position")
        axes[1].set_ylabel("Y Position")
        # axes[2].set_ylabel("Z Position")

        for ax in axes:
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.show()

    def plot_2d_trajectory(self, cluster_ids=None, save_path="2d_trajectory.png"):
        """Plots the 2D trajectory (X vs. Y) of agents."""
        plt.figure(figsize=(8, 6))

        if cluster_ids is None:
            cluster_ids = range(self.num_targets)

        for cluster_id in cluster_ids:
            if cluster_id not in self.controllers:
                continue

            cluster_agents_indices = np.where(np.array(self.allocation_result['final_allocation']) == cluster_id)[0]
            cmap = plt.cm.get_cmap('tab10', len(cluster_agents_indices))

            for i, agent_id in enumerate(cluster_agents_indices):
                state_trajectory = np.array(self.state_xi_reconstructed[agent_id])
                agent_color = cmap(i)

                if state_trajectory.shape[0] > 0:
                    plt.plot(state_trajectory[0, :], state_trajectory[1, :], label=f'Agent {agent_id}', color=agent_color)

        plt.xlabel("X Position"), plt.ylabel("Y Position")
        plt.title("2D Trajectory of Agents")
        plt.legend()
        plt.grid(True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"2D trajectory plot saved to {save_path}")


if __name__ == "__main__":
    cft = CFT(num_targets=3, num_agents=10)
    cft.run_simulation()
    cft.reconstruct_states()

    # cft.plot_state_evolution_xyz(cluster_ids=[0])

    cft.visualize_trajectory()

    # for i in range(cft.num_agents):
    #     target_idx = cft.allocation_result["final_allocation"][i]
    #     plot_positions(cft.controllers[0].t, cft.state_xi_reconstructed[i], cft.reference_trajectories[target_idx]["trajectory"], id=i, t_id=target_idx)

    plt.show()

    # filename = f"agent_{0}_cluster_{2}.mat"
    # t, state_xi, cluster_refs = load_from_mat(filename)

    # print(state_xi)
