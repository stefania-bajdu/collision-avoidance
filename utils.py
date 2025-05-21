import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def generate_formation_offsets(num_agents, d=0.5):
    offsets = np.zeros((num_agents, 3))

    if num_agents == 1:
        offsets[0] = [0, 0, 0]

    elif num_agents == 2:
        # Line formation
        offsets[0] = [0, -d/2, d - d/2]
        offsets[1] = [-d/2, 0, d - d/2]

    elif num_agents == 3:
        # Triangle formation
        offsets[0] = [0, -d/2, d - d/2]
        offsets[1] = [-d/2, d/2, d - d/2]
        offsets[2] = [d/2, d/2, d - d/2]

    elif num_agents == 4:
        # Square formation
        offsets[0] = [-d/2, -d/2, d - d/2]
        offsets[1] = [d/2, -d/2, d - d/2]
        offsets[2] = [-d/2, d/2, d - d/2]
        offsets[3] = [d/2, d/2, d - d/2]

    else:
        # Circular formation for 5+ agents
        theta = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
        for i in range(num_agents):
            offsets[i] = [d/2 * np.cos(theta[i]), d/2 * np.sin(theta[i]), d/2]

    # for i in range(num_agents):
    #     offsets[i][2] = 0  # Force Z to be 0

    return offsets


def save_to_mat(filename, t, state_xi,  cluster_refs):
    data_dict = {
        "time": t,
        "state_xi": state_xi,
        "cluster_refs": cluster_refs
    }
    scipy.io.savemat(filename, data_dict)
    print(f"Data saved to {filename}.")


def load_from_mat(filename):
    data = scipy.io.loadmat(filename)
    t = data['time'].flatten()
    state_xi = data['state_xi']
    cluster_refs = data['time']

    print(f"Data loaded from {filename}")
    return t, state_xi, cluster_refs


def plot_positions(t, state_xi, pos_ref, id=0, t_id=0):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['x (m)', 'y (m)', 'z (m)']
    for i in range(3):
        axs[i].plot(t, state_xi[i, :], label='sim')
        axs[i].plot(t, pos_ref[:, i], '--', label='ref')
        axs[i].legend()
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')
    fig.suptitle(f'Drone {id+1} Positions Cluster {t_id}', fontsize=14)
    # plt.show()


def plot_velocities(t, state_xi, v_ref, id=0):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['x (m)', 'y (m)', 'z (m)']
    for i in range(3):
        axs[i].plot(t, state_xi[i + 3, :], label='sim')
        axs[i].plot(t, v_ref[:, i], '--', label='ref')
        axs[i].legend()
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')
    fig.suptitle(f'Drone {id+1} Velocities', fontsize=14)
    # plt.show()


def plot_angles(t, state_eta, angle_refs, eps_max, id=0):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = [r'$\phi$ (deg)', r'$\theta$ (deg)', r'$\psi$ (deg)']

    for i in range(3):
        axs[i].plot(t, state_eta[i, :] * 180/np.pi, label='sim')
        axs[i].plot(t, angle_refs[i, :] * 180/np.pi, '--', label='ref')
        if i != 2:
            axs[i].axhline(eps_max*180/np.pi, color='r', linestyle='--', alpha=0.7)
            axs[i].axhline(-eps_max*180/np.pi, color='r', linestyle='--', alpha=0.7)
        axs[i].legend()
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')
    fig.suptitle(f'Drone {id+1} Angles', fontsize=14)
    # plt.show()


def plot_real_u(t, thrusts, angle_refs, id=0):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = [r'$\phi$ (deg) - Roll', r'$\theta$ (deg) - Pitch', 'Thrust']
    colors = ['b', 'g', 'r']

    axs[0].plot(t, angle_refs[0, :] * 180/np.pi, label='Desired Roll Angle', color=colors[0])
    axs[1].plot(t, angle_refs[1, :] * 180/np.pi, label='Desired Pitch Angle', color=colors[1])
    axs[2].plot(t, thrusts, label='Total Thrust', color=colors[2])

    for i in range(3):
        axs[i].legend()
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_xlabel('Time (s)')
    fig.suptitle(f'Drone {id+1} - Real Control Inputs', fontsize=14)
    # plt.show()


def plot_traj_animated(t, state_xi, cluster_refs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Robot Trajectories')
    ax.grid(True)

    # colors = ['r', 'g', 'b', 'm']
    colors = plt.cm.viridis(np.linspace(0, 1, len(state_xi)))

    Na = len(state_xi)
    num_targets = len(cluster_refs)
    trace_length = 12

    x_min = min(np.min(state_xi[i][0, :]) for i in range(Na)) - 0.1
    x_max = max(np.max(state_xi[i][0, :]) for i in range(Na)) + 0.1

    y_min = min(np.min(state_xi[i][1, :]) for i in range(Na)) - 0.1
    y_max = max(np.max(state_xi[i][1, :]) for i in range(Na)) + 0.1

    z_min = min(np.min(state_xi[i][2, :]) for i in range(Na)) - 0.1
    z_max = max(np.max(state_xi[i][2, :]) for i in range(Na)) + 0.1

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    for i in range(num_targets):
        ax.plot(cluster_refs[i][:, 0], cluster_refs[i][:, 1], cluster_refs[i][:, 2], 'k--', label='')

    # Initialize trajectory lines and scatter points
    trajectory_lines = []
    scatter_points = []

    for i in range(Na):
        line, = ax.plot([], [], [], color=colors[i], label=f'R{i+1}')
        scatter = ax.scatter(np.nan, np.nan, np.nan, color=colors[i], s=50)
        trajectory_lines.append(line)
        scatter_points.append(scatter)

    # Reference trajectory marker
    ref_markers = {}
    for i in range(num_targets):
        ref_markers[i], = ax.plot([], [], [], 'ro', markersize=4, label='')

    ax.legend()

    print("Starting animation...")
    skip_frames = 1

    for t_idx in range(len(t)):
        # print(f'Time step: {t_idx + 1} / {len(t)}')
        start_idx = max(0, t_idx - trace_length + 1)

        for i in range(Na):
            trajectory_lines[i].set_data(state_xi[i][0, start_idx:t_idx+1], state_xi[i][1, start_idx:t_idx+1])
            trajectory_lines[i].set_3d_properties(state_xi[i][2, start_idx:t_idx+1])

            scatter_points[i]._offsets3d = (np.array([state_xi[i][0, t_idx]]),
                                            np.array([state_xi[i][1, t_idx]]),
                                            np.array([state_xi[i][2, t_idx]]))
        for i in range(num_targets):
            ref_markers[i].set_data([cluster_refs[i][t_idx, 0]], [cluster_refs[i][t_idx, 1]])
            ref_markers[i].set_3d_properties([cluster_refs[i][t_idx, 2]])

        if t_idx % skip_frames == 0:
            plt.pause(0.1)

    print("Animation finished.")
    plt.show()


def plot_agent_distance(t, state_xi_1, state_xi_2, d0=0.25, ids=[0, 1]):
    pos1 = state_xi_1[0:3, :]
    pos2 = state_xi_2[0:3, :]
    distances = np.linalg.norm(pos1 - pos2, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(t, distances, label=f"Distance between Agents {ids[0]} - {ids[1]}", color='b', linewidth=2)
    plt.axhline(y=d0, color='r', linestyle='--', label=f"Safety Threshold {d0}m")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Evolution of Distance Between Agents")
    plt.legend()
    plt.grid()


def plot_all_agent_distances(t, state_xi, d0=0.3):
    """Plot distances between all pairs of agents over time."""
    Na = len(state_xi)
    plt.figure(figsize=(8, 5))

    for i in range(Na):
        for j in range(i + 1, Na):
            dist = np.linalg.norm(state_xi[i][0:3, :] - state_xi[j][0:3, :], axis=0)
            plt.plot(t, dist, label=f"{i}-{j}")

    plt.axhline(y=d0, color='r', linestyle='--', label=f"d0")
    plt.title("Distances Between Agents Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()


def plot_static_drones(Adjll, state_xi, cnt=0):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Initial Drone Positions and Connections")

    Na = len(state_xi)

    initial_positions = np.array([state_xi[i][:3, cnt] for i in range(Na)])

    # Plot initial positions
    for i, pos in enumerate(initial_positions):
        ax.scatter(pos[0], pos[1], s=100, label=f'R{i}', color='blue')

    # Draw connections
    for i in range(len(Adjll)):
        for j in range(i + 1, len(Adjll)):
            if Adjll[i, j] == 1:
                ax.plot([initial_positions[i, 0], initial_positions[j, 0]],
                        [initial_positions[i, 1], initial_positions[j, 1]],
                        'k--')
    plt.legend()


def plot_static_topology(Adjll, state_xi, cnt=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Na = len(state_xi)
    initial_positions = np.array([state_xi[i][:3, cnt] for i in range(Na)])

    # Plot initial positions
    for i, pos in enumerate(initial_positions):
        ax.scatter(pos[0], pos[1], pos[2], s=100, label=f'R{i}', color='blue')

    # Draw connections
    for i in range(len(Adjll)):
        for j in range(i + 1, len(Adjll)):
            if Adjll[i, j] == 1:
                ax.plot([initial_positions[i, 0], initial_positions[j, 0]],
                        [initial_positions[i, 1], initial_positions[j, 1]],
                        [initial_positions[i, 2], initial_positions[j, 2]],
                        'k--')

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("Initial Drone Positions and Communication Links")

    plt.legend()


def plot_formation_error_norm(t, formation_error, Na):
    plt.figure(figsize=(10, 6))

    for id in range(Na):
        plt.plot(t, np.linalg.norm(formation_error[id], axis=0), label=f'R{id}')

    plt.xlabel("Time (s)")
    plt.ylabel("Formation Error Norm")
    plt.title("Formation Error Norm Over Time")
    plt.legend()
    plt.grid()


def plot_formation_error(t, formation_error, Na):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axis_labels = ["X", "Y", "Z"]
    colors = ["r", "g", "b"]  # Red for X, Green for Y, Blue for Z

    for idx, ax in enumerate(axs):
        for drone_id in range(Na):
            # ax.plot(t, formation_error[drone_id][idx, :], label=f'R{drone_id}', linestyle='--')
            ax.plot(t, formation_error[drone_id][idx, :], label=f'R{drone_id}')

        ax.set_ylabel(f"Error in {axis_labels[idx]} (m)", fontsize=12)
        ax.grid(True)
        ax.legend()

    axs[2].set_xlabel("Time (s)", fontsize=12)
    fig.suptitle("Formation Error Over Time (Per Axis)", fontsize=14)
    plt.tight_layout()


# def plot_state_evolution_3d(self, cluster_ids=None):
#         """Plots the 3D state evolution of all agents or selected clusters."""
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')

#         colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']  # Define colors for clusters
#         cluster_refs = {i: self.controllers[i].pos_ref[0] for i in range(self.num_targets)}

#         # If no specific clusters are provided, plot all
#         if cluster_ids is None:
#             cluster_ids = range(self.num_targets)

#         for cluster_id in cluster_ids:
#             if cluster_id not in self.controllers:
#                 continue

#             color = colors[cluster_id % len(colors)]  # Cycle through colors
#             controller = self.controllers[cluster_id]
#             cluster_agents_indices = np.where(np.array(self.allocation_result['final_allocation']) == cluster_id)[0]

#             for agent_id in cluster_agents_indices:
#                 state_trajectory = np.array(self.state_xi_reconstructed[agent_id])  # Shape: (T, 3)

#                 if state_trajectory.shape[0] > 0:
#                     ax.plot(state_trajectory[0, :], state_trajectory[1, :], state_trajectory[2, :],
#                             label=f'Agent {agent_id} (Cluster {cluster_id})', color=color)

#             # Plot cluster reference trajectory
#             ref_traj = np.array(self.reference_trajectories[cluster_id]["trajectory"])[:, :3]
#             ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], linestyle='dashed', color=color, linewidth=2,
#                     label=f'Ref Cluster {cluster_id}')

#         ax.set_title("3D State Evolution of Agents")
#         ax.set_xlabel("X Position")
#         ax.set_ylabel("Y Position")
#         ax.set_zlabel("Z Position")
#         ax.legend()
#         ax.grid(True)

#         plt.show()
