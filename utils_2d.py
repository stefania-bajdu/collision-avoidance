import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


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
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    labels = ['x (m)', 'y (m)']
    for i in range(2):
        axs[i].plot(t, state_xi[i, :], label='sim')
        axs[i].plot(t, pos_ref[:, i], '--', label='ref')
        axs[i].legend()
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Drone {id+1} Positions Cluster {t_id}', fontsize=14)


def plot_velocities(t, state_xi, v_ref, id=0):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    labels = ['vx (m/s)', 'vy (m/s)']
    for i in range(2):
        axs[i].plot(t, state_xi[i + 2, :], label='sim')
        axs[i].plot(t, v_ref[:, i], '--', label='ref')
        axs[i].legend()
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Drone {id+1} Velocities', fontsize=14)


def plot_virtual_input(t, v, id=0):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    labels = ['ax (m/s^2)', 'ay (m/s^2)']
    for i in range(2):
        axs[i].plot(t, v[i, :], label='sim')
        axs[i].legend()
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Drone {id+1} Virtual Input', fontsize=14)


def plot_traj_animated(t, state_xi, cluster_refs):
    fig, ax = plt.subplots()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Robot Trajectories (2D)')
    ax.grid(True)

    colors = plt.cm.viridis(np.linspace(0, 1, len(state_xi)))

    Na = len(state_xi)
    num_targets = len(cluster_refs)
    trace_length = 12

    x_min = min(np.min(state_xi[i][0, :]) for i in range(Na)) - 0.1
    x_max = max(np.max(state_xi[i][0, :]) for i in range(Na)) + 0.1

    y_min = min(np.min(state_xi[i][1, :]) for i in range(Na)) - 0.1
    y_max = max(np.max(state_xi[i][1, :]) for i in range(Na)) + 0.1

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    for i in range(num_targets):
        ax.plot(cluster_refs[i][:, 0], cluster_refs[i][:, 1], 'k--', label='')

    # Initialize lines and scatter points
    trajectory_lines = []
    scatter_points = []

    for i in range(Na):
        line, = ax.plot([], [], color=colors[i], label=f'R{i+1}')
        scatter = ax.scatter(np.nan, np.nan, color=colors[i], s=50)
        trajectory_lines.append(line)
        scatter_points.append(scatter)

    ref_markers = {}
    for i in range(num_targets):
        ref_markers[i], = ax.plot([], [], 'ro', markersize=4, label='')

    ax.legend()

    print("Starting 2D animation...")
    skip_frames = 1

    for t_idx in range(len(t)):
        start_idx = max(0, t_idx - trace_length + 1)

        for i in range(Na):
            trajectory_lines[i].set_data(state_xi[i][0, start_idx:t_idx+1], state_xi[i][1, start_idx:t_idx+1])
            scatter_points[i].set_offsets([state_xi[i][0, t_idx], state_xi[i][1, t_idx]])

        for i in range(num_targets):
            ref_markers[i].set_data([cluster_refs[i][t_idx, 0]], [cluster_refs[i][t_idx, 1]])

        if t_idx % skip_frames == 0:
            plt.pause(0.2)

    print("Animation finished.")
    plt.show()


def plot_agent_distance(t, state_xi_1, state_xi_2, d0=0.25, ids=[0, 1]):
    pos1 = state_xi_1[0:2, :]
    pos2 = state_xi_2[0:2, :]
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
            dist = np.linalg.norm(state_xi[i][0:2, :] - state_xi[j][0:2, :], axis=0)
            plt.plot(t, dist, label=f"{i}-{j}")

    plt.axhline(y=d0, color='r', linestyle='--', label=f"d0")
    plt.title("Distances Between Agents Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
