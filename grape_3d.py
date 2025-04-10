import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os


class GrapeClustering3D:
    def __init__(self, num_targets, num_agents, comm_distance, deployment_type):
        self.num_targets = num_targets
        self.num_agents = num_agents
        self.deployment_type = deployment_type
        self.comm_distance = comm_distance

        # self.task_location_range = [3, 3]
        self.agent_location_range = [3, 3]

        # self.task_location_range = [30, 30]
        # self.agent_location_range = [10, 10]

    def verify_scenario_connectivity(self, agent_comm_matrix, flag_display=True):
        """ Verifies if all agents in the scenario are connected directly or indirectly."""
        eigenvalues = np.linalg.eigvalsh(agent_comm_matrix)
        zero_eigenvalues_count = np.sum(np.abs(eigenvalues) < 1e-12)
        if flag_display:
            print(f"[verify_scenario_connectivity] Number of zero eigenvalues: {zero_eigenvalues_count}")
        if zero_eigenvalues_count > 1:
            if flag_display:
                print(f"[verify_scenario_connectivity] The graph has {zero_eigenvalues_count + 1} connected components, not fully connected.")
                print("[verify_scenario_connectivity] WARNING - Regeneration of the scenario is required.")
            return False
        else:
            if flag_display:
                print("[verify_scenario_connectivity] The graph is connected. The scenario look okay")
            return True

    def generate_task_or_agent_location(self, num_entities, limit, existing_locations=None, gap=1, is_task=False):
        """ Generates locations for tasks or agents ensuring they respect specified constraints for minimum distance between them.

        Parameters:
        - num_entities (int): Number of entities (tasks or agents) to generate.
        - limit (np.array): Array with the maximum range for location generation.
        - existing_locations (np.array): Already generated locations of the other type (for tasks, the existing agent locations).
        - gap (float): Minimum distance between entities.
        - is_task (bool): Flag indicating if generating locations for tasks.
        """
        locations = np.zeros((num_entities, 3))
        for i in range(num_entities):
            ok = False
            while not ok:
                candidate = np.random.uniform(-limit, limit, 2)  # Generate only X and Y
                candidate = np.append(candidate, 0)  # Force Z to be 0

                # Check against other entities: their locations should not be too close to each other
                if i > 0 and np.any(np.linalg.norm(locations[:i, :2] - candidate[:2], axis=1) < gap):
                    continue

                if is_task is not True:
                    if self.deployment_type == 'circle' and np.linalg.norm(candidate[:2]) >= np.min(limit):
                        continue

                if is_task is True:
                    if np.linalg.norm(limit * 0.7) > np.linalg.norm(candidate[:2]):  # task position should not be too close to the agents
                        continue

                locations[i] = candidate
                ok = True
        return locations

    def generate_scenario(self, target_initial_locations, gap_agent=1, gap_target=3):
        agent_locations = self.generate_task_or_agent_location(self.num_agents, np.array(self.agent_location_range), gap=gap_agent)
        print(f"[generate_scenario] Generated agent_locations")
        # target_locations = self.generate_task_or_agent_location(self.num_targets, np.array(self.task_location_range), gap=gap_target, is_task=True)
        # print(f"[generate_scenario] Generated target_locations")
        target_locations = target_initial_locations

        agent_resources = np.random.uniform(10, 60, self.num_agents)
        target_importance = np.random.uniform(1000 * self.num_agents / self.num_targets, 2000 * self.num_agents / self.num_targets, self.num_targets)

        dist_agents = np.linalg.norm(agent_locations[:, np.newaxis, :] - agent_locations[np.newaxis, :, :], axis=-1)
        agent_comm_matrix = (dist_agents <= self.comm_distance).astype(int) - np.eye(self.num_agents, dtype=int)

        environment = {
            'target_locations': target_locations,
            'target_importance': target_importance,
            'agent_locations': agent_locations,
            'agent_resources': agent_resources
        }

        initial_allocation = np.full(self.num_agents, -1, dtype=int)
        print(f"[generate_scenario] Done")

        return {
            'initial_allocation': initial_allocation,
            'agent_comm_matrix': agent_comm_matrix,
            'num_agents': self.num_agents,
            'num_targets': self.num_targets,
            'environment': environment
        }

    def calculate_utility(self, agent_id, task_id, current_alloc, environment, util_type='constant_reward', ignore_cost=False):
        agent_locations = environment['agent_locations']
        agent_resources = environment['agent_resources']
        target_locations = environment['target_locations']
        target_importance = environment['target_importance']

        current_members = (np.array(current_alloc) == task_id)
        current_members[agent_id] = True
        num_participants = np.sum(current_members)

        dist = np.linalg.norm(target_locations[task_id] - agent_locations[agent_id])

        weight_rk = 1.0
        weight_dik = 4.0

        utility = weight_rk * target_importance[task_id] / num_participants - weight_dik * dist

        return max(utility, 0)

    def distributed_mutex(self, agents, agent_comm_matrix):
        num_agents = len(agents)
        agent_updates = [agent.copy() for agent in agents]

        for agent_id in range(num_agents):
            # Identifying neighbouring agents based on the communication matrix, including the agent itself
            neighbour_ids = np.where(agent_comm_matrix[agent_id] > 0)[0]
            neighbour_ids_inclusive = np.append(neighbour_ids, agent_id)

            # Extract iterations and timestamps for the agent and its neighbours
            iterations = np.array([agents[id]['iteration'] for id in neighbour_ids_inclusive])
            timestamps = np.array([agents[id]['time_stamp'] for id in neighbour_ids_inclusive])

            # Determine the agent with the maximum iteration number
            max_iteration = np.max(iterations)
            agents_max_iter = neighbour_ids_inclusive[iterations == max_iteration]

            # Resolve ties with the latest timestamp
            if len(agents_max_iter) > 1:
                timestamps_max_iter = timestamps[iterations == max_iteration]
                decision_maker_id = agents_max_iter[np.argmax(timestamps_max_iter)]
            else:
                decision_maker_id = agents_max_iter[0]

            # Prepare the update based on the decision maker's data
            update_info = {
                'id': agent_id,
                'allocation': agents[decision_maker_id]['allocation'].copy(),
                'iteration': agents[decision_maker_id]['iteration'],
                'time_stamp': agents[decision_maker_id]['time_stamp'],
                'satisfied_flag': agent_updates[agent_id]['satisfied_flag']
            }

            # Check for allocation changes to update the satisfaction flag accordingly
            if not np.array_equal(update_info['allocation'], agents[agent_id]['allocation']):
                update_info['satisfied_flag'] = False
            else:
                update_info['satisfied_flag'] = True

            agent_updates[agent_id].update(update_info)

        # Finalize updates by replacing the original agents' states with the updated ones
        for i in range(num_agents):
            agents[i].update(agent_updates[i])

        return agents

    def grape_allocation(self, scenario, display_progress=True, util_type='constant_reward'):
        num_agents = scenario['num_agents']
        num_targets = scenario['num_targets']
        environment = scenario['environment']
        agent_comm_matrix = scenario['agent_comm_matrix']
        initial_allocation = scenario['initial_allocation']

        # This is local information for each agent
        agents = [{
            'id': i,
            'allocation': initial_allocation.copy(),
            'iteration': 0,
            'time_stamp': np.random.rand(),
            'satisfied_flag': False,
            'util': 0.0
        } for i in range(num_agents)]

        consensus_step = 0
        satisfied_agents_count = 0
        allocation_history = []
        satisfied_agents_count_history = []
        iteration_history = []

        while satisfied_agents_count < num_agents:
            satisfied_agents_count = 0
            for agent_id in range(num_agents):  # Each agent makes its own decision using its local information
                agent = agents[agent_id]
                current_alloc = agent['allocation']
                current_task = current_alloc[agent_id]
                candidates = np.full(num_targets, -np.inf)

                for task_id in range(num_targets):
                    candidates[task_id] = self.calculate_utility(agent_id, task_id, current_alloc, environment, util_type=util_type)

                best_task = np.argmax(candidates)
                best_utility = candidates[best_task]

                if best_utility == 0:
                    best_task = -1

                if current_task == best_task:
                    agent['satisfied_flag'] = True
                    agent['util'] = best_utility
                    satisfied_agents_count += 1
                else:
                    agent['satisfied_flag'] = False
                    agent['time_stamp'] = np.random.rand()
                    agent['iteration'] += 1
                    agent['allocation'][agent_id] = best_task
                    agent['util'] = 0.0  # the agent needs to confirm again

            agents = self.distributed_mutex(agents, agent_comm_matrix)

            if display_progress and consensus_step % 10 == 0:
                print(f"[grape_allocation] Iteration: {consensus_step}, Satisfied Agents: {satisfied_agents_count}/{num_agents}")

            # Final local information
            final_allocation = [agent['allocation'][agent_id] for agent_id in range(num_agents)]
            final_utilities = [agent['util'] for agent in agents]

            # Save data for each consensus step
            consensus_step += 1
            allocation_history.append(final_allocation)
            satisfied_agents_count_history.append(satisfied_agents_count)
            iteration_history.append(max(agent["iteration"] for agent in agents))

        print(f"[grape_allocation] Done")

        history = {
            'allocation': allocation_history,
            'satisfied_agents_count': satisfied_agents_count_history,
            'iteration': iteration_history
        }

        return {
            'final_allocation': final_allocation,
            'final_utilities': final_utilities,
            'consensus_step': consensus_step,
            'history': history,
            'problem_flag': satisfied_agents_count != num_agents
        }

    def visualise_utility(self, agent_id, task_id, environment, max_participants, util_type='constant_reward', filename='fig_utility_type'):
        plt.figure(figsize=(10, 6))
        participant_range = np.arange(1, max_participants + 1)

        utilities = []
        for num_participant in participant_range:
            current_alloc = [task_id if i < num_participant else -1 for i in range(num_agents)]
            utility = self.calculate_utility(agent_id, task_id, current_alloc, environment, util_type, ignore_cost=True)
            utilities.append(utility)

        plt.subplot(1, 2, 1)
        plt.plot(participant_range, utilities, label=util_type)
        plt.subplot(1, 2, 2)
        plt.plot(participant_range, participant_range * np.array(utilities), label=f'Sum {util_type}')

        plt.subplot(1, 2, 1)
        plt.title('Individual Utility Value Changes')
        plt.xlabel('Number of Participants')
        plt.ylabel('Individual Utility Value')
        plt.legend(), plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.title('Aggregated Utility Value Changes')
        plt.xlabel('Number of Participants')
        plt.ylabel('Aggregated Utility Value')
        plt.legend(), plt.grid(True)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        # plt.show()

    def visualise_scenario(self, scenario, final_allocation=None, filename="result_vis.png", agent_heterogeneity=False):
        target_locations = scenario['environment']['target_locations']
        agent_locations = scenario['environment']['agent_locations']
        target_importance = scenario['environment']['target_importance']
        agent_resources = scenario['environment']['agent_resources']
        agent_comm_matrix = scenario['agent_comm_matrix']

        colours = plt.cm.viridis(np.linspace(0, 1, len(target_locations)))
        # colours = plt.cm.plasma(np.linspace(0, 1, len(target_locations)))

        target_importance_sizes = (target_importance - target_importance.min()) / (target_importance.max() - target_importance.min()) * 300 + 50

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')

        for i, j in zip(*np.where(np.triu(agent_comm_matrix) > 0)):
            ax.plot([agent_locations[i, 0], agent_locations[j, 0]],
                    [agent_locations[i, 1], agent_locations[j, 1]],
                    [agent_locations[i, 2], agent_locations[j, 2]],
                    '-', color='gray', linewidth=0.5)

        for i, location in enumerate(agent_locations):
            alloc = final_allocation[i] if final_allocation is not None else -1
            colour = 'black' if alloc == -1 else colours[int(alloc)]
            markersize = agent_resources[i] / np.max(agent_resources) * 10 if agent_heterogeneity is True else 5
            ax.scatter(agent_locations[i, 0], agent_locations[i, 1], agent_locations[i, 2],
                       s=markersize*10, edgecolor='gray', color=colour, marker='o')

        for i, location in enumerate(target_locations):
            ax.scatter(location[0], location[1], location[2], s=target_importance_sizes[i],
                       c=[colours[i]], label=f'Task {i+1}', marker='s')

        plt.title('3D Task Allocation Visualization')
        plt.legend()
        plt.show()

        if filename is None:
            pass
        else:
            plt.savefig(filename, dpi=300)
            plt.close()

    def generate_gif_from_history(self, scenario, allocation_result, filename='result_animation.gif'):
        images = []
        print(f"[generate_gif_from_history] Generating temporary images for GIF")
        final_time = len(allocation_result['history']['iteration'])
        for k in range(0, final_time, 10):  # Step through history every 10 iterations
            current_filename = f"temp_{k}.png"
            current_alloc = allocation_result['history']['allocation'][k]
            self.visualise_scenario(scenario, current_alloc, filename=current_filename)
            images.append(current_filename)

        with imageio.get_writer(filename, mode='I', duration=0.5) as writer:
            for image_path in images:
                image = imageio.imread(image_path)
                writer.append_data(image)
        # Cleanup: Remove temporary image files
        for image_path in images:
            os.remove(image_path)
        print(f"[generate_gif_from_history] Done")

    def run_clustering(self):
        scenario = self.generate_scenario(gap_agent=1, gap_target=3)
        self.verify_scenario_connectivity(scenario['agent_comm_matrix'])
        self.visualise_scenario(scenario, filename='fig_initial_allocation_visualization.png')

        agent_id = 0
        task_id = 0
        environment = scenario['environment']
        self.visualise_utility(agent_id, task_id, environment, max_participants=self.num_agents, filename='fig_utility')

        allocation_result = self.grape_allocation(scenario)
        self.visualise_scenario(scenario, allocation_result['final_allocation'], filename="fig_final_allocation_visualization.png")

        # self.generate_gif_from_history(scenario, allocation_result, filename='result_animation.gif')


if __name__ == "__main__":

    num_targets = 2
    num_agents = 7

    # num_targets = 3
    # num_agents = 10

    deployment_type = 'circle'
    comm_distance = 130

    grape = GrapeClustering3D(num_targets, num_agents, comm_distance, deployment_type)
    grape.run_clustering()
