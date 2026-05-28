import numpy as np
from collections import deque
from scipy.stats import spearmanr

"""Helperfunctions for analyses."""

def z_transform_intervals(r, n_simulations):
    """
    Compute error bars for the correlation coefficient using the Fisher Z transform
    r: correlations, np.array of shape (n_simulations, max_d)
    n_simulations: number of simulations

    Returns:
    mean_r: mean of correlation. np.array of shape (max_d)
    lower_r: lower bound. np.array of shape (max_d)
    higher_r: higher bound. np.array of shape (max_d)
    """

    
    ## If the correlation is 1 or -1, the z transform is infinite
    ## Decide on an epsilon value very small to not obtain infinite
    epsilon = 1e-9
    # Clip values of r to be strictly within (-1, 1)
    r = np.clip(r, -1 + epsilon, 1 - epsilon)
    z = np.arctanh(r)

    # Mean and SEM in Z-space
    mean_z = np.mean(z, axis=0)
    sem_z = np.std(z, axis=0, ddof=1) / np.sqrt(n_simulations)

    # Back-transform mean to r-space
    mean_r = np.tanh(mean_z)
    # Back-transform SEM to r-space (delta method)
    sem_r = sem_z * (1 - mean_r**2)

    return mean_r, sem_r

def calculate_distances_to_rewards(transition_matrix, reward_states):
    """Calculate distances from each state to the nearest reward state"""
    adjacency_list = convert_to_adjacency_list(transition_matrix)
    distances = {}
    for state in range(transition_matrix.shape[0]):
        distances[state] = bfs_shortest_path(adjacency_list, state, reward_states)
    return distances

def convert_to_adjacency_list(transition_matrix):
    """
    Convert the transition matrix to an adjacency list representation
    """
    # Generates a list 
    adjacency_list = {i: [] for i in range(transition_matrix.shape[0])}
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            # The sum accross actions can be more than 1, since if boundaries 3 actions could lead to the same state
            if transition_matrix[i, j] > 0:
                adjacency_list[i].append(j)
    return adjacency_list


def bfs_shortest_path(adjacency_list, start, reward_states):
    """Perform BFS to find the shortest path from start to any of the reward states"""
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)

    while queue:
        current, distance = queue.popleft()
        if current in reward_states:
            return distance
        
        for neighbor in adjacency_list[current]:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))
                visited.add(neighbor)
    return float('inf')  # If no path is found

def value_correlation(expert_value, agent_value, agent_states, true_tm, reward_states_per_sim, n_simulations, max_d, states="all"):
    """
    Calculate the correlation between the expert and agent values for each distance to the reward state
    
    expert_value: np.array of shape (n_simulations, n_states, n_actions)
    agent_value: np.array of shape (n_simulations, n_states, n_actions)
    agent_states: np.array of shape (n_simulations, n_episodes, max_steps)
    true_tm: np.array of shape (n_simulations, n_states, n_actions, n_states) 
    reward_states: list of reward states
    n_simulations: number of simulations
    max_d: maximum distance to the reward state
    states: string, consider "all" states, "visited" states, or "unvisited" states for the correlation calculation

    Returns:
    corrs: np.array of shape (n_simulations, max_d)
    """
    # This function takes into account not visited states
    corrs = np.zeros((n_simulations, max_d - 1))
    possible_distances = np.arange(1, max_d)

    n_states = np.array(true_tm).shape[1]
    all_states = np.arange(n_states)

    # Assuming calculate_distances_to_rewards and other functions can be optimized or vectorized
    for sim in range(n_simulations):

        # States visited by the agent
        agent_visited_states = np.unique(agent_states[sim][~np.isnan(agent_states[sim])].astype(int))

        if states == "all":
                considered_states = all_states
        elif states == "visited":
            # Only consider visited states
            considered_states = agent_visited_states
        elif states == "unvisited":
            # Only consider unvisited states
            considered_states = np.setdiff1d(all_states, agent_visited_states)

        # Sum the actions for each state
        true_tm_actions = np.sum(true_tm[sim], axis=1) # Transform to (n_states, n_states)
        distances_to_rewards = calculate_distances_to_rewards(true_tm_actions, reward_states_per_sim[sim][:,0].tolist())
        
        for p in possible_distances:
            state_list = [state for state in considered_states if distances_to_rewards[state] == p]
            
            if state_list:
                v_expert = expert_value[sim][state_list].flatten()
                v_agent = agent_value[sim][state_list].flatten()
                
                var_expert, var_agent = np.var(v_expert), np.var(v_agent)
                if var_expert == 0 or var_agent == 0:
                    corrs[sim, p - 1] = 0
                else:
                    corrs[sim, p - 1] = spearmanr(v_expert, v_agent)[0]

    return corrs

def tm_ztransform_distance(expert_tm, expert_states, agent_tm, agent_states, initial_tm, reward_states_per_sim, n_simulations, max_d, all_states=True):
    """
    Compute correlation between expert and agent TMs per distance, skipping distance 0.

    Returns:
        mean_r: mean correlation per distance (1..max_d-1)
        sem_r: SEM per distance (1..max_d-1)
        mean_z: mean z-transform
        sem_z: SEM z-transform
    """
    corrs = np.zeros((n_simulations, max_d - 1))  # skip distance 0
    possible_distances = np.arange(1, max_d)

    for sim in range(n_simulations):
        # Optional: consider only visited states
        if not all_states:
            if expert_states is not None:
                agent_visited = np.unique(agent_states[sim][~np.isnan(agent_states[sim])].astype(int))
                expert_visited = np.unique(expert_states[sim][~np.isnan(expert_states[sim])].astype(int))
                visited_states = np.intersect1d(agent_visited, expert_visited)
            else:
                visited_states = np.unique(agent_states[sim][~np.isnan(agent_states[sim])].astype(int))

        initial_tm_actions = np.sum(initial_tm[sim], axis=1)
        distances_to_rewards = calculate_distances_to_rewards(initial_tm_actions, reward_states_per_sim[sim][:,0].tolist())

        for p in possible_distances:
            if all_states:
                state_list = [s for s, d in distances_to_rewards.items() if d == p]
            else:
                state_list = [s for s in visited_states if distances_to_rewards[s] == p]

            if state_list:
                tm_expert = expert_tm[sim][state_list].flatten()
                tm_agent = agent_tm[sim][state_list].flatten()

                var_expert, var_agent = np.var(tm_expert), np.var(tm_agent)
                if var_expert == 0 or var_agent == 0:
                    corrs[sim, p - 1] = 0
                else:
                    r = np.corrcoef(tm_expert, tm_agent)[1, 0]
                    r = np.clip(r, -1 + 1e-9, 1 - 1e-9)
                    corrs[sim, p - 1] = r

    # Fisher z-transform
    z = np.arctanh(corrs)
    mean_z = np.mean(z, axis=0)
    sem_z = np.std(z, axis=0, ddof=1) / np.sqrt(n_simulations)

    mean_r = np.tanh(mean_z)
    sem_r = sem_z * (1 - mean_r**2)  # propagate SEM to r-space

    return mean_r, sem_r, mean_z, sem_z

def normalize_tm_correlation(agent_mean_z, agent_sem_z,
                             baseline_mean_z, baseline_sem_z):
    """
    Normalize agent correlations relative to baseline.
    Returns mean delta r and SEM delta r for plotting.
    
    agent_mean_z, baseline_mean_z : mean correlation in Z-space
    agent_sem_z, baseline_sem_z   : SEM in Z-space
    """

    # Convert means to r-space
    mean_agent_r = np.tanh(agent_mean_z)
    mean_baseline_r = np.tanh(baseline_mean_z)
    
    # Delta mean in r-space
    delta_mean_r = mean_agent_r - mean_baseline_r

    # Convert SEM from z -> r (delta method)
    sem_agent_r = agent_sem_z * (1 - mean_agent_r**2)
    sem_baseline_r = baseline_sem_z * (1 - mean_baseline_r**2)

    # Combine SEMs for delta
    delta_sem_r = np.sqrt(sem_agent_r**2 + sem_baseline_r**2)

    return delta_mean_r, delta_sem_r

def tm_agent_similarity_over_distances(expert_tm, expert_states,
                                       agent_tm, agent_states,
                                       initial_tm, reward_states_per_sim,
                                       n_simulations, max_d,
                                       all_states=True):
    """
    Compute learner–expert TM similarity per simulation,
    by averaging (in Fisher-z space) across distances.

    This is intended for robustness / generalization scatter plots.

    Returns:
        r_sim: shape (n_simulations,)
               One similarity value per agent/simulation.
    """
    corrs = np.zeros((n_simulations, max_d - 1))  # skip distance 0
    possible_distances = np.arange(1, max_d)

    for sim in range(n_simulations):

        # Optional: consider only visited states
        if not all_states:
            if expert_states is not None:
                agent_visited = np.unique(agent_states[sim][~np.isnan(agent_states[sim])].astype(int))
                expert_visited = np.unique(expert_states[sim][~np.isnan(expert_states[sim])].astype(int))
                visited_states = np.intersect1d(agent_visited, expert_visited)
            else:
                visited_states = np.unique(agent_states[sim][~np.isnan(agent_states[sim])].astype(int))

        initial_tm_actions = np.sum(initial_tm[sim], axis=1)
        distances_to_rewards = calculate_distances_to_rewards(
            initial_tm_actions,
            reward_states_per_sim[sim][:, 0].tolist()
        )

        for p in possible_distances:
            if all_states:
                state_list = [s for s, d in distances_to_rewards.items() if d == p]
            else:
                state_list = [s for s in visited_states if distances_to_rewards[s] == p]

            if state_list:
                tm_expert = expert_tm[sim][state_list].flatten()
                tm_agent  = agent_tm[sim][state_list].flatten()

                var_expert, var_agent = np.var(tm_expert), np.var(tm_agent)
                if var_expert == 0 or var_agent == 0:
                    corrs[sim, p - 1] = 0
                else:
                    r = np.corrcoef(tm_expert, tm_agent)[1, 0]
                    r = np.clip(r, -1 + 1e-9, 1 - 1e-9)
                    corrs[sim, p - 1] = r

    # ---- Collapse across distances (key difference) ----

    # Fisher z-transform
    z = np.arctanh(corrs)

    # Average across distances WITHIN each simulation
    z_sim = np.nanmean(z, axis=1)   # shape (n_simulations,)

    # Back to r-space
    r_sim = np.tanh(z_sim)

    return r_sim

def compute_true_value_function(true_tms, reward_info, discount_factor=0.99, theta=1e-6, max_iter=10000):
    """
    Generates true value function using Bellman equation
    Args:
        true_tms: List of true transition matrices for each simulation (n_simulations, n_states, n_actions, n_states)
        reward_info: List of reward states for each simulation (n_simulations, n_episodes, 2 (columns: state, reward))
        discount_factor: Discount factor for future rewards
        theta: Threshold for convergence
        max_iter: Maximum number of iterations
    Returns:
        List of true Q values for each simulation
    """
    n_simulations, n_states, n_actions, _ = np.array(true_tms).shape

    # Initialize true Q values
    true_q_values = np.ones((n_simulations, n_states, n_actions))

    for sim in range(n_simulations):
        # Get transition probabilities (same for all episodes)
        P = true_tms[sim]

        # Get reward per state 
        R = np.zeros(n_states, dtype=float)
        count = np.zeros(n_states, dtype=int) 
        is_goal = np.zeros(n_states, dtype=bool)
        for episode in reward_info[sim]: 
            for state, reward in episode:
                if reward > 0:  # Mark goal states
                    is_goal[int(state)] = True
                else:
                    reward = -1 # Moving cost
                R[int(state)] += reward 
                count[int(state)] += 1

        count[count == 0] = 1  # Avoid division by zero
        R = R/count # Average reward per state across episodes (because ofto account for noise in rewards)

        # Value iteration
        Q = np.ones((n_states, n_actions))

        for i in range(max_iter):
            delta = 0
            max_Q_next = np.max(Q, axis=1) # max_a' Q(s', a')
            max_Q_next[is_goal] = 0  # Set max Q for goal states to 0, as they are terminal states
            Q_new = np.sum(P * (R + discount_factor * max_Q_next), axis=2)  # Q*(s, a) = sum_s' P(s'|s,a) [R(s') + gamma * max_a' Q(s', a')]
            
            # Check convergence
            delta = max(delta, np.max(np.abs(Q_new - Q)))
            Q = Q_new
            if delta < theta:
                break

        true_q_values[sim] = Q
            
    return true_q_values
