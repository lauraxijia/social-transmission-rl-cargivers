import os

import numpy as np
from tqdm import tqdm
from .world import VillageWorld
from scipy.special import softmax
import json
from numba import njit

"""Helperfunctions for the simulations."""

def run_simulations(
    AgentClass,
    params,
    expert_data,
    rng,
    exp,
    worlds,
    rewards,
    rewards_exp2,
    n_simulations,
    n_episodes,
    max_steps,
    training_split,
    optimization
):
    """Run multiple independent simulations using the agent class."""

    results = []

    for sim in tqdm(range(n_simulations), disable=optimization):
        env = VillageWorld(worlds[sim], rng)

        agent = AgentClass(
            rng=rng,
            params=params,
            env=env,
            world=worlds[sim],
            rewards_info=rewards[sim],
            rewards_exp2=rewards_exp2[sim] if rewards_exp2 else None,
            expert_states=expert_data['states_saved'][sim],
            expert_actions=expert_data['actions_saved'][sim],
            n_episodes=n_episodes,
            training_split=training_split,
            max_steps=max_steps,
            optimization=optimization,
            exp=exp
        )

        sim_result = agent.run_full_simulation()
        results.append(sim_result)

    return results

def aggregate_results(results):
    """Aggregate results from multiple simulations."""
    data = {
        "sum_rewards": np.stack([r['reward_sum_episode'] for r in results]),
        "steps_to_reward": np.stack([r['steps_to_reward'] for r in results]),
        "values": np.stack([r['final_values'] for r in results]),
        "states": np.stack([r['states'] for r in results]),
        "actions": np.stack([r['actions'] for r in results]),
        "value_snapshots": np.stack([r['value_snapshots'] for r in results]),
    }
    return data

def save_results(results, exp, agent, expert_string):
    """Save the results of the simulations to a json file."""
    # Create directory if it doesn't exist
    folder = f'saved/{exp}'
    os.makedirs(folder, exist_ok=True)

    json_safe = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items() if k != "value_snapshots"}

    # Save data to json file
    with open(os.path.join(folder, f'{agent}_{expert_string}_{exp}.json'), 'w') as json_file:
        json.dump(json_safe, json_file, indent=4)
    
    # Save value snapshots to npz file
    np.savez_compressed(f'{folder}/{agent}_{expert_string}_{exp}_values_epi.npz', *results["value_snapshots"])

    print(f"Results saved to {folder}.")


#@njit
def softmax_policy(value, state, n_actions, beta, rng):
    """
    Softmax policy for action selection

    value: Value function (n_states, n_actions)
    state: Current state (int)
    n_actions: Number of actions (int)
    beta: Inverse temperature (float)

    Returns:
    tuple: (pi, action) where pi is the probability distribution over actions and action is the selected action
    """
    rescaled_value = value[state] - np.max(value[state])
    pi = softmax(rescaled_value * beta)

    action = rng.choice(n_actions, p=pi)
    return pi, action
 
 
#@njit
def find_reward(state, reward_placed):
    """
    Find the reward value of the current state
    If it is not a reward state with reward > 0, return -1

    reward states are defined from reward_placed
    """

    if (state in reward_placed[:, 0]) and (reward_placed[reward_placed[:,0] == state, 1][0] > 0):
        reward = reward_placed[:,1][np.where(reward_placed[:,0] == state)[0]][0]
        
    else:
        reward = -1
    
    return reward

def assign_fixed_rewards(env, rng):
    """
    Randomly assign fixed reward values for each reward state. 
    Args:
        env: VillageWorld object
        rng: np.random.Generator instance
    Returns:
        np.array: Array of shape (n_reward_states, 2) with the reward values
                  for each reward state -> [reward_state, reward_value]
    """
    reward_values = np.array([0, 25, 50, 100]) 
    shuffled_values = rng.choice(reward_values, len(env.reward_states), replace = False)
    reward_states = np.array(env.reward_states)
    return np.column_stack([reward_states, shuffled_values])


def sample_reward_offset(target_var):
    """
    Sample an integer reward offset with mean 0 and variance = target_var,
    using a shifted Binomial(n, 0.5).
    """
    assert target_var > 0, "target_var must be positive" # ensure that variance is positive

    #1. We want to sample Y ~ Binomial(n, p=0.5), so first we need to define n based on our target_var
    # Var = n/4, so n ≈ 4 * target_var
    n = 2 * int(round(2 * target_var))  # ensures n is even (which is necessary because later, we divide n//2)
    n = max(n, 2)  # Adding a catch with a minimum n to avoid degenerate cases
    p = 0.5 #ensures symmetry of the distribution

    #2. Now we can sample y
    y = np.random.binomial(n=n, p=p)
    # Center to mean 0 by subtracting n/2 from y
    x = y - n // 2  # n is even so this is integer

    # Now we have x, which has a mean of 0 and variance of n/4
    return x


@njit
def q_learning(value, state, action, next_state, reward, gamma, alpha):
    """
    Q-learning update rule
    value: Value function (n_states, n_actions)
    state: Current state (int)
    action: Action taken (int)
    next_state: Next state (int)
    reward: Reward received (float)
    gamma: Discount factor (float) - needs to be introduced as a parameter not the whole dictionary
    alpha: Learning rate (float) - needs to be introduced as a parameter not the whole dictionary

    Returns:
    np.array: Updated value function
    """
    # LEARNING - Q LEARNING
    # update value function of previous state
    # 1. Find the value of the current state
    q = value[state, action]
    
    # 2. Find the maximum value of the next_state
    if next_state is None:
        #print("Next state is None")
        max_next_q = 0
    else:
        max_next_q = np.max(value[next_state])
    
    
    # 3. Reward prediction error
    delta = reward + (gamma * max_next_q)  - q
    #print("reward", reward, "gamma", gamma, "max_next_q", max_next_q, "q", q, "delta", delta)
    
    # 4. Update the value of the current state-action pair
    value[state, action] = q + alpha * delta
    
    return value

def decision_bias(env, exp_states, 
                    agent_location, agent_state, 
                    value, params, world, 
                    episode, t, reward_placed, 
                    rng, modelbased, transition_belief):
    
    social_p = rng.random() < params['omega']

    # social policy
    pi_soc = social_policy(env, world, 
                             exp_states, episode, t, 
                             agent_state, agent_location, reward_placed, modelbased, transition_belief)
    # asocial policy
    pi_asoc, _ = softmax_policy(value, agent_state, 
                               env.n_actions, params['beta'], rng)

    # compute mixed policy
    pi_mixed = (1-social_p) * pi_asoc + social_p * pi_soc
    assert np.isclose(np.sum(pi_mixed), 1), f"pi_mixed: {pi_mixed}"

    # sample action 
    action = rng.choice(np.arange(env.n_actions), p=pi_mixed) # controlled by the random number generator

    return action 


def value_shaping(value, expert_action, expert_state, kappa):

    """
    Adds a bonus to the value of the agent for the expert's observed state and action
    value: Value function (n_states, n_actions)
    expert_action: Action taken by the expert (int)
    expert_state: State where the expert is (int)
    kappa: Value shaping parameter (float) - how much the agent should value the expert's action in the expert's state
    n_actions = 4

    Returns:
    np.array: Updated value function
    """

    expert_state = int(expert_state)
    expert_action = int(expert_action) 
    
    value[expert_state, expert_action] += kappa

    return value


def model_update(model_r, state, action, reward, next_state):
    """
    Updates the model for model-based learning
    model_r: Model of the environment (n_states, n_actions, 2)
    state: Current state (int)
    action: Action taken (int)
    reward: Reward received (float)
    next_state: Next state (int)

    Returns:
    np.array: Updated model
    """
    # Update the Next state of the model 
    model_r[state, action, 0] = reward
    if next_state is None:
        model_r[state, action, 1] = np.nan
    else:
        model_r[state, action, 1] = next_state
    return model_r


@njit
def dynaq_planner(value, model_r, state, action, next_state, n_steps, gamma, alpha, rng):
    """
    Dyna-Q planner for model-based learning
    value: Value function (n_states, n_actions)
    model_r: Model of the environment (n_states, n_actions, 2)
    state: Current state (int)
    action: Action taken (int)
    next_state: Next state (int)
    n_steps: Number of planning steps (int)
    gamma: Discount factor (float)
    alpha: Learning rate (float)

    Returns:
    np.array: Updated value function
    """
    
    ## PLANNING
    for p in range(n_steps):
        # Randomly select s & a from previously visited state

        seen_tuples = np.where(~np.isnan(model_r[:,:,1])) # np.array().T
        #idx = np.random.choice((len(seen_tuples[0])))
        idx = rng.integers(0, len(seen_tuples[0])) # controlled by the random number generator

        state, action = seen_tuples[0][idx], seen_tuples[1][idx]
 
        reward, next_state = model_r[state, action][0], int(model_r[state, action][1])    

        value = q_learning(value, state, action, next_state, reward, gamma, alpha) 
        
    return value

def load_data(name):
    """
    Load data from a json file
    name: Name of the file (str)
    """
    # Load data from the expert
    with open(f'{name}.json', 'r') as json_file:
        data = json.load(json_file)
    # Convert the saved lists to array
    arrays = [np.array(data[k]) for k in data.keys()]
    return tuple(arrays)


def compute_expected_distance_to_expert(
    expert_state, transition_belief, n_states, max_iter=200, tol=1e-4
):
    """
    Value iteration: V(s) = min_a Σ_{s'} P(s'|s,a) * (1 + V(s')) with V(expert_state) = 0
    Returns the expected distance from each state to the expert state based on the agent's transition belief.
    """
    V = np.full(n_states, np.inf)
    V[expert_state] = 0.0 # Agent reached expert state, so distance is 0

    for _ in range(max_iter):
        with np.errstate(invalid='ignore'):
            # (n_states, n_actions, n_states) * (n_states,) → (n_states, n_actions)
            expected_values = np.nansum(transition_belief * (1 + V), axis=2) # shape (n_states, n_actions)

        V_new = np.nanmin(expected_values, axis=1) # shape (n_states,)
        V_new[expert_state] = 0.0 

        finite_mask = np.isfinite(V_new) & np.isfinite(V)
        delta = np.max(np.abs(V_new[finite_mask] - V[finite_mask])) if np.any(finite_mask) else np.inf

        V = V_new
        if delta < tol:
            break

    return V

#@njit #doesnt work w/ njit bc Numba cannot handle costum pytjon classes (here: env)
def social_policy(env, world, exp_states, episode, t, agent_state, agent_location, reward_placed, modelbased, transition_belief):
    """
    Finds the action that reduces the distance from agent's to expert's state

    env: Environment object
    world: World matrix (n_rows, n_cols)
    exp_states: Expert's states (n_agent_episodes, max_steps) 
    episode: Current episode (int)
    t: Current step (int)
    agent_state: Agent's state (int)
    agent_location: Agent's location (tuple)
    modelbased: Boolean indicating whether the agent is model-based or model-free (bool)
    transition_belief: Transition belief of the agent (n_states, n_actions, n_states) - only used for model-based agents

    Returns:
    np_array: Prob distr pi_social, where 1 is assigned to action reducing distance to expert
    """

    # FIND THE LOCATION OF THE EXPERT
    if  ~np.isnan(exp_states[episode, t]):
        
        expert_state = int(exp_states[episode, t])
        # Find the location of the expert - has to be this way for Euclidean distance
        expert_location = np.column_stack(np.where(world == exp_states[episode, t])).reshape(2,)
      
    # If exper has found the reward - Find the last location
    else:
        last_loc_t = (~np.isnan(exp_states[episode, :])).cumsum().argmax()
        expert_state = int(exp_states[episode, last_loc_t])
        expert_location = np.column_stack(
            np.where(world == exp_states[episode, last_loc_t])
            ).reshape(2,)

    # CALCULATE THE NUMBER OF STEPS TO THE EXPERT
    dist = np.zeros(4)
    # Return new location based on the action

    if modelbased:
        n_states = transition_belief.shape[0]

        # Expected distance from every state to expert
        V = compute_expected_distance_to_expert(expert_state, transition_belief, n_states)

        for a in range(4):
            probs = transition_belief[agent_state, a]

            if np.sum(probs) == 0: # Action not possible or unknown
                dist[a] = np.inf
                continue

            with np.errstate(invalid='ignore'): # Ignore warnings for inf values in V
             dist[a] = np.nansum(probs * V)
    else:
        # Don't consider walls or boundaries
        for a in range(4):
            next_agent_location = env.get_naive_next_location(a, agent_location)
            dist[a] = (np.abs(expert_location[0] - next_agent_location[0]) + 
                        np.abs(expert_location[1] - next_agent_location[1]))


    # Choose the action with the minimum distance to the expert
    action = np.argmin(dist)


    # Create one-hot probability distribution over all actions (for mixing of policies)
    pi_social = np.zeros(4)
    pi_social[action] = 1
    return pi_social


def teaching_objective(method, **ctx):
    """Dispatcher for the pedagogical teacher's action-selection objective."""
    if method == "q_mismatch":
        return _obj_q_mismatch(**ctx)
    if method == "action_gap":
        return _obj_action_gap(**ctx)
    if method == "cumulative_reward":
        # Stub: full-trajectory rollout objective not yet implemented.
        teacher_Q = ctx["teacher_Q"]
        expert_state = ctx["expert_state"]
        return int(np.argmax(teacher_Q[expert_state]))
    raise ValueError(f"Unknown teaching objective: {method}")


def _obj_q_mismatch(predictions_at_t, learner_candidates, teacher_Q, **_):
    """Objective 2: pick a_T that minimizes the average L2 distance between
    Q_E(s_l, ·) and Q_l(s_l, ·) across learner candidates, evaluated at each
    learner's current state s_l after the hypothetical demo (s_t, a_T)."""
    scores = {}
    for a_T, by_learner in predictions_at_t.items():
        diffs = []
        for l_id, sim in by_learner.items():
            s = learner_candidates[l_id].current_state
            diffs.append(np.linalg.norm(teacher_Q[s] - sim['Q'][s]))
        scores[a_T] = float(np.mean(diffs))
    return int(min(scores, key=scores.get))


def _obj_action_gap(predictions_at_t, learner_candidates, teacher_Q, **_):
    """Objective 3: pick a_T that maximizes the average value gap
    Q_l(s_l, a*) - Q_l(s_l, â), where a* = argmax_a Q_E(s_l, a) is the
    expert-optimal action at the learner's state and â is the action the
    learner would sample after observing the hypothetical demo (s_t, a_T)."""
    scores = {}
    for a_T, by_learner in predictions_at_t.items():
        gaps = []
        for l_id, sim in by_learner.items():
            s = learner_candidates[l_id].current_state
            a_star = int(np.argmax(teacher_Q[s]))
            a_hat = int(sim['action'])
            Q_l = sim['Q']
            gaps.append(float(Q_l[s, a_star] - Q_l[s, a_hat]))
        scores[a_T] = float(np.mean(gaps))
    return int(max(scores, key=scores.get))

