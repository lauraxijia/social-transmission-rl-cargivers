import os
import numpy as np
import json
from scipy.optimize import differential_evolution

from models.mf_valueshaping import MFValueShapingAgent
from utils.helper_functions import run_simulations, aggregate_results

"""Optimization of the model-free social agent with value shaping using differential evolution."""

# Parameters
n_episodes = 20
n_simulations = 1000 # 1000
max_steps = 40 #40
n_calls = 15
training_split = 0.5
popsize = 5
objective = "action_gap"  # "q_mismatch" | "action_gap" | "cumulative_reward"

seed = 5
rng = np.random.default_rng(seed)


print(f"MF VS diff alg w/ max_steps {max_steps}, n_episodes {n_episodes}, n_simulations {n_simulations}, n_calls {n_calls}")

# Define the parameter space
param_search_space = [
    (-5, 5),    # Unbounded space for inverse temperature Real(-5, 5) - BETA
    (-10, 10),  # Unbounded space for discount factor Real(-10, 10)   - GAMMA
    (-5,5)      # Social parameter Real(-5,5)                         - KAPPA
]


## LOAD DATA ##
# Load learning parameters of the model-free asocial agent
with open('saved/opti_results/mfree_agent.json', 'r') as json_file:
    mb_agent_params = json.load(json_file)["opti_params"]
    alpha = mb_agent_params["alpha"]
# Load data from the expert
with open(f'saved/baseline/mbased_pedagogical_expert_{objective}_baseline.json', 'r') as json_file:
    expert_data= json.load(json_file)
# Convert the saved lists to array
for k in expert_data.keys():
    expert_data[k] = np.array(expert_data[k])

# Load the worlds 
loaded = np.load('saved/worlds.npz')
worlds = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

rewards_load = np.load('saved/rewards_info.npz')
rewards = [rewards_load[f'arr_{i}'] for i in range(len(rewards_load.files))]


## OPTIMIZATION ##
def objective_function(unbounded_params, AgentClass, expert_data,  worlds, rewards, n_simulations, max_steps, n_episodes, training_split, rng):

    # Bound parameters
    # Apply inverse transformations for continuous parameters
    beta = np.exp(unbounded_params[0])                    # For [0, +inf) bounded
    gamma = 1 / (1 + np.exp(-unbounded_params[1]))        # For [0, 1] bounded
    kappa_vs = np.exp(unbounded_params[2])                # For [0, +inf) bounded

    params  = {"beta": beta, "alpha": alpha, "gamma": gamma, "kappa":kappa_vs}


    rewards_result = np.zeros((n_simulations, n_episodes))
  

        
    rewards_result = run_simulations(
        AgentClass,
        params,
        expert_data, 
        rng,
        exp = "baseline",
        worlds = worlds, 
        rewards = rewards, 
        rewards_exp2 = None,
        n_simulations = n_simulations, 
        n_episodes = n_episodes, 
        max_steps = max_steps,
        training_split = training_split, 
        optimization = True
        )
    

    rewards_result = np.stack(rewards_result)

    return -np.mean(rewards_result[:, :int(n_episodes*training_split)])  


result = differential_evolution(objective_function, param_search_space, args=(MFValueShapingAgent, expert_data, worlds, rewards, n_simulations, max_steps, n_episodes, training_split, rng), maxiter=n_calls, popsize=popsize, disp=True)  


print("result.x, result.fun", result.x, result.fun)

## FINAL PARAMETER RETRIEVAL ##
# Retrieve the optimized parameters in the transformed space
unbounded_temp, unbounded_gamma, unbounded_kappa = result.x

# Apply inverse transformations to obtain the original bounded parameters
inverse_temp = np.exp(unbounded_temp)             # For [0, +inf) bounded
gamma = 1 / (1 + np.exp(-unbounded_gamma))        # For [0, 1] bounded
kappa_vs = np.exp(unbounded_kappa)                # For [0, +inf) bounded

print(f"Optimized parameters: beta = {inverse_temp}, alpha = {alpha}, gamma = {gamma}, kappa_vs = {kappa_vs}")
opti_params = {"beta": inverse_temp,
               "alpha": alpha,
               "gamma": gamma,
               "kappa": kappa_vs}

# If folder does not exist, create it
if not os.path.exists('saved/opti_results'):
    os.makedirs('saved/opti_results')
with open(f'saved/opti_results/mfree_vshaping_{objective}.json', 'w') as json_file:
    json.dump({'opti_params': opti_params, 'fun': result.fun}, json_file)