import os
import numpy as np
import json
import time

from scipy.optimize import differential_evolution
from utils.social_functions import social_sim_mb
from models.mb_dbias import mb_policy

"""Optimization of the model-based social agent with decision bias using differential evolution."""

# Parameters
n_episodes = 20
n_simulations = 1000 #1000
max_steps = 40 
n_calls = 15 # maximum number of generations in differential evolution
training = 0.5 # proportion of episodes used for training
popsize = 5 # population size for differential evolution

seed = 5
rng = np.random.default_rng(seed) 

print(f"MB DB diff alg w/ max_steps {max_steps}, n_episodes {n_episodes}, n_simulations {n_simulations}, n_calls {n_calls}")

# Define parameter space
param_search_space = [
    (-5, 5),    # Unbounded space for inverse temperature Real(-5, 5)  – BETA
    (-10, 10),  # Unbounded space for discount factor Real (-10, 10)   – GAMMA
    (1, 40),    # Categorical (integer) space for number of steps      - LAMBDA
    (-10, 10)   # Social policy parameter                              - OMEGA
]

## LOAD DATA ##
# Load learning parameters of the model-based asocial agent
with open('saved/opti_results/mbased_agent.json', 'r') as json_file:
    mb_agent_params = json.load(json_file)["opti_params"]
    alpha = mb_agent_params["alpha"]
    alpha_t = mb_agent_params["alpha_t"]
# Load data from the expert
with open('saved/baseline/mbased_expert_baseline.json', 'r') as json_file:
    expert_data= json.load(json_file)
# Convert the saved lists to array
for k in expert_data.keys():
    expert_data[k] = np.array(expert_data[k])

# Load saved worlds and rewards
loaded = np.load('saved/worlds.npz')
worlds_saved = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

rewards_load = np.load('saved/rewards_info.npz')
rewards_shuffled = [rewards_load[f'arr_{i}'] for i in range(len(rewards_load.files))]

## OPTIMIZATION ##
def objective_function(unbounded_params, 
                       mb_policy, expert_data,  
                       worlds_saved, rewards_shuffled, 
                       n_simulations, max_steps, n_episodes, training, 
                       rng):
    
    # Bounded parameters
    # Apply inverse transformations for continuous parameters
    beta = np.exp(unbounded_params[0])                            # For [0, +inf) bounded
    gamma = 1 / (1 + np.exp(-unbounded_params[1]))                # For [0, 1] bounded
    lambda_mean = unbounded_params[2]                             # For [0, 40) bounded
    omega = 1 / (1 + np.exp(-unbounded_params[3]))                # For [0, 1] bounded

    rewards_result = np.zeros((n_simulations, n_episodes))

    params  = {"beta": beta, 
               "alpha": alpha, 
               "gamma": gamma, 
               "lambda": lambda_mean, 
               "alpha_t": alpha_t, 
               "omega":omega}

 
    rewards_result = social_sim_mb(mb_policy, 
                                   expert_data, 
                                   worlds_saved, 
                                   rewards_shuffled, 
                                   n_simulations, 
                                   max_steps, 
                                   n_episodes, 
                                   params, 
                                   training, 
                                   rng, 
                                   optimization = True,
                                   world_model='baseline',
                                   rewards_exp2 = None)
    

    return -np.mean(rewards_result[:, :int(n_episodes*training)])  

result = differential_evolution(objective_function, param_search_space, 
                                args=(mb_policy, 
                                      expert_data, 
                                      worlds_saved, 
                                      rewards_shuffled, 
                                      n_simulations, 
                                      max_steps, 
                                      n_episodes, 
                                      training, 
                                      rng), 
                                maxiter=n_calls, 
                                popsize=popsize, 
                                disp=True,
                                rng=rng) 


print("result.x, result.fun", result.x, result.fun)
print("result.message:", result.message) 

## FINAL PARAMETER RETRIEVAL ##
# Retrieve the optimized parameters in the transformed space
unbounded_temp, unbounded_gamma, unbounded_lambda, unbounded_omega = result.x

# Apply inverse transformations to obtain the original bounded parameters
inverse_temp = np.exp(unbounded_temp)           # For [0, +inf) bounded 
gamma = 1 / (1 + np.exp(-unbounded_gamma))      # For [0, 1] bounded
omega = 1 / (1 + np.exp(-unbounded_omega))      # For [0, 1] bounded
lambda_mean = unbounded_lambda                  # For [0, 40) bounded

print(f"Optimized parameters: beta = {inverse_temp}, alpha = {alpha}, gamma = {gamma}, lambda = {lambda_mean}, alpha_t = {alpha_t}, omega = {omega}")
opti_params = {"beta": inverse_temp,
               "alpha": alpha,
               "gamma": gamma,
               "lambda": lambda_mean,
               "alpha_t": alpha_t,
               "omega": omega}

# If folder does not exist, create it
if not os.path.exists('saved/opti_results'):
    os.makedirs('saved/opti_results')
with open('saved/opti_results/mbased_dbias.json', 'w') as json_file:
    json.dump({'opti_params': opti_params, 'fun': result.fun}, json_file)