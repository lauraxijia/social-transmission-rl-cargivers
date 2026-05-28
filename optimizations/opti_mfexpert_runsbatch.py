import json
import numpy as np
import os
from utils.world import VillageWorld
from models.mf_expert import expert_ql
import time
from scipy.optimize import differential_evolution


"""
Optimization of the model-free expert/asocial model using differential evolution.

Note: we only use the agent version of the model-free here.

For expert, set agent = expert and n_episodes = 120.
For agent, set agent = agent and n_episodes = 20.
"""

# Parameters
agent = "agent"  # "expert" or "agent"
n_episodes = 20 #120
max_steps = 40 #40
n_simulations = 1000  #1000
n_calls = 15
n_popsize = 5
training = 1

seed = 5
rng =np.random.default_rng(seed)


print(f"Model free diff algorithm w/ max_steps {max_steps}, n_episodes {n_episodes}, n_simulations {n_simulations}, n_calls {n_calls} for {agent}")



# Define parameter space
param_search_space = [
    (-5, 5),    # Unbounded space for inverse temperature Real(-5, 5)  - BETA
    (-10, 10),  # Unbounded space for learning rate 1 Real(-10, 10)    - ALPHA
    (-10, 10)   # Unbounded space for discount factor Real(-10, 10)    - GAMMA
]

## LOAD DATA ##
loaded = np.load('saved/worlds.npz')
worlds_saved = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

rewards_load = np.load('saved/rewards_info.npz')
rewards_shuffled = [rewards_load[f'arr_{i}'] for i in range(len(rewards_load.files))]


## OPTIMIZATION ##

def objective_function(unbounded_params, worlds_saved, rewards_shuffled, max_steps, n_episodes, n_simulations, rng):
    # Must be in the form f(x, *args), where x is the argument in the form of a 1-D array and args is a 
    # tuple of any additional fixed parameters needed to completely specify the function
    # Computes the mean reward over n_simulations for 1 set of parameters

    # Bound parameters
    # Apply inverse transformations for continuous parameters
    beta = np.exp(unbounded_params[0])                    # For [0, +inf) bounded
    alpha = 1 / (1 + np.exp(-unbounded_params[1]))        # For [0, 1] bounded
    gamma = 1 / (1 + np.exp(-unbounded_params[2]))        # For [0, 1] bounded


    rewards_saved = np.zeros((n_simulations, n_episodes))
    params  = {"beta": beta, "alpha": alpha, "gamma": gamma}
    for sim in range(n_simulations):
        
        env = VillageWorld(worlds_saved[sim], rng)

        rewards_saved[sim] = expert_ql(params, 
                                       env, 
                                       worlds_saved[sim], 
                                       rewards_shuffled[sim], 
                                       max_steps, 
                                       n_episodes, 
                                       training,
                                       rng=rng, 
                                       optimization=True,
                                       agent=agent,
                                       world_model='baseline',
                                       rewards_exp2=None)
    

    return -np.mean(rewards_saved) 

result_time = time.time()
result = differential_evolution(objective_function, 
                                param_search_space, 
                                args=(worlds_saved,
                                      rewards_shuffled, 
                                      max_steps, 
                                      n_episodes, 
                                      n_simulations, 
                                      rng), 
                                maxiter=n_calls, 
                                popsize=n_popsize, 
                                disp=True,
                                rng=rng)  
result_timefinal = (time.time() - result_time)/60
print("The optimization takes", result_timefinal, "seconds")


print("result.x, result.fun", result.x, result.fun)

## FINAL PARAMETER RETRIEVAL ##

# Retrieve the optimized parameters in the transformed space
unbounded_temp, unbounded_alpha, unbounded_gamma = result.x

# Apply inverse transformations to obtain the original bounded parameters
inverse_temp = np.exp(unbounded_temp)             # For [0, +inf) bounded
alpha = 1 / (1 + np.exp(-unbounded_alpha))        # For [0, 1] bounded
gamma = 1 / (1 + np.exp(-unbounded_gamma))        # For [0, 1] bounded

print(f"Optimized parameters: beta = {inverse_temp}, alpha = {alpha}, gamma = {gamma}")
opti_params = {
    "beta": inverse_temp,
    "alpha": alpha,
    "gamma": gamma
}

# If folder does not exist, create it
if not os.path.exists('saved/opti_results'):
    os.makedirs('saved/opti_results')
np.savez(f'saved/opti_results/mfree_{agent}.npz', x=result.x, fun=result.fun)
with open(f'saved/opti_results/mfree_{agent}.json', 'w')as json_file:
    json.dump({'opti_params': opti_params, 'fun': result.fun}, json_file)
