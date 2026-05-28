import os
import numpy as np
import json
from tqdm import tqdm

from utils.world import VillageWorld
from models.mf_expert import expert_ql
from utils.plot_functions import plot_performance

"""Simulation of the model-free expert and agent."""

# Parameters
save = True
agent = "agent"  # "expert" or "agent"
n_episodes = 20  #120
world_model = "exp3" #baseline" #"exp3" # "exp2"
training = 0.5
n_states = 100
n_actions = 4
n_simulations = 1000
max_steps = 40 

seed = 5
rng = np.random.default_rng(seed)

## LOAD DATA ##
if agent == "expert":
    print("Params expert")
    # Load parameters
    with open(f'saved/opti_results/mfree_{agent}.json', 'r') as json_file:
        params = json.load(json_file)["opti_params"]
else:
    print("Params agent")
    # Load parameters
    with open(f'saved/opti_results/mfree_{agent}.json', 'r') as json_file:
        params = json.load(json_file)["opti_params"]
print(params)        

# Load worlds and rewards
loaded = np.load('saved/worlds.npz')
worlds_saved = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

rewards_load = np.load('saved/rewards_info.npz')
rewards_shuffled = [rewards_load[f'arr_{i}'] for i in range(len(rewards_load.files))]

if world_model == "exp2":
    rewards_load_exp2 = np.load('saved/rewards_exp2.npz')
    rewards_exp2 = [rewards_load_exp2[f'arr_{i}'] for i in range(len(rewards_load_exp2.files))]
else:
    rewards_exp2 = None

# Load agent function
agent_function = expert_ql

# INITIALIZE STORAGE ARRAYS ##

# total reward for each episode
rewards_result_epi_saved = np.zeros((n_simulations, n_episodes))
# Save the reward for each step
rewards_result_steps = np.zeros((n_simulations, n_episodes))
# N* of steps until agent finds a reward
steps_saved = np.zeros((n_simulations, n_episodes))
# Final value per simulation
value_saved = np.zeros((n_simulations,n_states,n_actions ))
# States expert goes through - mainly to check results
states_saved = np.zeros((n_simulations, n_episodes, max_steps +1 ))
# Actions taken by the expert - Policy
actions_saved = np.zeros((n_simulations, n_episodes, max_steps))
# Value per episode
value_epi_saved = []


### SIMULATION LOOP ##
print(f"MF Agent simulations with {world_model} world model")

for sim in tqdm(range(n_simulations)):
    # Initialize the world
    env = VillageWorld(worlds_saved[sim], rng)
    
    final_value, reward_sums_epi, state_mat, action_mat, steps_to_reward, value_epi, reward_sums_steps = agent_function(params, 
                                                                                                                        env, 
                                                                                                                        worlds_saved[sim], 
                                                                                                                        rewards_shuffled[sim], 
                                                                                                                        max_steps, 
                                                                                                                        n_episodes, 
                                                                                                                        training, 
                                                                                                                        rng, 
                                                                                                                        optimization=None, 
                                                                                                                        agent = agent, 
                                                                                                                        world_model=world_model,
                                                                                                                        rewards_exp2=rewards_exp2[sim] if world_model == "exp2" else None)
                                                                                                                    
    
    # Total rewards per episode
    rewards_result_epi_saved[sim, :] = reward_sums_epi
    steps_saved[sim,:] = steps_to_reward
    value_saved[sim, :, :] = final_value
    states_saved[sim, :, :] = state_mat
    value_epi_saved.append(value_epi)
    actions_saved[sim, :, :] = action_mat
    rewards_result_steps[sim, :] = np.sum(reward_sums_steps, axis = 1)



## SAVE DATA ##
data_mf = {"sum_rewards": rewards_result_epi_saved.tolist(), 
                "steps_to_reward": steps_saved.tolist(), 
                "value": value_saved.tolist(), 
                "states_saved": states_saved.tolist(),
                "actions_saved": actions_saved.tolist(), 
                "reward result steps": rewards_result_steps.tolist()
                }

if save: 
    # Create directory if it doesn't exist
    saving_path = f'saved/{world_model}'
    os.makedirs(saving_path, exist_ok=True)

    with open(f'saved/{world_model}/mfree_{agent}_{world_model}.json', 'w') as json_file:
        json.dump(data_mf, json_file, indent=4)
    np.savez_compressed(f'saved/{world_model}/{agent}_mfree_value_epi.npz', *value_epi_saved)
    print(f"MF {agent} data saved")

    ## PLOT PERFORMANCE ##
    title = f"MF {agent} {n_simulations} sim"
    fig1, ax1 = plot_performance(rewards_result_epi_saved, "Episodes", title , "Performance", None)
    fig1.savefig(f'saved/figures/{world_model}/mf_{agent}_{world_model}_performance.png', bbox_inches='tight')
    fig2, ax2 = plot_performance(steps_saved, "Episodes", title, "Steps to reward", None)
    fig2.savefig(f'saved/figures/{world_model}/mf_{agent}_{world_model}_steps.png', bbox_inches='tight')
    print(f"Figures saved for MF {agent}")






