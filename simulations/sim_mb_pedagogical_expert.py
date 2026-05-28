import numpy as np
import json
import tqdm
import os

from models.mb_pedagogical_expert import mb_pedagogical_expert
from models.mf_valueshaping import MFValueShapingAgent
from utils.plot_functions import plot_performance
from utils.world import VillageWorld

"""Simulation of model-based pedagogical expert."""

# Simulation parameters
save = True
optimization = False
world_model = "baseline" # "baseline" "exp2" "exp3"
n_states = 100
n_simulations = 1 # 1000
n_train_episodes = 100
n_teach_episodes = 20
n_episodes_total = n_train_episodes + n_teach_episodes 
max_steps = 40
n_actions = 4
learner = "MF-VS"
n_learner = 10

# Set random seed for reproducibility
seed = 5
rng = np.random.default_rng(seed)

## LOAD DATA ##
# Load parameters
print("Params pedagogical expert:")
with open(f'saved/opti_results/mbased_pedagogical_expert.json', 'r') as json_file:
    params = json.load(json_file)["opti_params"]
print(params)

# Load worlds and rewards
loaded = np.load('saved/worlds.npz')
worlds_saved = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

rewards_load = np.load('saved/rewards_info.npz')
rewards_shuffled = [rewards_load[f'arr_{i}'] for i in range(len(rewards_load.files))]

# Load learner class and params
if learner == "MF-VS":
    learner_function = MFValueShapingAgent
    with open(f'saved/opti_results/mf_valueshaping.json', 'r') as json_file:
        learner_params = json.load(json_file)["opti_params"]

# TODO: Is there anything else we want to store?

## INITIALIZE STORAGE ARRAYS ##
# Total reward for each episode
rewards_result_epi_saved = np.zeros((n_simulations, n_episodes_total))
# Save the reward for each step
rewards_result_steps = np.zeros((n_simulations, n_episodes_total))
# N* of steps until agent finds a reward
steps_saved = np.zeros((n_simulations, n_episodes_total))
# Final value per simulation
value_saved = np.zeros((n_simulations,n_states,n_actions ))
# States expert goes through - mainly to check results
states_saved = np.zeros((n_simulations, n_episodes_total, max_steps +1 ))
# Actions taken by the expert - Policy
actions_saved = np.zeros((n_simulations, n_episodes_total, max_steps))
# Model saved
model_saved = np.zeros((n_simulations, n_states, n_actions, 2)) 
# Final tm of each simulation
tm_saved = []
# Value per episode
value_epi_saved = []
# Tm every 5 episodes
tm_epi_saved = []

# Load agent function
agent_function = mb_pedagogical_expert

## SIMULATION LOOP ##
print(f"MB pedagogical expert simulations with {world_model} world model")
for sim in tqdm(range(n_simulations)):
    env = VillageWorld(worlds_saved[sim], rng)
    final_value, reward_sums_epi, state_mat, action_mat, steps_to_reward, tm_final, model_r, value_epi, tm_epi, reward_sums_steps  = agent_function(params, 
                                                                                                                                                    env, 
                                                                                                                                                    worlds_saved[sim], 
                                                                                                                                                    rewards_shuffled[sim], 
                                                                                                                                                    max_steps, 
                                                                                                                                                    n_episodes_total, 
                                                                                                                                                    n_teach_episodes,
                                                                                                                                                    rng, 
                                                                                                                                                    optimization = False,
                                                                                                                                                    world_model = world_model,
                                                                                                                                                    learner_function = learner_function,
                                                                                                                                                    n_learner = n_learner)
    
    rewards_result_epi_saved[sim] = reward_sums_epi
    steps_saved[sim,:] = steps_to_reward
    actions_saved[sim,:] = action_mat
    value_saved[sim, :, :] = final_value
    states_saved[sim, :, :] = state_mat
    model_saved[sim, :, :, :] = model_r
    tm_saved.append(tm_final)
    value_epi_saved.append(value_epi)
    tm_epi_saved.append(tm_epi)


## PLOT PERFORMANCE ##
title = f"MB pedagogical expert {n_simulations} sim"
fig1, ax1 = plot_performance(rewards_result_epi_saved, "Episodes", title , "Performance", None)
fig2, ax2 = plot_performance(steps_saved, "Episodes", title, "Steps to reward", None)

## SAVE DATA ##
data = {"sum_rewards": rewards_result_epi_saved.tolist(), 
        "steps_to_reward": steps_saved.tolist(), 
        "value": value_saved.tolist(), 
        "states_saved": states_saved.tolist(),
        "actions_saved": actions_saved.tolist(),
        "model_saved": model_saved.tolist()
        }

if save:
    # Create directory if it doesn't exist
    saving_path = f'saved/{world_model}'
    os.makedirs(saving_path, exist_ok=True)
    os.makedirs(f'{saving_path}/tmss', exist_ok=True)

    with open(f'saved/{world_model}/mbased_pedagogical_expert_{world_model}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    np.savez_compressed(f'saved/{world_model}/tmss/mbased_pedagogical_expert_tm.npz', *tm_saved)
    np.savez_compressed(f'saved/{world_model}/mbased_pedagogical_expert_value_epi.npz', *value_epi_saved)
    np.savez_compressed(f'saved/{world_model}/tmss/mbased_pedagogical_expert_tm_epi.npz', *tm_epi_saved)
    print(f"MB pedagogical expert data saved")

    # Save figures
    folder_path = os.path.join('saved', 'figures', str(world_model))
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    fig1.savefig(os.path.join(folder_path, f'mb_pedagogical_expert_{world_model}_performance.png'), bbox_inches='tight')
    fig2.savefig(os.path.join(folder_path, f'mb_pedagogical_expert_{world_model}_steps.png'), bbox_inches='tight')
    print(f"Figures saved for MB pedagogical expert")