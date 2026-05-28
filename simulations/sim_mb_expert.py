import numpy as np
import json
from tqdm import tqdm
import os

from utils.world import VillageWorld
from models.mb_expert import mb_expert
from utils.plot_functions import plot_performance


"""Simulation of the model-based expert and agent."""

# Simulation parameters
save = True
agent = "expert"  # "expert" or "agent"
optimization = False
world_model = "baseline" # "baseline" "exp2" "exp3"
training = 0.5 
n_states = 100
n_simulations = 1000 # 1000
n_episodes = 120 # 120 for expert, 20 for agent
max_steps = 40
n_actions = 4

seed = 5
rng = np.random.default_rng(seed)

## LOAD DATA ##
if agent == "expert":
    print("Params expert:")
    # Load parameters
    with open(f'saved/opti_results/mbased_{agent}.json', 'r') as json_file:
        params = json.load(json_file)["opti_params"]
    print(params)
else:
    print("Params agent")
    # Load parameters
    with open(f'saved/opti_results/mbased_{agent}.json', 'r') as json_file:
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

## INITIALIZE STORAGE ARRAYS ##

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
# Model saved
model_saved = np.zeros((n_simulations, n_states, n_actions, 2)) 
# Final tm of each simulation
tm_saved = []
# Value per episode
value_epi_saved = []
# Tm every 5 episodes
tm_epi_saved = []

# Load agent function
agent_function = mb_expert

## SIMULATION LOOP ##
print(f"MB {agent} simulations with {world_model} world model")
for sim in tqdm(range(n_simulations)):

    env = VillageWorld(worlds_saved[sim], rng)
    if optimization:
        reward_sums_epi = agent_function(params, 
                                    env, 
                                    worlds_saved[sim], 
                                    rewards_shuffled[sim], 
                                    max_steps, 
                                    n_episodes, 
                                    training,
                                    rng,
                                    agent,
                                    optimization = optimization,
                                    world_model = world_model,
                                    rewards_exp2=rewards_exp2[sim] if world_model == "exp2" else None)
        
    else:
        final_value, reward_sums_epi, state_mat, action_mat, steps_to_reward, tm_final, model_r, value_epi, tm_epi, reward_sums_steps  = agent_function(params, 
                                                                                                                                                        env, 
                                                                                                                                                        worlds_saved[sim], 
                                                                                                                                                        rewards_shuffled[sim], 
                                                                                                                                                        max_steps, 
                                                                                                                                                        n_episodes, 
                                                                                                                                                        training,
                                                                                                                                                        rng, 
                                                                                                                                                        agent,
                                                                                                                                                        optimization = optimization,
                                                                                                                                                        world_model = world_model,
                                                                                                                                                        rewards_exp2=rewards_exp2[sim] if world_model == "exp2" else None)
                                                                                                                                                       



    

    rewards_result_epi_saved[sim] = reward_sums_epi
    steps_saved[sim,:] = steps_to_reward
    actions_saved[sim,:] = action_mat
    value_saved[sim, :, :] = final_value
    states_saved[sim, :, :] = state_mat
    model_saved[sim, :, :, :] = model_r
    tm_saved.append(tm_final)
    value_epi_saved.append(value_epi)
    tm_epi_saved.append(tm_epi)
    # Not saved 
    rewards_result_steps[sim, :] = np.sum(reward_sums_steps, axis = 1)
    


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

    with open(f'saved/{world_model}/mbased_{agent}_{world_model}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    np.savez_compressed(f'saved/{world_model}/tmss/mbased_{agent}_tm.npz', *tm_saved)
    np.savez_compressed(f'saved/{world_model}/mbased_{agent}_value_epi.npz', *value_epi_saved)
    np.savez_compressed(f'saved/{world_model}/tmss/mbased_{agent}_tm_epi.npz', *tm_epi_saved)
    print(f"MB {agent} data saved")


    ## PLOT PERFORMANCE ##
    if agent == "agent" and (world_model == "exp2" or world_model == "exp3"):
        start_gen = n_episodes*training
    else:
        start_gen = None

    ## SAVE FIGURES ##
    folder_path = os.path.join('saved', 'figures', str(world_model))

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    title = f"MB {agent} {n_simulations} sim"
    fig1, ax1 = plot_performance(rewards_result_epi_saved, "Episodes", title , "Performance", start_gen)
    fig1.savefig(os.path.join(folder_path, f'mb_{agent}_{world_model}_performance.png'), bbox_inches='tight')
    fig2, ax2 = plot_performance(steps_saved, "Episodes", title, "Steps to reward", start_gen)
    fig2.savefig(os.path.join(folder_path, f'mb_{agent}_{world_model}_steps.png'), bbox_inches='tight')
    print(f"Figures saved for MB {agent}")