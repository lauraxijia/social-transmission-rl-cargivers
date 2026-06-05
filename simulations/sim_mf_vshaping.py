import os
import numpy as np
import json

from models.mf_valueshaping import MFValueShapingAgent
from utils.helper_functions import run_simulations, aggregate_results, save_results
from utils.plot_functions import plot_performance

"""Simulation of the model-free value shaping agent."""

AgentClass = MFValueShapingAgent

# --- Set simulation parameters --- #
save = False
n_episodes = 20
exp = "baseline" # "baseline" | "exp3" | "exp2"
training_split = 0.5
n_states = 100
n_simulations = 1 #1000
max_steps = 40
n_actions = 4
expert_type = "mbased_expert" # "mbased_pedagogical_expert" | "mbased_expert" 
teaching_objective = "action_gap"  # "q_mismatch" | "action_gap" | "cumulative_reward"

learner_string = "mfree_vshaping"
expert_string = f"{expert_type}_{teaching_objective}" if expert_type == "mbased_pedagogical_expert" else expert_type

# Set seed for reproducibility
seed = 5
rng = np.random.default_rng(seed)

# --- Load data --- #
# Load agent parameters
with open(f'saved/opti_results/mfree_vshaping.json', 'r') as json_file:
    params = json.load(json_file)["opti_params"]
print("Params agent MF-VS: \n ", params)

# Load data from the expert
with open(f'saved/baseline/{expert_string}_baseline.json', 'r') as json_file:
    expert_data = json.load(json_file)
for k in expert_data.keys():
    expert_data[k] = np.array(expert_data[k])


# Load worlds
#loaded = np.load('saved/worlds.npz')
words_loaded = np.load(f'saved/dummy_worlds.npz')
worlds = [words_loaded[f'arr_{i}'] for i in range(len(words_loaded.files))]

# Load rewards
#rewards_load = np.load('saved/rewards_info.npz')
rewards_loaded = np.load(f'saved/dummy_rewards_fixed.npz')
rewards = [rewards_loaded[f'arr_{i}'] for i in range(len(rewards_loaded.files))]

# Load modified rewards for test phase for exp2
if exp == "exp2":
    rewards_load_exp2 = np.load('saved/rewards_exp2.npz')
    rewards_exp2 = [rewards_load_exp2[f'arr_{i}'] for i in range(len(rewards_load_exp2.files))]
else:
    rewards_exp2 = None


# --- Run simulation --- #
print(f"Run exp {exp} with MF Value Shaping agent for {n_simulations} simulation(s) à {n_episodes} episodes, and max {max_steps} steps per episode with {expert_string}.")
results = run_simulations(
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
    optimization = False
    )

# --- Aggregate results --- #
results = aggregate_results(results)

# --- Plot performance --- #
fig1, ax1 = plot_performance(results["sum_rewards"], "Episodes", f"MF Value Shaping {params['kappa']} kappa", "Performance", n_episodes*training_split)
fig2, ax2 = plot_performance(results["steps_to_reward"], "Episodes", f"MF Value Shaping {params['kappa']} kappa", "Steps to reward", n_episodes*training_split )  

# --- Save results --- #
if save:
    save_results(results, exp, learner_string, expert_string)
    fig1.savefig(f'saved/figures/{exp}/{learner_string}_{expert_string}_{exp}_performance.png')
    fig2.savefig(f'saved/figures/{exp}/{learner_string}_{expert_string}_{exp}_steps_to_reward.png')