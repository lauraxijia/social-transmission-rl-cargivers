import numpy as np
from .world import VillageWorld
from tqdm import tqdm

"""Helperfunctions for social learning for the simulations."""

def social_sim_mf(algorithm, expert_data, 
                  worlds_saved, rewards_placed, 
                  n_simulations, max_steps, n_episodes, 
                  params, training, 
                  rng, optimization = False, world_model = 'baseline', rewards_exp2= None):
    
    # total reward for each episode
    rewards_result_epi_saved = np.zeros((n_simulations, n_episodes))
    # Save the reward for each step
    rewards_result_steps = np.zeros((n_simulations, n_episodes))
    steps_to_reward = np.zeros((n_simulations, n_episodes))

    final_value_saved = np.zeros((n_simulations, 100, 4))
    
    # States expert goes through - mainly to check results
    states_saved = np.zeros((n_simulations, n_episodes, max_steps+1))
    # Actions taken by the expert - Policy
    actions_saved = np.zeros((n_simulations, n_episodes, max_steps))
    # Value per episode
    value_epi_saved = []

    if optimization:
        
        for sim in range(n_simulations):

            # Extract data from the expert
            env = VillageWorld(worlds_saved[sim], rng)
            
            experts_actions = expert_data['actions_saved'][sim]
            experts_states = expert_data['states_saved'][sim]

            if rewards_exp2:
                rwds_exp2 = rewards_exp2[sim]
            else:
                rwds_exp2 = None

            rewards_result_epi_saved[sim, :] = algorithm(env, rewards_placed[sim], 
                                                         experts_states, experts_actions, 
                                                         worlds_saved[sim],
                                                         max_steps, n_episodes, params, 
                                                         training, 
                                                         rng, 
                                                         optimization, world_model, rwds_exp2)
                   
        return rewards_result_epi_saved
    
    else:
        
        for sim in tqdm(range(n_simulations)):
            
            env = VillageWorld(worlds_saved[sim],rng)
            
            experts_actions = expert_data['actions_saved'][sim]
            experts_states = expert_data['states_saved'][sim]

            if rewards_exp2:
                rwds_exp2 = rewards_exp2[sim]
            else:
                rwds_exp2 = None


            final_value, reward_sums_epi, state_mat, action_mat, t_to_reward, value_epi, reward_sums_steps = algorithm(env, rewards_placed[sim], experts_states, experts_actions, 
                                                                                            worlds_saved[sim], max_steps, n_episodes, params, 
                                                                                            training, rng, optimization = False, world_model = world_model, rewards_exp2 = rwds_exp2)
        
            rewards_result_epi_saved[sim, :] = reward_sums_epi
            steps_to_reward[sim,:] = t_to_reward
            final_value_saved[sim,:,:] = final_value
            states_saved[sim, :, :] = state_mat
            actions_saved[sim, :, :] = action_mat
            value_epi_saved.append(value_epi)
            # Not saved 
            rewards_result_steps[sim, :] = np.sum(reward_sums_steps, axis = 1)

        return final_value_saved, rewards_result_epi_saved, states_saved, actions_saved, steps_to_reward, value_epi_saved, rewards_result_steps


def social_sim_mb(algorithm,
                  expert_data,
                  worlds_saved, 
                  rewards_placed, 
                  n_simulations, 
                  max_steps, 
                  n_episodes, 
                  params, 
                  training, 
                  rng, 
                  optimization = False, 
                  world_model = "baseline",
                  rewards_exp2= None):

    n_states = 100
    freq = n_episodes // 5
    # Final transition matrix for each 5 simulation
    transition_mat_saved = np.zeros((n_simulations, freq + 1, n_states, 4, n_states))
    final_tm_saved = np.zeros((n_simulations, n_states, 4, n_states))
    # total reward for each episode
    rewards_result_epi_saved = np.zeros((n_simulations, n_episodes))
    # Save the reward for each step
    rewards_result_steps = np.zeros((n_simulations, n_episodes))
    steps_to_reward = np.zeros((n_simulations, n_episodes))
    value_saved = np.zeros((n_simulations,n_episodes+1, n_states, 4))
    final_value_saved = np.zeros((n_simulations, n_states, 4))
    # States expert goes through - mainly to check results
    states_saved = np.zeros((n_simulations, n_episodes, max_steps+1))
    # Actions taken by the expert - Policy
    actions_saved = np.zeros((n_simulations, n_episodes, max_steps))
    # Final tm of each simulation
    tm_saved = []
    # Value per episode
    value_epi_saved = []
    # Tm every 5 episodes
    tm_epi_saved = []


    if optimization:

        for sim in range(n_simulations):
            
            env = VillageWorld(worlds_saved[sim], rng)
            
            experts_actions = expert_data['actions_saved'][sim]
            experts_states = expert_data['states_saved'][sim]

            if rewards_exp2:
                rwds_exp2 = rewards_exp2[sim]
            else:
                rwds_exp2 = None

            rewards_result_epi_saved[sim, :] = algorithm(env, 
                                                         rewards_placed[sim], 
                                                         experts_states, 
                                                         experts_actions, 
                                                         worlds_saved[sim], 
                                                         max_steps, 
                                                         n_episodes, 
                                                         params, 
                                                         training, 
                                                         rng, 
                                                         optimization,
                                                         world_model,
                                                         rwds_exp2)
            
        return rewards_result_epi_saved
    
    else:
        
        for sim in tqdm(range(n_simulations)):
            
            env = VillageWorld(worlds_saved[sim], rng)
            
            experts_actions = expert_data['actions_saved'][sim]
            experts_states = expert_data['states_saved'][sim]
            
            if rewards_exp2:
                rwds_exp2 = rewards_exp2[sim]
            else:
                rwds_exp2 = None

            final_value, reward_sums_epi, state_mat, action_mat, t_to_reward, tm_final, model_r, value_epi, tm_epi, reward_sums_steps = algorithm(env, rewards_placed[sim], experts_states, experts_actions, 
                                                                                            worlds_saved[sim], max_steps, n_episodes, params, 
                                                                                            training, 
                                                                                            rng, optimization = False, 
                                                                                            world_model = world_model,
                                                                                            rewards_exp2 = rwds_exp2)
                                                                                                            
            rewards_result_epi_saved[sim, :] = reward_sums_epi                              
            steps_to_reward[sim,:] = t_to_reward
            final_value_saved[sim,:,:] = final_value
            states_saved[sim, :, :] = state_mat
            actions_saved[sim, :, :] = action_mat
            tm_saved.append(tm_final)
            value_epi_saved.append(value_epi)
            tm_epi_saved.append(tm_epi)
            # Not saved 
            rewards_result_steps[sim, :] = np.sum(reward_sums_steps, axis = 1)

        return final_value_saved, rewards_result_epi_saved, states_saved, actions_saved, steps_to_reward, tm_saved, value_epi_saved, tm_epi_saved


def simulate_learning(algorithm, n_simulations, max_steps,  n_episodes, params, *args):
    sum_rewards = np.zeros((n_simulations, n_episodes))
    steps_to_reward = np.zeros((n_simulations, n_episodes))

    reward_distribution = np.array([0, 0, 0, 0, 50, 50])

    for sim in range(n_simulations):
        # Initialize the world
        grid_world = VillageWorld()

        reward_epi_steps, t_to_reward = algorithm(grid_world, max_steps, n_episodes, params, reward_distribution)
        
        sum_rewards[sim,:] = np.sum(reward_epi_steps, axis=1)
        steps_to_reward[sim,:] = t_to_reward


    return sum_rewards, steps_to_reward

def extract_belief(algorithm, n_simulations, expert_dic, max_steps, n_episodes, params, x):

    # Final transition matrix for each simulation
    freq = n_episodes // 5
    transition_mat_saved = np.zeros((n_simulations, freq + 1, 64, 4, 64))

    reward_dis = np.array([0,0,0,0,50,50])

    for sim in range(n_simulations):

        # Extract data from the expert
        experts_rewards = expert_dic['reward_location'][sim]
        experts_init_loc = expert_dic['initial_loc'][sim] 
        experts_actions = expert_dic['actions_saved'][sim]
        experts_states = expert_dic['states_saved'][sim]
        experts_worlds  = expert_dic['experts_world'][sim]

        tran_dq = algorithm(experts_init_loc, experts_rewards, experts_states, experts_actions, experts_worlds, max_steps, 
                                                    n_episodes, params, reward_dis, x)
        
        transition_mat_saved[sim, :, :, :, :] = tran_dq

        return transition_mat_saved
