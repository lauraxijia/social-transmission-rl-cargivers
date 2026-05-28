import numpy as np
from utils.helper_functions import softmax_policy, q_learning, value_shaping, find_reward

"""Policy of model-free value shaping agent."""
                             
def mf_valueshaping(env, rewards_info, experts_states, experts_actions, world, max_steps, n_episodes, params, training, rng, optimization = False, world_model='baseline', rewards_exp2= None):

    # Assert that exp is either 'baseline' or 'exp2' or 'exp3'
    assert world_model in ['baseline', 'exp2', 'exp3'], "exp must be 'baseline', 'exp2', or 'exp3'"
      
    # Start with a uniform value function
    value = np.ones((env.n_states, env.n_actions)) # value matrix
    # To save the value for each episode
    value_perepi = np.zeros((n_episodes, env.n_states, env.n_actions))

    
    # Initialize the reward mat
    reward_sums_steps = np.zeros((n_episodes, max_steps))
    reward_sums_epi = np.zeros((n_episodes))
    
    # Initialize state mat as 
    state_mat = np.nan*np.zeros((n_episodes, max_steps+1), dtype=int)
    
    # Initialize action mat as 
    action_mat = np.nan*np.zeros((n_episodes, max_steps))
    
    steps_to_reward = np.nan*np.zeros((n_episodes,))
    
    
    # Loop over episodes
    for episode in range(n_episodes):

        # Asocial Possible initial loc
        if episode < (n_episodes * training): # Training
            agent_location = env.initial_loc(exp = "baseline")
        else: # Test
            agent_location = env.initial_loc(exp = world_model)

        # Initial state - needed for the value function 
        state = world[agent_location]
        # save initial state
        state_mat[episode, 0] = state
        
        # Take agent's states and actions for the last number of episodes of the agent
        exp_states, exp_actions = experts_states[-n_episodes:,:], experts_actions[-n_episodes:,:]
        
        for t in range(max_steps):

            ## SOCIAL LEARNING - the agent observes and updates its value in from observing the expert
            # During training 
            if episode < (n_episodes*training):
                # Take the value of the rewards for that episode
                # Takes the rewards from the 20 last episodes
                reward_placed = rewards_info[-n_episodes:, :][episode][:,:2]
                
                # If expert hasn't found the reward
                if ~np.isnan(exp_actions[episode, t]): 
                    
                    # Observe expert's state (t+1 because of initial state) and action (taken at time t)
                    exp_state, exp_action = exp_states[episode, t], exp_actions[episode, t]
                    # Update value shaping 
                    value = value_shaping(value, exp_action, exp_state, params['kappa'])
                    _, action = softmax_policy(value, state, env.n_actions, params['beta'], rng)

                # If the expert has found the reward - Individual learning
                else:
                    _, action = softmax_policy(value, state, env.n_actions, params['beta'], rng)


            ## -- END SOCIAL LEARNING
            
            # During test:
            else:
                if world_model == "exp2":
                    if rewards_exp2 is not None:
                        reward_placed = rewards_exp2[-n_episodes:, :][episode][:,:2]
                    else:
                        raise ValueError("rewards_exp2 must be provided for exp2")
                else:
                    reward_placed = rewards_info[-n_episodes:, :][episode][:,:2]

                # Normal action and value has not been modified
                _, action = softmax_policy(value, state, env.n_actions, params['beta'], rng)

                ## -- END SOCIAL LEARNING

            # If expert had found the reward - individual learning

            ## INDIVIDUAL LEARNING 
            #Calculate new location based on the action
            next_agent_location, next_state = env.move_agent(action, state, agent_location, reward_placed)
            # observe reward for that action 
            reward = find_reward(state, reward_placed) 
            # Update rewards based in agent location
            reward_sums_steps[episode, t] = reward
            reward_sums_epi[episode] += reward

             ## -- APPLY THE LEARNING RULE - Q LEARNING -- ##
            value = q_learning(value, state, action, next_state, reward, params['gamma'], params['alpha'])
            ## -- End of Learning Rule --  
            ## -- End INDIVIDUAL LEARNING
            
            ## -- UPDATES --
            
            # Update the state function
            action_mat[episode,t] = action
            
            # FOR THE TERMINATION EPISODE
            if reward > 0:
                steps_to_reward[episode] = t + 1
                break  # episode ends
            
            # Update agent location and state
            state = next_state
            agent_location = next_agent_location

             # Save the state
            state_mat[episode, t+1] = state
        # If the episode doesnt terminate, it locates a 40
        steps_to_reward[episode] = t + 1 
        value_perepi[episode, :, :] = value
    
    if optimization:
        return reward_sums_epi
    
    else:
       return value, reward_sums_epi, state_mat, action_mat, steps_to_reward, value_perepi, reward_sums_steps
