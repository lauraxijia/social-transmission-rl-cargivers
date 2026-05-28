import numpy as np
from utils.helper_functions import softmax_policy, q_learning, find_reward, decision_bias


"""Policy of model-free decision bias agent."""

def mf_policy(env, rewards_info, experts_states, experts_actions, 
              world, 
              max_steps, n_episodes, 
              params, 
              training, 
              rng, 
              optimization = False, world_model = 'baseline', rewards_exp2= None): 

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
    state_mat = np.nan*np.zeros((n_episodes, max_steps+1))
    # Initialize action mat as 
    action_mat = np.nan*np.zeros((n_episodes, max_steps))

    steps_to_reward = np.nan*np.zeros((n_episodes,))
    
    # Loop over episodes
    for episode in range(n_episodes):

        # initial loc
        if episode < (n_episodes * training): # Training
            agent_location = env.initial_loc(exp = "baseline")
        else: # Test
            agent_location = env.initial_loc(exp = world_model)
            
        # From the location finds the state
        state = world[agent_location]
        # save initial state
        state_mat[episode, 0] = state
  
        # Take agent's last states and actions 
        exp_states, exp_actions = experts_states[-n_episodes:,:], experts_actions[-n_episodes:,:]

        # Loop over steps
        for t in range(max_steps):

            # During training
            if episode < (n_episodes * training):
                # Take the value of the rewards for that episode
                reward_placed = rewards_info[-n_episodes:, :][episode][:,:2]

                # Action is determined by the action of the expert
                action = decision_bias(env, exp_states, agent_location, 
                                         state, value, 
                                         params, world, episode, t, reward_placed,
                                         rng=rng, modelbased = False, transition_belief=None)
            
            # Test
            else:
                if world_model == "exp2":
                    if rewards_exp2 is not None:
                        reward_placed = rewards_exp2[-n_episodes:, :][episode][:,:2]
                    else:
                        raise ValueError("rewards_exp2 must be provided for exp2")
                else:
                    reward_placed = rewards_info[-n_episodes:, :][episode][:,:2]
                # Action is determined by asocial learning
                _, action = softmax_policy(value, state, env.n_actions, params['beta'], rng)

            # Calculate new location based on the action
            next_agent_location, next_state = env.move_agent(action, state, agent_location, reward_placed)

            # observe reward for that action 
            reward = find_reward(state, reward_placed)  
            
            # Update rewards based in agent location
            reward_sums_steps[episode, t] = reward
            reward_sums_epi[episode] += reward

            ## -- APPLY THE LEARNING RULE - Q LEARNING --
            value = q_learning(value, state, action, 
                               next_state, reward, 
                               params['gamma'], params['alpha'])
            ## -- End of Learning Rule --

        
            ## -- UPDATES --
            
            # Update the state function            
            action_mat[episode,t] = action
            
            # FOR THE TERMINATION EPISODE
            if reward > 0:
                steps_to_reward[episode] = t +1 
                
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



