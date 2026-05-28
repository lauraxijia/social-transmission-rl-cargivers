import numpy as np
from utils.helper_functions import softmax_policy, q_learning, dynaq_planner, find_reward,  model_update, decision_bias

"""Policy of model-based decision bias agent."""

def mb_policy(env, 
              rewards_info, 
              experts_states, 
              experts_actions, 
              world, 
              max_steps, 
              n_episodes, 
              params, 
              training, 
              rng, 
              optimization = False,
              world_model = 'baseline',
              rewards_exp2 = None): 
    
    # Assert that exp is either 'baseline' or 'exp2' or 'exp3'
    assert world_model in ['baseline', 'exp2', 'exp3'], "exp must be 'baseline', 'exp2', or 'exp3'"

    # Start with a uniform value function
    value = np.ones((env.n_states, env.n_actions))
    # To save the value function for each episode
    value_perepi = np.zeros((n_episodes, env.n_states, env.n_actions))

    # Initialize the Transition Belief
    transition_belief, true_tm = env.transition_probabilities()
    # Sum by state for calculate adjacent
    ground_truth_tm = true_tm.sum(axis=1)
    # To save the transition for every 5 episodes
    freq = n_episodes // 5
    belief_perepi = np.zeros((freq+1 , env.n_states, env.n_actions, env.n_states))
    # Add initial transition 
    belief_perepi[0, :, :] = transition_belief

    model_r = np.nan*np.zeros((env.n_states, env.n_actions, 2))
    
    # Initialize the reward mat
    reward_sums_steps = np.zeros((n_episodes, max_steps))
    reward_sums_epi = np.zeros((n_episodes))
    
    # Initialize state mat as 
    state_mat = np.nan*np.zeros((n_episodes, max_steps+1))
    
    # Initialize action mat as 
    action_mat = np.nan*np.zeros((n_episodes, max_steps))
    
    steps_to_reward = np.nan*np.zeros((n_episodes,))

    
    e = 1
    # Loop over episodes
    for episode in range(n_episodes):
        
        # Asocial Possible initial loc
        if episode < (n_episodes * training): # Training
            agent_location = env.initial_loc(exp = "baseline")
        else: # Test
            agent_location = env.initial_loc(exp = world_model)

        state = world[agent_location]
        # Save initial state
        state_mat[episode, 0] = state

        
        # Take agent's last states and actions 
        exp_states, _ = experts_states[-n_episodes:,:], experts_actions[-n_episodes:,:]

        # Sample n_steps from Poisson distribution with mean lambda 
        n_steps = rng.poisson(params['lambda'], size=max_steps)
        
        # Loop over steps
        for t in range(max_steps):
            
            
            # Training
            if episode < (n_episodes * training):

                # Take the value of the rewards for that episode
                reward_placed = rewards_info[-n_episodes:, :][episode][:,:2]

                # Action is determined by the action of the expert
                action = decision_bias(env, 
                                         exp_states, 
                                         agent_location, 
                                         state, value, 
                                         params, world, episode, t, reward_placed,
                                         rng,
                                         modelbased = True,
                                         transition_belief = transition_belief)

            
            # Test
            else:
                # Take the value of the rewards for that episode
                if  world_model == 'exp2':
                    if rewards_exp2 is not None:
                        reward_placed = rewards_exp2[-n_episodes:, :][episode][:,:2]
                    else: 
                        raise ValueError("rewards_exp2 must be provided for exp2")
                else:
                    reward_placed = rewards_info[-n_episodes:, :][episode][:,:2]
                
                # Select action based on softmax policy
                _, action = softmax_policy(value, 
                                        state, 
                                        env.n_actions, 
                                        params['beta'], 
                                        rng)
                
            
            # Calculate new location based on the action
            next_agent_location, next_state = env.move_agent(action, 
                                                             state, 
                                                             agent_location, 
                                                             reward_placed)
            
            # observe reward for that action 
            reward = find_reward(state, reward_placed)  
            # Update rewards based in agent location
            reward_sums_steps[episode, t] = reward
            reward_sums_epi[episode] += reward
           
            ## -- APPLY THE LEARNING RULE - Q LEARNING --           
            value = q_learning(value, 
                               state, 
                               action, 
                               next_state, 
                               reward, 
                               params['gamma'], params['alpha'])
           
            ## -- End of Learning Rule --

            ## -- DYNA-Q PLANNING -- ##
            model_r = model_update(model_r, state, action, reward, next_state)
            
            value = dynaq_planner(value, model_r, 
                                  state, action, 
                                  next_state, 
                                  n_steps[t], params['gamma'], params['alpha'],
                                  rng)

            ## -- End of DYNA-Q PLANNING -- ##

            # Update rewards based in agent location            

            ## -- TRANSITION MATRIX UPDATE -- ##
            # Initialized after each t - shape as state vector
            kronecker_delta = np.zeros((env.n_states,))
            if next_state == None:
                next_state = state

            kronecker_delta[next_state] = 1
            
            # Update the model 
            transition_belief[state, action] = transition_belief[state, action] + params['alpha_t']*(kronecker_delta - transition_belief[state, action])
            # After each update - Normalize
            transition_belief[state, action, :] = transition_belief[state, action, :] / np.sum(transition_belief[state, action, :])
            ## -- End of DYNA-Q UPDATE -- ##
           
            # Update the states and actions for comparison
            
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
        # If the episode doesnt terminate, it locates a 39
        steps_to_reward[episode] = t+1
        # Save final value of the whole episode
        value_perepi[episode, :, :] = value
        
    if optimization:
        return reward_sums_epi
    
    else:
       return value, reward_sums_epi, state_mat, action_mat, steps_to_reward, transition_belief, model_r, value_perepi, belief_perepi, reward_sums_steps

