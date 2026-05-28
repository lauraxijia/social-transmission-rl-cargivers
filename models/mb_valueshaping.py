import numpy as np
from utils.helper_functions import softmax_policy, q_learning, dynaq_planner, find_reward,model_update, value_shaping

"""Policy of model-based value shaping agent."""

def mb_valueshaping(env, 
                    rewards_info, 
                    experts_states, 
                    experts_actions, 
                    world, max_steps, 
                    n_episodes, params, 
                    training, 
                    rng, optimization = False, world_model = 'baseline', rewards_exp2= None):
    
    # Assert that exp is either 'baseline' or 'exp2' or 'exp3'
    assert world_model in ['baseline', 'exp2', 'exp3'], "exp must be 'baseline', 'exp2', or 'exp3'"

    # Start with a uniform value function
    value = np.ones((env.n_states, env.n_actions)) # value matrix
    # To save the value function for each episode
    value_perepi = np.zeros((n_episodes, env.n_states, env.n_actions))

    # Initialize de Transition Belief
    transition_belief = env.init_transit_mat 
    # To save the transition for every 5 episodes
    freq = n_episodes // 5
    belief_perepi = np.zeros((freq+1, env.n_states, env.n_actions, env.n_states))
    # Add initial transition 
    belief_perepi[0, :, :] = transition_belief
    
    # Initialize the model. Saves the reward and transition for each state-action pair
    model_r = np.nan*np.zeros((env.n_states, env.n_actions, 2))
    
    # Initialize the reward mat
    reward_sums_steps = np.zeros((n_episodes, max_steps))
    reward_sums_epi = np.zeros((n_episodes))
    
    # Initialize state mat as 
    state_mat = np.nan*np.zeros((n_episodes, max_steps+1), dtype=int)
    
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
        
        # Initial state - needed for the value function
        state = world[agent_location]
        # save initial state
        state_mat[episode, 0] = state

        # Take agent's states and actions for the last number of episodes of the agent
        exp_states, exp_actions = experts_states[-n_episodes:,:], experts_actions[-n_episodes:,:]

        
        # Sample lambda from Poisson distribution with mean lambda_mean
        n_steps = rng.poisson(params['lambda'], size=max_steps)
        
        # Loop over steps
        for t in range(max_steps):
      
            # SOCIAL LEARNING - the agent observes and updates its value from observing the expert
            # During training 
            if episode < (n_episodes*training):
                # Take the value of the rewards for that episode
                # Takes the rewards from the 20 last episodes
                reward_placed = rewards_info[-n_episodes:,:][episode][:,:2]
                # If expert hasn't found the reward
                if  ~np.isnan(exp_actions[episode, t]): 
                    # Observe expert's state and action (taken at time t)
                    exp_state, exp_action = exp_states[episode, t], exp_actions[episode, t]   
                    # Update value shaping 
                    value = value_shaping(value, exp_action, exp_state, params['kappa'])
                    _, action = softmax_policy(value, state, env.n_actions, params['beta'], rng)
                
                # If the expert has found the reward - Individual learning
                else:
                    _, action = softmax_policy(value, state, 
                                            env.n_actions, 
                                            params['beta'], 
                                            rng)

            ## -- END SOCIAL LEARNING

            # During test # INDIVIDUAL LEARNING
            else:
                if world_model == "exp2":
                    if rewards_exp2 is not None:
                        reward_placed = rewards_exp2[-n_episodes:, :][episode][:,:2]
                    else: 
                        raise ValueError("rewards_exp2 must be provided for exp2")
                else:
                    reward_placed = rewards_info[-n_episodes:, :][episode][:,:2]

                # Normal action and value has not been modified
                _, action = softmax_policy(value, state, 
                                           env.n_actions, 
                                           params['beta'], 
                                           rng)        

            ## INDIVIDUAL LEARNING 
            # Calculate new location based on the action
            next_agent_location, next_state = env.move_agent(action, state, 
                                                             agent_location, 
                                                             reward_placed)
            # observe reward for that action 
            reward = find_reward(state, reward_placed) 
            # Update rewards based in agent location            
            reward_sums_steps[episode, t] = reward
            reward_sums_epi[episode] += reward
            
            ## -- APPLY THE LEARNING RULE - Q LEARNING -- ##
            value = q_learning(value, state, action, 
                               next_state, reward, 
                               params['gamma'], params['alpha'])
            ## -- End of Learning Rule --      
            
            ## -- DYNA-Q PLANNING -- ##
            model_r = model_update(model_r, state, action, reward, next_state)
            
            value = dynaq_planner(value, 
                                  model_r, 
                                  state, action, next_state, 
                                  n_steps[t], params['gamma'], params['alpha'],
                                  rng)
            ## -- End of DYNA-Q PLANNING -- ##
        

            ## -- TRANSITION BELIEF UPDATE -- ##
            # Initialized after each t - shape as state vector
            kronecker_delta = np.zeros((env.n_states,))
            # If the reward is found, the next state is the same as the current state. This change is to be able to compute the probabilities
            if next_state == None:
                next_state = state

            kronecker_delta[next_state] = 1
            # Update the belief
            transition_belief[state, action] = transition_belief[state, action] + params['alpha_t']*(kronecker_delta - transition_belief[state, action])
            # After each update - Normalize
            transition_belief[state, action, :] = transition_belief[state, action, :] / np.sum(transition_belief[state, action, :])


            
            ## -- UPDATES
            # Update actions for comparison
            action_mat[episode,t] = action

            # FOR THE TERMINATION EPISODE
            if reward > 0: # termination when it reaches a reward
                steps_to_reward[episode] = t + 1
                break  # episode ends

            # Update agent location and state
            state = next_state 
            agent_location = next_agent_location

            # Save the state
            state_mat[episode, t+1] = state
        # If the episode doesnt terminate, it locates a 39
        steps_to_reward[episode] = t + 1
        # Save final value of the whole episode
        value_perepi[episode, :, :] = value
        
    if optimization:
        return reward_sums_epi
    else:
       return value, reward_sums_epi, state_mat, action_mat, steps_to_reward, transition_belief, model_r, value_perepi, belief_perepi, reward_sums_steps


