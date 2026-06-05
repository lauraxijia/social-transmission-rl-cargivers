import numpy as np
from utils.helper_functions import softmax_policy, q_learning, dynaq_planner, find_reward, model_update

"""Policy of model-based asocial agent/expert."""

#@njit
def mb_expert(params, env, world, rewards_info, max_steps, n_episodes, training, rng, agent = "agent", optimization = False, world_model='baseline', rewards_exp2 = None):

    # Assert that exp is either 'baseline' or 'exp2' or 'exp3'
    assert world_model in ['baseline', 'exp2', 'exp3'], "exp must be 'baseline', 'exp2', or 'exp3'"

    ## Initializations ##

    # Start with a uniform value function
    value = np.ones((env.n_states, env.n_actions)) 
    # To save the value function for each episode
    value_perepi = np.zeros((n_episodes, env.n_states, env.n_actions))

    # Initialize de Transition Belief
    transition_belief = env.init_transit_mat
    agent_episodes = 20 
    freq = agent_episodes // 5
    belief_perepi = np.zeros((freq+1 , env.n_states, env.n_actions, env.n_states))
    # Add initial transition 
    belief_perepi[0, :, :] = transition_belief

    # Initialize the model. Saves the reward and transition for each state-action pair
    model_r = np.nan*np.zeros((env.n_states, env.n_actions, 2))
        
    # Initialize the reward mat
    reward_per_step = np.zeros((n_episodes, max_steps)) 
    reward_sums_epi = np.zeros((n_episodes))

    # Initialize state mat as 
    state_mat = np.nan*np.zeros((n_episodes, max_steps+1), dtype=int)

    # Initialize action mat as 
    action_mat = np.nan*np.zeros((n_episodes, max_steps), dtype=int)

    steps_to_reward = np.nan*np.zeros((n_episodes,))

    
    ## LOOP OVER EPISODES ##
    value_steps_list_epi = []
    value_finalepi_list = []
    # Loop over episodes
    for episode in range(n_episodes):
        
        value_steps_list = []

        reward_placed = rewards_info[-n_episodes:, :][episode][:,:2]

        # Take the value of the rewards for that episode - In the case of the expert takes all the 120 saved rewards
        # - In the case of the agent takes the last 20 rewards
        # Test phase (only for the agent)
        if episode >= (n_episodes * training) and agent == "agent":
            if world_model == 'exp2':
                if rewards_exp2 is not None:
                    reward_placed = rewards_exp2[-n_episodes:, :][episode][:,:2]
                else: 
                  raise ValueError("rewards_exp2 must be provided for exp2")
        
            agent_location = env.initial_loc(exp = world_model)
        # Training phase or expert
        else:
            agent_location = env.initial_loc(exp = "baseline")
        print(f"Episode {episode} - Starting at location {agent_location} in world model {world_model} with reward placements {reward_placed}")

        ## Initial state - needed for the value function
        state = world[agent_location]
        # save initial state
        state_mat[episode, 0] = state

        # Sample n_steps from Poisson distribution with mean lambda 
        n_steps = rng.poisson(params['lambda'], size=max_steps)
        # Loop over steps
        for t in range(max_steps):

            # Chose the next action randomly
            _, action = softmax_policy(value, state, env.n_actions, params['beta'], rng)
            
            # Calculate new location based on the action
            next_agent_location, next_state = env.move_agent(action, state, agent_location, reward_placed)

            # Observe reward for that action 
            reward = find_reward(state, env, reward_placed)  
            # Update rewards based in agent location
            reward_per_step[episode, t] = reward 
            reward_sums_epi[episode] += reward

            ## -- APPLY THE LEARNING RULE - Q LEARNING -- ##
            value = q_learning(value, state, action, next_state, reward, params['gamma'], params['alpha'])
            ## -- End of Learning Rule --
           
            
            ## -- DYNA-Q PLANNING -- ##
            model_r = model_update(model_r, state, action, reward, next_state)

            value = dynaq_planner(value, 
                                  model_r, 
                                  state, action, next_state, 
                                  n_steps[t], params['gamma'], params['alpha'], rng)
            ## -- End of DYNA-Q PLANNING -- ##
            

            ## -- TRANSITION MATRIX UPDATE -- ##
            # Initialized after each t - shape as state vector
            kronecker_delta = np.zeros((env.n_states,))
            # Each entry is 0 except for the experienced next_state - Antonov, 2023
            # This is necessary for the update of the transition belief
            if next_state == None:
                next_state = state
            kronecker_delta[next_state] = 1

            # Update the model 
            transition_belief[state, action] = transition_belief[state, action] + params['alpha_t']*(kronecker_delta - transition_belief[state, action])
            # Normalize the transition matrix
            transition_belief[state, action, :] = transition_belief[state, action, :] / np.sum(transition_belief[state, action, :])

            
            ## -- End of DYNA-Q UPDATE -- ##

            value_steps_list.append(np.copy(value)) 

            ## -- UPDATES --          
            
            action_mat[episode,t] = action
             
            # For the Termination Episode
            if reward > 0:
                steps_to_reward[episode] = t + 1
                # Save the state
                break  # episode ends
              
            # Update the state
            state = next_state
            agent_location = next_agent_location

            # Save the state
            state_mat[episode, t+1] = state
            
        value_steps_list_epi.append(value_steps_list)
        value_finalepi_list.append(np.copy(value))
        # If the episode doesnt terminate, it locates a 40
        steps_to_reward[episode] = t+1

    if optimization:    
        return reward_sums_epi
    
    else:
       return value, reward_sums_epi, state_mat, action_mat, steps_to_reward, transition_belief, model_r, value_perepi, belief_perepi, reward_per_step # value_finalepi_list, value_steps_list_epi