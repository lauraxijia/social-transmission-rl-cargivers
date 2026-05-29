from collections import defaultdict

import numpy as np
from utils.helper_functions import softmax_policy, q_learning, dynaq_planner, find_reward, model_update, teaching_objective

"""Policy of model-based pedagogical expert."""

def mb_pedagogical_expert(params, env, world, rewards_info, max_steps, n_episodes_total, n_teach_episodes, rng, optimization = False, exp='baseline', learner_function = None, learner_params = None, n_learner = None, *, objective):

    # Assert that exp is either 'baseline' or 'exp2' or 'exp3'
    assert exp in ['baseline', 'exp2', 'exp3'], "exp must be 'baseline', 'exp2', or 'exp3'"

    ## Initializations ##
    # Start with a uniform value function
    value = np.ones((env.n_states, env.n_actions)) 
    # To save the value function for each episode
    value_perepi = np.zeros((n_episodes_total, env.n_states, env.n_actions))

    # Initialize transition belief
    transition_belief = env.init_transit_mat
    agent_episodes = 20 
    freq = agent_episodes // 5
    belief_perepi = np.zeros((freq+1 , env.n_states, env.n_actions, env.n_states))
    # Add initial transition 
    belief_perepi[0, :, :] = transition_belief

    # Initialize the model. Saves the reward and transition for each state-action pair
    model_r = np.nan*np.zeros((env.n_states, env.n_actions, 2))
        
    # Initialize the reward mat
    reward_per_step = np.zeros((n_episodes_total, max_steps)) 
    reward_sums_epi = np.zeros((n_episodes_total))

    # Initialize state mat as 
    state_mat = np.nan*np.zeros((n_episodes_total, max_steps+1), dtype=int)

    # Initialize action mat as 
    action_mat = np.nan*np.zeros((n_episodes_total, max_steps), dtype=int)

    steps_to_reward = np.nan*np.zeros((n_episodes_total,))

    # Initialize storage for value function per episode
    value_steps_list_epi = []
    value_finalepi_list = []

    # Initialize dictionary of learners for n_learner
    learner_candidates = {
        i: learner_function(
            rng=rng,
            params=learner_params,
            env=env,
            world=world,
            rewards_info=rewards_info,
            rewards_exp2=None, 
            expert_states=np.full((n_episodes_total, max_steps), np.nan),
            expert_actions=np.full((n_episodes_total, max_steps), np.nan),
            n_episodes=n_teach_episodes,  # Learners only learn during the teaching episodes
            training_split=1.0,  # always "training" during teaching
            max_steps=max_steps,
            optimization=False,
            exp=exp
        )
        for i in range(n_learner)
    }

    # Initialize teacher predictions
    teacher_predictions = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(dict)
            )
        )

    train_episodes = n_episodes_total - n_teach_episodes

    ## LOOP OVER EPISODES ##
    for episode in range(n_episodes_total):
        
        value_steps_list = []

        # Get rewards
        reward_placed = rewards_info[-n_episodes_total:, :][episode][:,:2]

        # Get location
        agent_location = env.initial_loc(exp = "baseline")
        state = world[agent_location]
        state_mat[episode, 0] = state

        # Sample n_steps from Poisson distribution with mean lambda 
        n_steps = rng.poisson(params['lambda'], size=max_steps)

        # Initialize start positions for learner candidates
        if episode >= train_episodes:
            teach_episode = 0
            for learner in learner_candidates.values():
                learner.learning_mode = learner.TRAINING
                learner.initialize_start_position()

        # Loop over steps
        for t in range(max_steps):

            # --- Training (identical to mb_expert) ---
            if episode < train_episodes:
                # Chose the next action randomly
                _, action = softmax_policy(value, state, env.n_actions, params['beta'], rng)
            
                # Calculate new location based on the action
                next_agent_location, next_state = env.move_agent(action, state, agent_location, reward_placed)

                # Observe reward for that action 
                reward = find_reward(state, reward_placed)  
                reward_per_step[episode, t] = reward 
                reward_sums_epi[episode] += reward

                # Update the value function based on the observed reward and transition
                value = q_learning(value, state, action, next_state, reward, params['gamma'], params['alpha'])
            
                # Update reward model
                model_r = model_update(model_r, state, action, reward, next_state)
                
                # DYNA-Q planning 
                value = dynaq_planner(value, 
                                    model_r, 
                                    state, action, next_state, 
                                    n_steps[t], params['gamma'], params['alpha'], rng)


                # Update transition beliefs
                kronecker_delta = np.zeros((env.n_states,))
                if next_state == None:
                    next_state = state
                kronecker_delta[next_state] = 1
                transition_belief[state, action] = transition_belief[state, action] + params['alpha_t']*(kronecker_delta - transition_belief[state, action])
                # Normalize the transition matrix
                transition_belief[state, action, :] = transition_belief[state, action, :] / np.sum(transition_belief[state, action, :])

                value_steps_list.append(np.copy(value))          

            # --- Teaching ---
            else:
                for a_T in range(env.n_actions):
                    for learner_id, learner in learner_candidates.items():
                        # Simulate learner's behavior 
                        Q_sim, learner_action_sim = learner.simulate_single_step(reward_placed, expert_state = state, expert_action = a_T)
                        teacher_predictions[teach_episode][t][a_T][learner_id] = {
                            'Q': Q_sim,
                            'action': learner_action_sim
                        }
            
                # Choose the next action to demonstrate based on the teaching objective
                action = teaching_objective(
                    method=objective,
                    predictions_at_t=teacher_predictions[teach_episode][t],
                    learner_candidates=learner_candidates,
                    teacher_Q=value,
                    env=env,
                    rng=rng,
                    reward_placed=reward_placed,
                    expert_state=state,
                    horizon=max_steps - t,
                    n_actions=env.n_actions,
                )

                # Update the learners based on the observed expert action
                for learner in learner_candidates.values():
                    learner.perform_single_step(reward_placed, expert_state = state, expert_action = action)

                # Calculate new location based on the action
                next_agent_location, next_state = env.move_agent(action, state, agent_location, reward_placed)

                # Observe reward for that action 
                reward = find_reward(state, reward_placed)  
                reward_per_step[episode, t] = reward 
                reward_sums_epi[episode] += reward

                teach_episode += 1


            # Store action
            action_mat[episode,t] = action
            
            # If positive reward is found, end episode
            if reward > 0:
                steps_to_reward[episode] = t + 1
                break 
            
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
        # Only return the reward sums per episode for the optimization procedure
        return reward_sums_epi
    
    else:
       return value, reward_sums_epi, state_mat, action_mat, steps_to_reward, transition_belief, model_r, value_perepi, belief_perepi, reward_per_step, teacher_predictions