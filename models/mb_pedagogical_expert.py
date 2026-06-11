from collections import defaultdict

import numpy as np
from utils.helper_functions import softmax_policy, q_learning, dynaq_planner, find_reward, model_update, teaching_objective

"""Policy of model-based pedagogical expert."""

def mb_pedagogical_expert(params, env, world, rewards_info, max_steps, n_episodes_total, n_teach_episodes, rng, optimization = False, exp='baseline', learner_function = None, learner_params = None, n_learner = None, n_test_episodes = 0, kappa_show = 1.0, *, objective):

    # Assert that exp is either 'baseline' or 'exp2' or 'exp3'
    assert exp in ['baseline', 'exp2', 'exp3'], "exp must be 'baseline', 'exp2', or 'exp3'"

    ## Phase sizes ##
    # The n_social social episodes are split into a teaching phase (teacher
    # demonstrates, learner value-shapes) followed by a test phase (teacher
    # leaves, learner runs solo). They sit after the asocial training episodes.
    n_social = n_teach_episodes + n_test_episodes
    train_episodes = n_episodes_total - n_social

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

    # Initialize dictionary of learners for n_learner. Each learner is active for
    # all n_social episodes: the first n_teach_episodes are its (social) training
    # episodes, the remaining n_test_episodes are solo test episodes.
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
            n_episodes=n_social,  # learner log spans teaching + test episodes
            training_split=(n_teach_episodes / n_social) if n_social else 1.0,
            max_steps=max_steps,
            optimization=False,
            exp=exp
        )
        for i in range(n_learner)
    }

    # Initialize teacher predictions, keyed by [social episode][step][demo action][learner]
    teacher_predictions = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(dict)
            )
        )

    ## LOOP OVER EPISODES ##
    for episode in range(n_episodes_total):

        value_steps_list = []

        # Get rewards
        reward_placed = rewards_info[-n_episodes_total:, :][episode][:,:2]

        # Sample n_steps from Poisson distribution with mean lambda
        n_steps = rng.poisson(params['lambda'], size=max_steps)

        # --- Asocial teacher training (identical to mb_expert) ---
        if episode < train_episodes:

            # Get location
            agent_location = env.initial_loc(exp = "baseline")
            state = world[agent_location]
            state_mat[episode, 0] = state

            for t in range(max_steps):
                # Chose the next action randomly
                _, action = softmax_policy(value, state, env.n_actions, params['beta'], rng)

                # Calculate new location based on the action
                next_agent_location, next_state = env.move_agent(action, state, agent_location, reward_placed)

                # Observe reward for that action
                reward = find_reward(state, env, reward_placed)
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

                # Store action
                action_mat[episode, t] = action

                # If positive reward is found, end episode
                if reward > 0:
                    steps_to_reward[episode] = t + 1
                    break

                # Update the state
                state = next_state
                agent_location = next_agent_location
                state_mat[episode, t+1] = state

            value_steps_list_epi.append(value_steps_list)
            value_finalepi_list.append(np.copy(value))
            # If the episode doesnt terminate, steps_to_reward stays at t+1
            steps_to_reward[episode] = t+1
            continue

        # --- Social phase: teaching then test ---
        se = episode - train_episodes  # 0-indexed social episode
        is_teaching = se < n_teach_episodes

        # (Re)initialize every learner at the start of each social episode and
        # sync the counters used for logging (this is what makes the learner's
        # per-episode trajectory and performance get recorded correctly).
        for learner in learner_candidates.values():
            learner.learning_mode = learner.TRAINING if is_teaching else learner.TEST
            learner.current_episode = se
            learner.current_step = 0
            learner.initialize_start_position()
            learner.log['states'][se, 0] = learner.current_state

        # Track which learners have already reached the reward this episode so we
        # stop stepping them (otherwise they would re-collect the reward).
        finished_learners = set()

        if is_teaching:
            # Teacher is present and demonstrates. Set up its own trajectory.
            agent_location = env.initial_loc(exp = "baseline")
            state = world[agent_location]
            state_mat[episode, 0] = state

            for t in range(max_steps):
                active = [lid for lid in learner_candidates if lid not in finished_learners]

                if active:
                    # Simulate each candidate demo action for every active learner
                    for a_T in range(env.n_actions):
                        for learner_id in active:
                            learner = learner_candidates[learner_id]
                            Q_sim, learner_action_sim = learner.simulate_single_step(reward_placed, expert_state = state, expert_action = a_T)
                            teacher_predictions[se][t][a_T][learner_id] = {
                                'Q': Q_sim,
                                'action': learner_action_sim
                            }

                    # Choose the next action to demonstrate based on the objective
                    action = teaching_objective(
                        method=objective,
                        predictions_at_t=teacher_predictions[se][t],
                        learner_candidates=learner_candidates,
                        teacher_Q=value,
                        env=env,
                        rng=rng,
                        reward_placed=reward_placed,
                        expert_state=state,
                        horizon=max_steps - t,
                        n_actions=env.n_actions,
                        kappa_show=kappa_show,
                    )

                    # Active learners observe the demo and take their own step
                    for learner_id in active:
                        learner = learner_candidates[learner_id]
                        learner.current_step = t
                        found = learner.perform_single_step(reward_placed, expert_state = state, expert_action = action)
                        if found:
                            learner.log['steps_to_reward'][se] = t + 1
                            finished_learners.add(learner_id)
                else:
                    # No learners left to teach: teacher follows its own greedy policy
                    action = int(np.argmax(value[state]))

                # Calculate new location based on the action
                next_agent_location, next_state = env.move_agent(action, state, agent_location, reward_placed)

                # Observe reward for that action
                reward = find_reward(state, env, reward_placed)
                reward_per_step[episode, t] = reward
                reward_sums_epi[episode] += reward

                # Store action
                action_mat[episode, t] = action

                # If positive reward is found, end episode
                if reward > 0:
                    steps_to_reward[episode] = t + 1
                    break

                # Update the state
                state = next_state
                agent_location = next_agent_location
                state_mat[episode, t+1] = state

            steps_to_reward[episode] = t + 1

        else:
            # Test phase: the teacher has left, the learner(s) run solo with no
            # demonstration. The teacher trajectory (state_mat/action_mat) stays
            # NaN for these episodes.
            for t in range(max_steps):
                for learner_id, learner in learner_candidates.items():
                    if learner_id in finished_learners:
                        continue
                    learner.current_step = t
                    found = learner.perform_single_step(reward_placed, expert_state = np.nan, expert_action = np.nan)
                    if found:
                        learner.log['steps_to_reward'][se] = t + 1
                        finished_learners.add(learner_id)
                if len(finished_learners) == len(learner_candidates):
                    break

        # End-of-social-episode learner bookkeeping
        for learner_id, learner in learner_candidates.items():
            if learner_id not in finished_learners:
                # Did not reach the reward this episode
                learner.log['steps_to_reward'][se] = max_steps
            learner.log['value_snapshots'][se] = np.copy(learner.Q)

        value_finalepi_list.append(np.copy(value))

    # Collect each learner's full log (states, actions, rewards, steps_to_reward, ...)
    learner_logs = {i: lc.log for i, lc in learner_candidates.items()}

    if optimization:
        # Only return the reward sums per episode for the optimization procedure
        return reward_sums_epi

    else:
       return value, reward_sums_epi, state_mat, action_mat, steps_to_reward, transition_belief, model_r, value_perepi, belief_perepi, reward_per_step, teacher_predictions, learner_logs
