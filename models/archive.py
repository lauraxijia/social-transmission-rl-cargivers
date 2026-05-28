import numpy as np
from utils.helper_functions import softmax_policy, q_learning, value_shaping, find_reward

class MFValueShapingAgent:
    """
    A class implementing a model-free value shaping learner. 
    This learner augments its value function for observed actions of an expert during training.
    This learner does not learn a model of the environment.

    Attributes:  
    """
    # TODO:
    # - What of these need to be attributes of the class? Which can be just local variables in the function?
    # - Add functionality for simulations
    # - ggf. refactor    
    # - adapt rest of simulation pipeline to work with class-based agent
    def __init__(self, rng, params, env, world, rewards_info, rewards_exp2, expert_states, expert_actions, n_episodes, training_split, max_steps, optimization = False, world_model = "baseline"):
        
        # Learning parameters
        self.alpha = params['alpha'] # learning rate
        self.beta  = params['beta']  # inverse temperature for softmax action selection
        self.gamma = params['gamma'] # discount factor
        self.kappa = params['kappa'] # value shaping bonus

        # Simulation configuration
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.training_split = training_split
        self.learning_mode = None # "training" or "test"
        self.optimization = optimization

        # Environment   
        self.rng = rng # for reproducibility
        assert world_model in ['baseline', 'exp2', 'exp3'], "exp must be 'baseline', 'exp2', or 'exp3'"
        self.world_model = world_model
        self.env = env
        self.world = world

        # Rewards
        self.rewards_info = rewards_info[-self.n_episodes:, :]
        self.rewards_exp2 = rewards_exp2[-self.n_episodes:, :]

        # Expert states and actions
        self.expert_states, self.expert_actions = expert_states[-self.n_episodes:,:], expert_actions[-self.n_episodes:,:]

        # Representations
        self.Q = np.ones((env.n_states, env.n_actions))

        # Tracking
        self.current_episode = 0
        self.current_step = 0
        self.initial_location = None
        self.current_location = None
        self.current_state = None
        self.history = {
            'values': None, # Final value function at the end of the simulation
            'states': np.nan*np.zeros((n_episodes, max_steps+1), dtype=int),
            'actions': np.nan*np.zeros((n_episodes, max_steps)),
            'steps_to_reward': np.nan*np.zeros((n_episodes,)),
            'value_snapshots': np.zeros((n_episodes, env.n_states, env.n_actions)), # Q values per epsiode
            'reward_per_step': np.zeros((n_episodes, max_steps)),
            'reward_sum_episode': np.zeros((n_episodes))
        }

    # --- Agent functions ---
    def act(self, state):
        """Sample action from softmax policy over current Q values."""
        _, action = softmax_policy(self.Q, state, self.env.n_actions, self.beta, self.rng)
        return action
    
    def learn(self, state, action, reward, next_state):
        """Q-learning update."""
        self.Q = q_learning(self.Q, state, action, next_state, reward, self.gamma, self.alpha)

    def value_shaping_bonus(self, expert_state, expert_action):
        """Add value shaping bonus for observed expert action in expert state."""
        self.Q = value_shaping(self.Q, expert_action, expert_state, self.kappa) 

    # --- Single full learning step ---
    def perform_single_step(self, reward_placed, expert_state, expert_action, learning_mode):
        """Perform a single step of interaction with the environment, including social learning from expert if in training phase."""
        if learning_mode == "training" and not np.isnan(expert_action):
            # Apply value shaping bonus for observed expert action in expert state
            self.value_shaping_bonus(expert_state, expert_action)

        # Select action based on softmax policy
        action = self.act(self.current_state)
        # Track action
        self.history['actions'][self.current_episode, self.current_step] = action

        # Calculate new location based on the action
        next_agent_location, next_state = self.env.move_agent(action, self.current_state, self.current_location, reward_placed)

        # Observe reward for that action
        reward = find_reward(self.current_state, reward_placed)
        # Track rewards
        self.history['reward_per_step'][self.current_episode, self.current_step] = reward
        self.history['reward_sum_episode'][self.current_episode] += reward
        if reward > 0:
            return True  # End step if reward is found

        # Q-learning update
        self.learn(self.current_state, action, reward, next_state)

        # Update current state and location
        self.current_state = next_state
        self.current_location = next_agent_location
        # Track state 
        self.history['states'][self.current_episode, self.current_step + 1] = next_state
        return False

    # --- Main loop over all episodes ---
    def run_full_simulation(self):
        """Run the full simulation over all episodes, including training and test phase."""

        for episode in range(self.n_episodes):
            self.current_episode = episode
            self.learning_mode = "training" if episode < (self.n_episodes * self.training_split) else "test"

            # Place agent at random initial location (training is always identical to baseline)
            if self.learning_mode == "training":
                self.current_location = self.env.initial_loc(exp = "baseline")
            else:
                self.current_location = self.env.initial_loc(exp = self.world_model)
            self.current_state = self.world[self.current_location]
            # Track initial state          
            self.history['states'][self.current_episode, 0] = self.current_state

            found_reward = False
            for step in range(self.max_steps):
                self.current_step = step

                # Training phase: Social learning from expert
                if self.learning_mode == "training": 
                    # Get reward values for that episode
                    reward_placed = self.rewards_info[episode][:,:2]

                    # Get expert state and action for that episode and step
                    expert_state, expert_action = self.expert_states[episode, step], self.expert_actions[episode, step]
                    
                    # Perform single step with value shaping 
                    found_reward = self.perform_single_step(reward_placed, expert_state, expert_action, self.learning_mode)
                
                # Test phase: Individual learning without expert influence
                else: 
                    # Get reward values for that episode
                    if self.world_model == "exp2":
                        if self.rewards_exp2 is not None:
                            reward_placed = self.rewards_exp2[episode][:,:2]
                        else:
                            raise ValueError("rewards_exp2 must be provided for exp2")
                    else:
                        reward_placed = self.rewards_info[episode][:,:2]

                    # No expert influence during test phase, so pass NaN for expert state and action
                    found_reward = self.perform_single_step(reward_placed, np.nan, np.nan, self.learning_mode)
            
                # End episode if reward is found
                if found_reward:
                    break  

            # Track steps to reward and value function at the end of the episode
            self.history['steps_to_reward'][self.current_episode] = self.current_step + 1
            self.history['value_snapshots'][self.current_episode, :, :] = np.copy(self.Q)

        if self.optimization:
            # Only return final reward sum for each episode for optimization purposes
            return self.history['reward_sum_episode']
        else:
            # Return full history for analysis
            self.history['values'] = np.copy(self.Q)  # Final value function at the end of the simulation
            return self.history
            






    
    