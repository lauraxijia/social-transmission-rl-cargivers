import numpy as np
from utils.helper_functions import softmax_policy, q_learning, value_shaping, find_reward

class MFValueShapingAgent:
    """
    A class implementing a model-free value shaping learner. 
    This learner augments its value function for observed actions of an expert during training.
    This learner does not learn a model of the environment.

    """

    TRAINING = "training"
    TEST = "test"

    def __init__(self, rng, params, env, world, rewards_info, rewards_exp2, expert_states, expert_actions, n_episodes, training_split, max_steps, optimization = False, exp = "baseline"):
        
        # Learning parameters
        self.alpha = params['alpha'] # learning rate
        self.beta  = params['beta']  # inverse temperature for softmax action selection
        self.gamma = params['gamma'] # discount factor
        self.kappa = params['kappa'] # value shaping bonus

        # Simulation configuration
        self.n_episodes = int(n_episodes)
        self.n_training_episodes = int(self.n_episodes * training_split)
        self.max_steps = int(max_steps)
        self.learning_mode = None # "training" or "test"
        self.optimization = optimization

        # Environment   
        self.rng = rng # for reproducibility
        assert exp in ['baseline', 'exp2', 'exp3'], "exp must be 'baseline', 'exp2', or 'exp3'"
        self.exp = exp
        self.env = env
        self.world = world

        # Rewards
        self.rewards_info = rewards_info[-self.n_episodes:, :, :2]
        self.rewards_exp2 = rewards_exp2[-self.n_episodes:, :, :2] if rewards_exp2 is not None else None

        # Expert states and actions
        self.expert_states  = expert_states[-self.n_episodes:,:]
        self.expert_actions = expert_actions[-self.n_episodes:,:]

        # Representations
        self.Q = np.ones((env.n_states, env.n_actions))

        # Tracking
        self.current_episode = 0
        self.current_step = 0
        self.initial_location = None
        self.current_location = None
        self.current_state = None
        self.log = {
            'final_values': None, # Final value function at the end of the simulation
            'states': np.nan*np.zeros((self.n_episodes, self.max_steps + 1), dtype=int),
            'actions': np.nan*np.zeros((self.n_episodes, self.max_steps)),
            'steps_to_reward': np.nan*np.zeros((self.n_episodes,)),
            'value_snapshots': np.zeros((self.n_episodes, env.n_states, env.n_actions)), # Q values per epsiode
            'reward_per_step': np.zeros((self.n_episodes, self.max_steps)),
            'reward_sum_episode': np.zeros((self.n_episodes))
        }

    # --- Agent functions --- #
    def act(self, values, state):
        """Sample action from softmax policy over current Q values."""
        _, action = softmax_policy(values, state, self.env.n_actions, self.beta, self.rng)
        return action
    
    def learn(self, values, state, action, reward, next_state):
        """Q-learning update."""
        return q_learning(values, state, action, next_state, reward, self.gamma, self.alpha)
    
    def observe_expert(self, episode, step):
        """Observe expert state and action for the current episode and step."""
        return self.expert_states[episode, step], self.expert_actions[episode, step]

    def value_shaping_bonus(self, values,expert_state, expert_action):
        """Add value shaping bonus for observed expert action in expert state."""
        return value_shaping(values, expert_action, expert_state, self.kappa) 
    
    # --- Helper functions --- #
    def initialize_start_position(self):
        """Initialize the agent's starting position based on the current learning mode."""
        exp = "baseline" if self.learning_mode == self.TRAINING else self.exp
        self.current_location = self.env.initial_loc(exp=exp)
        self.current_state = self.world[self.current_location]

    def found_reward(self, reward):
        """Helper function to check if a reward was found."""
        return reward > 0
    
    def get_rewards_for_episode(self, episode):
        """Helper function to get the reward placements for the current episode based on the world model."""
        if self.exp == "exp2" and self.learning_mode == self.TEST:
            if self.rewards_exp2 is None:
                raise ValueError("rewards_exp2 must be provided for exp2!")
            rewards = self.rewards_exp2
        else:             
            rewards = self.rewards_info
        return rewards[episode]

    # --- Single full learning step --- #
    def perform_single_step(self, reward_placed, expert_state, expert_action):
        """Perform a real step of interaction with the environment, including social learning from expert if in training phase and logging."""

        episode, step = self.current_episode, self.current_step

        # 1. Optional value shaping based on observed expert action
        if self.learning_mode == self.TRAINING and not np.isnan(expert_action):
            self.Q = self.value_shaping_bonus(self.Q, expert_state, expert_action)

        # 2. Select action based on softmax policy
        action = self.act(self.Q, self.current_state)
        self.log['actions'][episode, step] = action

        # 3. Compute next state based on action
        next_agent_location, next_state = self.env.move_agent(action, self.current_state, self.current_location, reward_placed)

        # 4. Get reward 
        reward = find_reward(self.current_state, reward_placed)
        self.log['reward_per_step'][episode, step] = reward
        self.log['reward_sum_episode'][episode] += reward

        # 5. Check if reward is found
        if self.found_reward(reward):
            return True  

        # 6. Q-learning update
        self.Q = self.learn(self.Q, self.current_state, action, reward, next_state)

        # 7. Move to next state
        self.current_state = next_state
        self.current_location = next_agent_location
        self.log['states'][episode, step + 1] = next_state

        return False
    
    # --- Simulate single learning step --- #
    def simulate_single_step(self, reward_placed, expert_state, expert_action):
        """Simulate the outcome of a single learning step without actually performing the step."""

        Q_sim = np.copy(self.Q)

        # 1. Optional value shaping based on observed expert action
        if self.learning_mode == self.TRAINING and not np.isnan(expert_action):
            Q_sim = self.value_shaping_bonus(Q_sim, expert_state, expert_action)

        # 2. Select action based on softmax policy
        action = self.act(Q_sim, self.current_state)

        # 3. Compute next state based on the action
        _, next_state = self.env.move_agent(action, self.current_state, self.current_location, reward_placed)

        # 4. Get reward
        reward = find_reward(self.current_state, reward_placed)

        # 5. Q-learning update
        Q_sim = self.learn(Q_sim, self.current_state, action, reward, next_state)
        return Q_sim, action


    # --- Main loop over all episodes --- #
    def run_full_simulation(self):
        """Run the full simulation over all episodes, including training and test phase."""

        for episode in range(self.n_episodes):
            self.current_episode = episode

            self.learning_mode = (
                self.TRAINING if episode < self.n_training_episodes else self.TEST
            )

            # Place agent at random initial location 
            self.initialize_start_position()     
            self.log['states'][episode, 0] = self.current_state

            found_reward = False

            for step in range(self.max_steps):
                self.current_step = step

                reward_placed = self.get_rewards_for_episode(episode)

                # In training phase, observe expert
                if self.learning_mode == self.TRAINING: 
                    expert_state, expert_action = self.observe_expert(episode, step)
                else:
                    expert_state, expert_action = np.nan, np.nan
                    
                # Perform single step 
                found_reward = self.perform_single_step(reward_placed, expert_state, expert_action)
                # End episode if reward is found
                if found_reward:
                    break  

            # Track steps to reward and value function at the end of the episode
            self.log['steps_to_reward'][episode] = self.current_step + 1
            self.log['value_snapshots'][episode] = np.copy(self.Q)
            
        # Only return final reward sum for each episode for optimization purposes
        if self.optimization:
            return self.log['reward_sum_episode']

        # Final value function at the end of the simulation
        self.log['final_values'] = np.copy(self.Q)  
        return self.log
            






    
    