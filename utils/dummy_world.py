import numpy as np
from .tile import Tile

class DummyWorld():
    """
    A class to represent a dummy environment to test the pedagogical expert.

    Attributes:
        name (str): Name of the environment.
        n_states (int): Number of states in the environment.
        n_actions (int): Number of actions available to the agent.
        n_tiles (int): Number of tiles in the environment.
        dim_x (int): Dimension of the environment in the x direction.
        dim_y (int): Dimension of the environment in the y direction.
        tiles (list): List of Tile objects representing the tiles in the environment.
    """
    
    # Attributes
    def __init__(self, world = None, rng=None):
       self.name = "DummyWorld"
       self.n_states = 25
       self.n_actions = 4
       self.n_tiles = 1
       self.dim_x = 5
       self.dim_y = 5
       self.rng = rng if rng is not None else np.random.default_rng()

       # Create four tiles with appropriate starting numbers
       xdim_tiles, ydim_tiles = 5, 5
       
       # Reward 
       self.reward_states = [0]

       # Hazards 
       self.hazard_states = [7, 16] 
       self.hazard_penalty = -5

       assert not any(state in self.hazard_states for state in self.reward_states), "Reward states and hazard states cannot overlap."

       # No boundaries in the dummy world
       boundaries = []

       # Define the grid world as a single tile
       self.tile = Tile(xdim_tiles, ydim_tiles, 0, 25, self.reward_states, self.hazard_states, boundaries, None)
       self.world_matrix = self.tile.states
       self.init_transit_mat, self.true_transition_mat = self.transition_probabilities()
  
        
    # Move the agent
    def move_agent(self, action, state, agent_location, reward_placed):
        """
        Move the agent based on the action taken
        Args:
            action: The action taken by the agent
            state: The current state of the agent
            agent_location: The current location of the agent
        Returns:
            agent_location: The new location of the agent
            new_state: The new state of the agent
        """
        # Check if the current state is a reward state and reward is > 0, else continue searching
        if (state in reward_placed[:,0]) and (reward_placed[reward_placed[:,0] == state, 1][0] > 0):
                return None, None
        # Calculate new location based on action
        # Returns x and y
        new_location = self.calculate_new_location(action, agent_location) 
        # Calculates state from location => state 2
        new_state = self.get_state_from_location(new_location)

        # Check if new location crosses a boundary => state 1
        current_state = self.get_state_from_location(agent_location)
        
        # Check if there is a boundary between current state and next_state
        is_boundary = self.tile.is_boundary(current_state, new_state)

        # If it is boundary do not update the location and remain in state
        if self.tile.is_boundary(current_state, new_state):
            return agent_location, self.world_matrix[agent_location]
        else:
            return new_location, new_state
                
    # Get the new location given the action and current location
    def calculate_new_location(self, action, agent_location):
        """
        Calculate the new location of the agent based on the action taken
        Args:
            action: The action taken by the agent int(0-3)
            agent_location: The current location of the agent tuple(x,y)
        Returns:
            agent_location: The new location of the agent
        """
        # Calculate new location based on action
        x, y = agent_location
        if action == 0: # up
            x -= 1
        elif action == 1: # "right"
            y += 1
        elif action == 2: # "down"
            x += 1
        elif action == 3 : # "left"
            y -= 1
        else:
            print("Action not possible")
            
        # Check boundaries (8x8 grid) returns again agent location
        if 0 <= x < self.dim_x and 0 <= y < self.dim_y:
            agent_location = (x, y) 
            
            return agent_location
        else:
           # print("Move not allowed: Agent would move out of bounds")
            return agent_location
    
    # Transform location into state    
    def get_state_from_location(self, location):
        # Convert a location in the grid to a state number
        return self.world_matrix[location]

        
    def transition_probabilities(self):
        """
        Generate the initial and true transition probability matrices for the environment

        Args:
            -
        Returns:
            init_transit_mat: Initial transition probability matrix (100,4,100)
            Initial transition matrix is the same for all worlds. All transitions are possible
            true_transit_mat: True transition probability matrix (100,4,100)
        """
        
        init_transit_mat = np.zeros((self.n_states, self.n_actions, self.n_states))
        true_transit_mat = np.zeros((self.n_states, self.n_actions, self.n_states))

        # Action mappings
        action_dict = {
            0: (-1, 0),  # up
            1: (0, 1) ,  # right
            2: (1, 0),   # down
            3: (0, -1),  # left
        }
        
        # Populate the transition probability matrix
        for state in range(self.n_states):
            coords = np.where(self.world_matrix == state)
            x, y = coords[0][0], coords[1][0]

            # Determine valid actions and their probabilities
            for action in range(self.n_actions):
                dx, dy = action_dict[action]

                new_x, new_y = x + dx, y + dy

                # Check if the new state is within the grid boundaries
                if 0 <= new_x < self.dim_x and 0 <= new_y < self.dim_y:
                    
                    # Get the new state
                    new_state = self.world_matrix[new_x, new_y]
                    # Update INITIAL TM probability for the new state
                    init_transit_mat[state, action, new_state] = 1 

                    # Only for TRUE TM 
                    # Find in which tile it is so we can call self. 

                    is_boundary = self.tile.is_boundary(state, new_state)

                    # Check if there is a boundary between state and new_state 
                    if is_boundary:
                        # If yes, update same state with probability 1
                        true_transit_mat[state, action, state] = 1 # stays in the same state with p=1
                    # No boundary
                    else:
                        # Update TRUE TM probability for new state
                        true_transit_mat[state, action, new_state] = 1

                # If the next state is outside the grid boundaries, same state = 1
                else:
                    init_transit_mat[state, action, state] = 1
                    true_transit_mat[state, action, state] = 1
                

        return init_transit_mat, true_transit_mat
    
    def initial_loc(self, exp = "baseline"):
        """
        Randomly selects a initial location from the possible initial locations.
            baseline or exp2 - center of the world
            exp3 - 1 step away from center towards corners
        Returns:
            agent_location: Initial location of the agent
        """
        initial_loc = (4,4)
        agent_location = initial_loc    
        return agent_location
    
    def __repr__(self):
        # Optional: representation of the grid world and agent location
        return f"Agent Location: {self.agent_location}\n{self.world_matrix}\n{self.world_matrix[self.agent_location]}"
