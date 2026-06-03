import numpy as np
from .tile import Tile

class VillageWorld():
    """
    A class to represent the Village World environment.

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
       self.name = "VillageWorld"
       self.n_states = 100
       self.n_actions = 4
       self.n_tiles = 4
       self.dim_x = 10
       self.dim_y = 10
       self.n_tiles = 4
       self.rng = rng if rng is not None else np.random.default_rng()

       # Create four tiles with appropriate starting numbers
       xdim_tiles, ydim_tiles = 5, 5
       
       # Reward states
       self.reward_states = [16, 43, 57, 87]

       # Define boundaries within each tile
       boundary_tileA =[(9,14), (13,14), (13,18), (17,18), (17,22), (16,21), (15,16), (11,16), (11,12), (7,12), (8,7)]
       boundary_tileB = [(35,36), (40,41), (41,46), (42,47), (43,48),(43,44), (38,43), (37,42), (36,37)]
       boundary_tileC = [(67,68), (62,63), (57,62), (56,57), (52,57), (53,58), (58,59), (63,64), (68,69)] 
       boundary_tileD = [(75,80), (80,81),(81,86), (86,87), (87,92), (92,93)]

       # Define the rotation angles for later random choise
       possible_rotation_angles = np.array([0,90,180,270])
       rotation_angles = self.rng.choice(possible_rotation_angles, size=self.n_tiles, replace=True)
       
       # Initialize tiles with boundaries, rotation and reward states
       self.tileA = Tile(xdim_tiles, ydim_tiles, 0, 25, 16, boundary_tileA, rotation_angles[0])
       self.tileB = Tile(xdim_tiles, ydim_tiles, 25, 50, 43, boundary_tileB, rotation_angles[1])
       self.tileC = Tile(xdim_tiles, ydim_tiles, 50, 75, 57, boundary_tileC, rotation_angles[2])
       self.tileD = Tile(xdim_tiles, ydim_tiles, 75, 100, 87, boundary_tileD, rotation_angles[3])
       
       self.tiles = [self.tileA, self.tileB, self.tileC, self.tileD]
       
       self.world_matrix = self.get_world_matrix() if world is None else world
       self.init_transit_mat, self.true_transition_mat = self.transition_probabilities()

    # Design the world  
    def get_world_matrix(self):
        """
        Generate the world matrix by combining the states of the four tiles
        Returns:
            shape_world: The world matrix
        """

        #tiles = [self.tileA, self.tileB, self.tileC, self.tileD]

        # Apply rotation to each tile
        for tile in self.tiles:
            tile.states_rotate()
        
        # Shuffle the tiles to create different world configurations
        shuffled_tiles = self.rng.choice(self.tiles, size=len(self.tiles), replace=False)

        
        # Combine the states of all tiles into a single matrix
        top_row = np.hstack([shuffled_tiles[0].states, shuffled_tiles[1].states])
        bottom_row = np.hstack([shuffled_tiles[2].states, shuffled_tiles[3].states])
        
        world_map = np.vstack([top_row, bottom_row]) 
        
        return world_map         
    
    
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
        # Learn in which tile it is so we can call self. 
        current_tile, new_tile = self.get_tile_from_state(current_state), self.get_tile_from_state(new_state)
        
        # Check if there is a boundary between current state and next_state
        if current_tile.is_boundary(current_state, new_state):

            # If it is boundary do not update the location and remain in state
         
            return agent_location, self.world_matrix[agent_location]
            
        else: 
            # Update location if move is allowed
            agent_location = new_location
           
            
            return agent_location, new_state
        
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

    
    # Transform state into tile
    def get_tile_from_state(self, state):
        # Determine which tile a given state belongs to
        if state < 25:
            return self.tileA
        elif state < 50:
            return self.tileB
        elif state < 75:
            return self.tileC
        else:
            return self.tileD
        
    def get_all_boundaries(self): 
        """
        Returns a list with tuples with the states between which there is a boundary

        """        
        tiles = [self.tileA, self.tileB, self.tileC, self.tileD]
        
        boundaries = []
        # loop through the tiles and get the tile's boundaries
        for tile in tiles:
            boundaries.append(tile.boundaries) 
        
        flat_boundaries = [item for sublist in boundaries for item in sublist]
        return flat_boundaries
            

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
            #print(state)
            x, y = np.where(self.world_matrix == state)

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
                    current_tile, new_tile = self.get_tile_from_state(state), self.get_tile_from_state(new_state)


                    is_boundary = current_tile.is_boundary(state, new_state)

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
        if exp == "exp3":
            possible_initial_loc = [(3, 3), (4,3), (5, 3), (6,3), (6,4), (6,5), (6,6), (5,6), (4,6), (3,6), (3,5), (3,4)]
        else:
            possible_initial_loc = [(4,4), (5,4), (4,5), (5,5)]
        # Return random initial location 
        n_start_locations = len(possible_initial_loc)
        in_index = self.rng.integers(0, n_start_locations)
        agent_location = possible_initial_loc[in_index]
        # Make sure that the start location is not a reward state
        while self.world_matrix[agent_location] in self.reward_states:
            in_index = self.rng.integers(0, n_start_locations)
            agent_location = possible_initial_loc[in_index]
    
        return agent_location
    
    def __repr__(self):
        # Optional: representation of the grid world and agent location
        return f"Agent Location: {self.agent_location}\n{self.world_matrix}\n{self.world_matrix[self.agent_location]}"
