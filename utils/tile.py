import numpy as np


class Tile():
    """Class for the tile in the village world. Each tile is a 4x4 grid of states with a reward state and boundaries."""
    
    # Attributes
    def __init__(self,xdim, ydim, start_state, end_state, reward_state, boundaries, rotation):
        # Initialize a 5x5 grid of states. 
        # Each state can be represented as a tuple, a number, or any data structure that suits your need
        # self.states = [[0 for _ in range(4)] for _ in range(4)]
        self.xdim = xdim
        self.ydim =ydim
        self.rotation = rotation
        self.states = np.arange(start_state, end_state).reshape((self.xdim, self.ydim)) # assign integer values to grid pos
        # self.name = name
        #self.rewards = rewards
        #self.rewards = self.get_rewards(reward_states)
        #self.v_boundaries = v_boundaries
        #self.h_boundaries = h_boundaries
        self.boundaries = boundaries  # Initialize a set to store impassable boundaries
        self.reward_state = reward_state

    
    def states_rotate(self): # Called from world.py
        # Rotate the tile based on the specified rotation angle
        if self.rotation == 90:
            self.states = np.rot90(self.states, 1)  # Rotate 90 degrees counterclockwise
        elif self.rotation == 180:
            self.states = np.rot90(self.states, 2)  # Rotate 180 degrees
        elif self.rotation == 270:
            self.states = np.rot90(self.states, 3)  # Rotate 270 degrees counterclockwise
    
    def is_boundary(self, state1, state2):
        # Check if the transition between state1 and state2 is a boundary
        boundary_tupple = (state1, state2) 
        return boundary_tupple in self.boundaries or boundary_tupple[::-1] in self.boundaries 
    

    def __repr__(self):
        return f"{self.states}" # string representation for debugging/printing instances
    
    