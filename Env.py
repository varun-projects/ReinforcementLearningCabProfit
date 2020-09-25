# Import routines

import numpy as np
import math
import random
from itertools import permutations,product

# Defining hyperparameters
num_cities = 5 # number of cities, ranges from 1 ..... m
num_hours = 24 # number of hours, ranges from 0 .... t-1
num_days = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)] + list(permutations(range(0, num_cities), 2))
                
        self.state_space = list(product(*[list(range(0, num_cities)), list(range(0, num_hours)), list(range(0, num_days))]))
        
        self.num_actions = len(self.action_space)

        #Variables for On-Hot Encoding
        self.eye_loc=np.eye(num_cities)
        self.eye_hour=np.eye(num_hours)
        self.eye_day=np.eye(num_days)
        
        # adding the possible and not possible actions vector in the input encoding vector
        self.state_onehot_size = self.num_actions + num_cities + num_hours + num_days

        self.Time_matrix = np.load("TM.npy")

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state, pos_actions_indices):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
                
        """-----------------------------------------------------Fix One---------------------------------------------------------
        To solve the the problem of the model picking wrong actions I added the possible actions and not possible actions in one
        encoding vector possible (1) and not possible (-1). that's it :) 
        -----------------------------------------------------------------------------------------------------------------------"""
        
        pos_actions_onehot = np.zeros(self.num_actions)
        
        pos_actions_onehot.fill(-1.0)
        
        for index in pos_actions_indices:
            pos_actions_onehot[index] = 1.0
            
        
        return np.hstack([pos_actions_onehot, self.eye_loc[state[0]], self.eye_hour[state[1]], self.eye_day[state[2]]])


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        requests = 0
        if location == 0:
            requests = np.random.poisson(2)

        if location == 1:
            requests = np.random.poisson(12)
        
        if location == 2:
            requests = np.random.poisson(4)

        if location == 3:
            requests = np.random.poisson(7)

        if location == 4:
            requests = np.random.poisson(8)


        if requests >15:
            requests = 15
        
        if requests == 0:
            return [0], [(0, 0)]
        else:
            possible_actions_index = random.sample(range(0, self.num_actions), requests)
             # (0,0) is not considered as customer request
            actions = [self.action_space[i] for i in possible_actions_index]

            return possible_actions_index, actions

     # see if the state is a terminal state or not.
    def is_terminal_state(self, state):
        if state[1] == 23 and state[2] == 6:
            return True
        
        return False

    def training_reward_func(self, state, action, possible_actions):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        """-----------------------------------------------------Fix Two---------------------------------------------------------
        If the model choice an action which isn't allowed, it receives massive punishment in the form of very large
        minus reward number. However, it's not enough. the agent must receive a small positive reward number if choose an action
        that is allowed. that's it :) 
        -----------------------------------------------------------------------------------------------------------------------"""
        
        reward = 0
        
        if action not in possible_actions:
            return -100000
        
        if action == (0,0):
            reward = -C
        else:
            reward = (R*self.Time_matrix[action[0]][action[1]][state[1]][state[2]]) - C*(self.Time_matrix[action[0]][action[1]][state[1]][state[2]] + self.Time_matrix[state[0]][action[0]][state[1]][state[2]])
            
        return reward

    
    def acutal_reward_func(self, state, action):
        
        reward = 0

        if action == (0,0):
            reward = -C
        else:
            reward = (R*self.Time_matrix[action[0]][action[1]][state[1]][state[2]]) - C*(self.Time_matrix[action[0]][action[1]][state[1]][state[2]] + self.Time_matrix[state[0]][action[0]][state[1]][state[2]])
            
        return reward
        

    def next_state_func(self, state, action, possible_actions):
        """Takes state and action as input and returns next state"""

        if action not in possible_actions:
            return False, state
        
        if action == (0,0):
            if state[1] >= 23:
                next_state = (state[0],0 , state[2]+1)
                if state[2] >= 6:
                    next_state = (state[0],0 , 0)
            else: 
                next_state = (state[0], state[1] + 1, state[2])

        else:
            ride_time = self.Time_matrix[action[0]][action[1]][state[1]][state[2]] + self.Time_matrix[state[0]][action[0]][state[1]][state[2]]
            #print(ride_time)
            next_hr_day = int((state[1]+ride_time)%24)
            
            next_state = (action[1], next_hr_day, state[2])
            if state[1] == 23:
                next_state = (action[1], next_hr_day, state[2]+1)
                if state[2] >= 6:
                    next_state = (action[1], next_hr_day, 0)
            

        return True, next_state


    def reset(self):
        return (random.choice(range(num_cities)), 0, 0)