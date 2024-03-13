import numpy as np
import pandas as pd


class MonopolyEnv:
    def __init__(self, num_states, max_turns=100):
        self.actions = ["skip", "buy", "give"]
        self.x = 0
        self.done = False
        self.episode_length = 0
        self.no_operation = False
        
        self.num_states = num_states
        self.board = np.zeros(self.num_states)
        self.state_observation = [self.x, self.board]
        self.max_turns = max_turns
        
    def reset(self):
        self.done = False
        self.episode_length = 0
        self.x = 0
        self.board = np.zeros(self.num_states)
        self.state_observation = [self.x, self.board]
        return [self.x, self.board]
    
    def action_space(self):
        return self.actions

    def update_position_roll(self):
        # Update the game state based on the action
        # roll = np.random.randint(low=1, high=5)
        roll = np.random.randint(low=1, high=3)
        self.x = (self.x + roll) % self.num_states
        return [self.x, self.board]
  
    def step(self, action):
        
        if self.episode_length > self.max_turns:
            self.done = True
            self.no_operation = True
            return self.state_observation, self.reward, self.done, self.no_operation, self.episode_length
            
        if np.all(self.board == 2):
            self.done = True
            self.no_operation = True
            return self.state_observation, self.reward, self.done, self.no_operation, self.episode_length

        self.action = action
        self.reward = self.get_reward()
        self.state_observation = self.take_action()
        self.episode_length += 1
        self.no_operation = False
        
        if(self.episode_length >= self.max_turns):
            self.done = True
        
        return self.state_observation, self.reward, self.done, self.no_operation, self.episode_length
    
    def get_reward(self):
        '''
        Return value : rewards
        Input argument. 
        '''
        self.reward = 0
        if self.board[self.x] == 0:
            if self.action == "buy":
                self.reward += 0
            elif self.action == "give":
                # Invalid action
                self.reward += -10
            else:
                # Skipping even when it can buy
                self.reward += -2

        elif self.board[self.x] == 1:
            if self.action == "buy":
                # Trying to buy already bought land
                self.reward += -1
            elif self.action == "give":
                # Good action
                self.reward += 0
            else:
                # Skipping even when it can sell
                self.reward += -2

        else:
            if self.action == "buy":
                # Trying to buy already bought land
                self.reward += -1
            elif self.action == "give":
                # Invalid action
                self.reward += -1
            else:
                # Skipping correct action
                self.reward += 0


        if np.all(self.board == 2):
            self.reward += 3
        else:
            self.reward += -1
    
        return self.reward
    
    def take_action(self):
        if self.action == "buy":
            if int(self.board[self.x]) == 0:        
                self.board[self.x] = 1
                
        elif self.action == "give":
            if int(self.board[self.x]) == 1:
                self.board[self.x] = 2
                
        return [self.x, self.board]
    