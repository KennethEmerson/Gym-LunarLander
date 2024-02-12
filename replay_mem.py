"""
    file contains the Replay Memory Class which is used by the agent
"""
from collections import deque
import numpy as np
import torch
import random


class Replay_Memory:
    def __init__(self,memory_size,batch_size,state_dim,action_dim):
        """initialize the replay memory object

        Args:
            memory_size (int): total size of the memory in number of samples
            batch_size (int): number of samples in each batch for the network update
            state_dim (int): dimensionality of the statespace
            action_dim (int): dimensionality of the actionspace
        """

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.samples_in_memory = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()
    

    def reset(self):
        """ resets the replay memory
        """

        self.mem_pointer = 0
        self.states = torch.zeros((self.memory_size,self.state_dim)).float()
        self.non_terminals = torch.zeros(self.memory_size).bool()
        self.actions = torch.zeros(self.memory_size,1).long()
        self.rewards = torch.zeros(self.memory_size).float()
        self.next_states = torch.zeros((self.memory_size,self.state_dim)).float()


    def add_timestep(self,state,non_terminal,action,reward,next_state):
        """adds a timestep/sample to the memory

        Args:
            state (np.array): the actual state from which the agent moved
            non_terminal (bool): is the next_state a non terminal state
            action (int): the action taken by the agent in "state"
            reward (float): the reward received by entering next_state
            next_state (np.array): the state in which the agent arrived after performing action in state
        """

        assert(state.shape == (self.state_dim ,))
        assert(next_state.shape == (self.state_dim ,))
        self.states[self.mem_pointer] = torch.from_numpy(state.flatten())
        self.non_terminals[self.mem_pointer] = non_terminal
        self.actions[self.mem_pointer] = action
        self.rewards[self.mem_pointer] = reward
        self.next_states[self.mem_pointer] = torch.from_numpy(next_state.flatten())
        self.mem_pointer += 1
        if self.mem_pointer >= self.memory_size:
            self.mem_pointer = 0

        if self.samples_in_memory < self.memory_size:
            self.samples_in_memory += 1
        

    def is_batch_size_reached(self):
        """checks if the minimum number of samples is in the memory to create a batch

        Returns:
            [bool]: true is the minimal number of samples is available to create a batch
        """

        return self.samples_in_memory >= self.batch_size
    

    def get_new_batch(self):
        """creates a new batch from the replay memory

        Returns:
            tuple: tuple containing the tensors with states,non_terminals,actions,rewards,next_states
        """
        
        assert self.samples_in_memory >= self.batch_size

        #only takes actual samples into account for sampling minibatch hence "samples_in_memory"
        samples = random.sample(range(self.samples_in_memory), self.batch_size) 
        indexes = torch.tensor(samples)
        
        states = self.states.index_select(0,indexes)
        non_terminals = self.non_terminals.index_select(0,indexes)
        actions = self.actions.index_select(0,indexes)
        rewards = self.rewards.index_select(0,indexes)
        next_states = self.next_states.index_select(0,indexes)
        
        return states,non_terminals,actions,rewards,next_states
