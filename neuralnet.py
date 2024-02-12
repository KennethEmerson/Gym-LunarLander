"""
    contains the neural network architecture for the agent
"""
from torch import nn, from_numpy
import numpy as np

class NeuralNet(nn.Module):
    
    def __init__(self,state_dim,action_dim):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                )
                    
    def forward(self, input):
        # converts numpy array to tensor
        if isinstance(input,np.ndarray):
            input = from_numpy(input).float()
        
        return self.model(input)