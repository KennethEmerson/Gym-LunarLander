"""
    File contains the list of compared agents as used in the report
"""
from agent import Agent

def list1(state_dim,action_dim): 
    return [
        Agent(state_dim,action_dim,
                epsilon_decay= 200,
                repl_mem_size= 100000,
                repl_batch_size = 64,
                gamma = 0.99,
                learning_rate = 0.0001,
                loss = "SmoothL1",
                update_target_after_episode=1),

        Agent(state_dim,action_dim,
                epsilon_decay= 200,
                repl_mem_size= 100000,
                repl_batch_size = 64,
                gamma = 0.99,
                learning_rate = 0.0001,
                loss = "SmoothL1",
                update_target_after_episode=30), 

        Agent(state_dim,action_dim,
                epsilon_decay= 200,
                repl_mem_size= 100000,
                repl_batch_size = 64,
                gamma = 0.99,
                learning_rate = 0.0001,
                loss = "SmoothL1",
                update_target_after_episode=15),

        Agent(state_dim,action_dim,
                epsilon_decay= 200,
                repl_mem_size= 100000,
                repl_batch_size = 64,
                gamma = 0.99,
                learning_rate = 0.0001,
                loss = "SmoothL1",
                update_target_after_episode=50),      
        ]
