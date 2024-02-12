"""
    File contains the Agent class as used in the exercise
"""
import numpy as np
from torch import nn, optim ,tensor, argmax, save, load, no_grad, from_numpy
import random, math
from datetime import datetime
import os

from replay_mem import Replay_Memory
from neuralnet import NeuralNet

EPSILON_MAX = 0.9
EPSILON_MIN = 0.05


class Agent:
    def __init__(self,state_dim,
                action_dim,
                epsilon_decay,
                repl_mem_size,
                repl_batch_size,
                gamma,
                learning_rate,
                loss,
                update_target_after_episode):
        
        """ initializes and constructs the agent object

        Args:
            state_dim (int): dimensionality of the statespace 
            action_dim (int): dimensionality of the action space
            epsilon_decay (int): determines the speed of decay (smaller decays faster)
            repl_mem_size (int): total number of samples to be stored
            repl_batch_size (int): size of sample batch from the replay memory used for each training
            gamma (float): gamma hyperparameter
            learning_rate (float): hyperparameter used for the neural net optimizer
            loss (String): the Pytorch Loss function to be used two options: "MSE" or "SmoothL1"
            update_target_after_episode (int): interval between the Target network update in episodes
        """
        
        assert isinstance(state_dim, int) and state_dim>=1
        assert isinstance(action_dim, int) and action_dim>=2
        assert isinstance(repl_mem_size,int)
        assert isinstance(repl_batch_size,int) and repl_batch_size <= repl_mem_size
        assert loss =="MSE" or loss == "SmoothL1"
        
        self.state_dim =state_dim
        self.action_dim = action_dim
        
        self.epsilon_max = EPSILON_MAX
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = epsilon_decay

        self.repl_mem = Replay_Memory(repl_mem_size,repl_batch_size,state_dim,action_dim)

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.update_target_after_episode = update_target_after_episode

        # select the corresponding Loss function
        self.loss_name = loss
        if self.loss_name == "MSE":
            self.loss_function = nn.MSELoss()
        else:
            self.loss_function = nn.SmoothL1Loss()
        
        self.online_network = NeuralNet(state_dim,action_dim)
        self.target_network = NeuralNet(state_dim,action_dim)
        
        # make sure network are identical at start
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.optimizer = optim.Adam(self.online_network.parameters(),lr = learning_rate)
        
        # log of the trial (is stored in agent for reproducing plots)
        self.log_samples = None
        self.log_episodes = None
        self.log_rewards = None
        self.log_avg_rewards = None

        
    def __str__(self):
        """ gives a summary of the agent settings as a string

        Returns:
            [String]: summary of the agent configuration
        """

        return  f"e_decay: {self.epsilon_decay}, mem: {self.repl_mem.memory_size}, " \
                + f"batch: {self.repl_mem.batch_size}, loss: {self.loss_name}, g: {self.gamma}, " \
                + f"LR: {self.learning_rate}, TN_upd: {self.update_target_after_episode}"


    def init_logging(self,samples):
        """initializes the logs and sets the size in accordance to the number of expected samples given

        Args:
            samples (int): number of samples in total to log (can differ from episodes if not all episodes ar logged)
        """

        self.log_samples = samples
        self.log_sample = 0
        self.log_episodes = np.zeros(samples,dtype=int)
        self.log_rewards = np.zeros(samples,dtype=float)
        self.log_avg_rewards = np.zeros(samples,dtype=float)


    def log(self,episode,reward,avg_reward):
        """adds a sample to the log

        Args:
            episode (int): which episode the logged values are from
            reward (float): the actual reward for that episode
            avg_reward (float): the average reward in the specific episode
        """
        
        # make sure logs are initialised
        assert self.log_samples is not None
        
        # assure the logging memory is not full and then add sample
        if self.log_sample >= self.log_samples:
            print("[ERROR] sample logs full") 
        else:
            self.log_episodes[self.log_sample] = episode
            self.log_rewards[self.log_sample] = reward
            self.log_avg_rewards[self.log_sample] = avg_reward
        self.log_sample +=1


    def epsilon(self,episode):
        """ calculates the actual value of epsilon in repsect to decay

        Args:
            episode (int): the actual episode for which to calculate epsilon

        Returns:
            [float]: actual value of epsilon
        """

        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
                         math.exp(-1. * episode / self.epsilon_decay) 


    def select_action(self,state,episode,greedy=False):
        """ select action based on e-greedy

        Args:
            state (np.Array): the actual state of the agent
            episode (int): the episode of the game 
            greedy (bool, optional): if set to True the agent works greedy (always selects optimal action). 
                                     Defaults to False.
        
        Returns:
            [int]: the value of the action to take
        """

        with no_grad():
            # find optimal action
            action_values = self.online_network.forward(state)
            action = argmax(action_values).item()

            # choose random action
            if random.random() < self.epsilon(episode) and not greedy:
                    action = random.randrange(0,self.action_dim,1)
        
        return action


    def update_networks(self,state,is_terminal,action,reward,next_state,episode):
        """adds sample to replay memory, 
            perform Q-learning on online network if minimal number of samples in replay memory is reached 
            and update target network if interval is passed

        Args:
            state (np.Array): the state in which the agent performed the action
            is_terminal (bool): is the state in which the agent arrived a terminal state
            action (int): the index value of the action taken between state and next_state
            reward (float): the reward received for arriving in next_state
            next_state (np.Array): the state in which the agent arrived
            episode (int): the episode of the game 

        Returns:
            [float]: the loss value 
        """

        self.online_network.train()
        self.target_network.train()

        # REPLAY MEMORY STORES NON_TERMINALS NOT TERMINALS
        self.repl_mem.add_timestep(state,(not is_terminal),action,reward,next_state)

        # updating networks is only allowed if sufficient samples are in the replay memory
        if self.repl_mem.is_batch_size_reached():
            # collect a batch from the repay memory
            states,non_terminals,actions,rewards,next_states = self.repl_mem.get_new_batch()
            
            # get the predicted Q-values from the state actions out of the batch
            state_action_values = self.online_network(states).gather(1,actions).squeeze() 
            
            # gather only values for actual actions taken by mutliplying by non terminals
            next_state_action_values = self.target_network(next_states).max(1)[0]*(non_terminals)
            expected_state_action_values = rewards + (self.gamma * next_state_action_values)

            # calculate the loss and gradients
            loss = self.loss_function(state_action_values, expected_state_action_values)
            self.optimizer.zero_grad()
            loss.backward()
            
            # prevent exploding gradients
            for param in self.online_network.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            # copy policy network to target network at end of set number of episodes 
            if (is_terminal and episode%self.update_target_after_episode==0):
                self.target_network.load_state_dict(self.online_network.state_dict())
                self.timestep = 0

            return loss.item()


    def get_filename(self):
        """aux function that composes the filename for the saved agent

        Returns:
            [String]: filename for the stored agent
        """

        now = datetime.now()
        
        time_string = now.strftime("%Y%m%d %H%M%S")
        
        return  f"e{self.epsilon_decay}_mem{self.repl_mem.memory_size}_" \
                + f"batch{self.repl_mem.batch_size}_{self.loss_name}_g{self.gamma}_" \
                + f"LR{self.learning_rate}_TN_upd{self.update_target_after_episode}_" \
                + time_string + ".pt"  


    def save(self, checkpoint_name=None):
        """saves the (trained) agent together with the logging of the training 

        Args:
            checkpoint_name ([String], optional): an optional filename can be passed. Defaults to None.
                                                  if none is given, standard filename will be used from get_filename
        """

        path = os.getcwd()
        
        if checkpoint_name is None:
            checkpoint_name = self.get_filename()
        
        checkpoint = {
            'agent_name': self.__str__(),
            'state_dim': self.state_dim,
            'action_dim':  self.action_dim,
            'epsilon_decay': self.epsilon_decay,
            'repl_mem_size': self.repl_mem.memory_size,
            'repl_batch_size': self.repl_mem.batch_size,
            'gamma': self.gamma,
            'learning_rate':self.learning_rate,
            'update_after_episode': self.update_target_after_episode,
            'loss': self.loss_name,
            'policy_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'log_episodes': self.log_episodes,
            'log_reward': self.log_rewards,
            'log_avg_reward': self.log_avg_rewards
        }
        save(checkpoint, os.path.join(path,"saved_agents", checkpoint_name))


    @classmethod
    def load(cls,filename):
        """loads an existing agent from file

        Args:
            filename (String): the name of the file containing the agent

        Raises:
            OSError: error is raised when file could not get loaded

        Returns:
            [Agent]: the actual agent from the file
        """

        if os.path.isfile(filename):
            
            checkpoint = load(filename)

            agent = Agent(state_dim = checkpoint['state_dim'],
                            action_dim = checkpoint['action_dim'],
                            epsilon_decay = checkpoint['epsilon_decay'],
                            repl_mem_size = checkpoint['repl_mem_size'],
                            repl_batch_size = checkpoint['repl_batch_size'],
                            gamma = checkpoint['gamma'],
                            learning_rate = checkpoint['learning_rate'],
                            loss = checkpoint['loss'],
                            update_target_after_episode = checkpoint['update_after_episode'])
            
            agent.online_network.load_state_dict(checkpoint['policy_network'])
            agent.target_network.load_state_dict(checkpoint['target_network'])
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
            agent.log_episodes = checkpoint['log_episodes']
            agent.log_rewards = checkpoint['log_reward']
            agent.log_avg_rewards = checkpoint['log_avg_reward']

            print('checkpoint loaded at {}'.format(filename))
            return agent
        else:
            raise OSError("Checkpoint file not found.")    
