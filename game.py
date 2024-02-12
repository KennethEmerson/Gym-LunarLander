"""
    File contains the functions to run the game, facilitate the agent-env interaction and logging
    of the results.
"""
import random
import numpy as np
import gym
import sys
import torch

from plot import plot_cum_rewards, plot_comparison

def get_state_action_space(env):
    """gets the actual state and action space from the AIGym environment

    Args:
        env (AIGym.Env): the loaded AIGym environment to be played
    """

    def multiply(tuple) : 
        result = 1 
        for i in tuple: 
            result *= i 
        return result
    
    action_space = env.action_space
    state_space = env.observation_space
    
    if isinstance(action_space,gym.spaces.discrete.Discrete):
        action_dim = action_space.n
    elif isinstance(action_space,gym.spaces.box.Box):
        action_dim = multiply(action_space.shape)
    
    if isinstance(state_space,gym.spaces.discrete.Discrete):
        state_dim = state_space.n
    elif isinstance(state_space,gym.spaces.box.Box):
        state_dim = multiply(state_space.shape)
    return state_dim,action_dim


def run_trial(env,agent_list,episodes,episode_print=None,eval_only=False,logging=True,save_agents=False):
    """runs the training trial on a list of one or more agents and facilitates the agent env interaction,
       the logging of the values and saves the agents if requested 

    Args:
        env (AIGym.Env): the AIGym environment in which to run the training trial
        agent_list ([Agent]): list of agent objects, the trial run will be run on each agent
        episodes (int): number of episodes the trial will run for each agent
        episode_print (int, optional): interval of episodes to print results. Defaults to None.
        eval_only (bool, optional): if true the agent will act greedy and not learn. Defaults to False.
        logging (bool, optional): if true the results will be logged in the agnet object. Defaults to True.
        save_agents (bool, optional): if true the agent will be saved after the trial. Defaults to False.
    """

    # perform the trial for each agent in the agent list
    for index, agent in enumerate(agent_list):
    
        # set seeds
        torch.manual_seed(0)
        random.seed(0)	
        np.random.seed(0)
        env.seed(0)
        
        # if logging is requested: initialize logs
        if logging:
            agent.init_logging(episodes)
            average_cumulative_reward = 0.0
        
        # print the results of the episode
        if episode_print is not None:
            print("\n" + "-" *90)
            print(agent)
            print("-" * 90 + "\n")
            if not eval_only:
                print("\n epi.   reward  avg rew.            loss    timesteps     mem   e-decay\n")
            else:
                print("\n epi.   reward  avg rew.    timesteps\n")
        
        # Loop over episodes
        for episode in range(episodes):
            state = env.reset()
            terminate = False
            cumulative_reward = 0.0
            timestep = 0
            
            # Loop over time-steps
            while not terminate:
                timestep +=1
                # Choose an action
                a = agent.select_action(state,episode,greedy=eval_only)  

                # Perform the action
                next_state, r, terminate, info = env.step(a)
                
                # if not eval_only: update the agent networks
                if not eval_only:
                    loss = agent.update_networks(state,terminate,a,r,next_state,episode)

                # Update statistics
                cumulative_reward += r
                state = next_state

            # Per-episode statistics
            average_cumulative_reward *= 0.95
            average_cumulative_reward += 0.05 * cumulative_reward

            # log the results in the agent
            if logging:
                agent.log(episode,cumulative_reward,average_cumulative_reward)
            
            #print the results per episode if requested
            if episode_print is not None and episode % episode_print==0:
                # whe the agent is in evaluation mode the printed results are different
                if not eval_only:
                    print(f"{episode:5} {round(cumulative_reward,2):8}\t" + 
                        f"{round(average_cumulative_reward,2):8}\t{round(loss,3):8}\t" + 
                        f"{timestep:5}\t{agent.repl_mem.samples_in_memory:5}\t{round(agent.epsilon(episode),5):6}")
                else:
                    print(f"{episode:5} {round(cumulative_reward,2):8}\t" + 
                        f"{round(average_cumulative_reward,2):8}\t" + 
                        f"{timestep:5}")
        
        # if requested, save the agent
        if save_agents:
            agent.save()

    
