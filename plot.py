"""
 contains aux functions for plotting the results   
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

markers = [".","o","v","^","<",">","s","p","*"]
colors = ["black","red","blue","green"]

def plot_cum_rewards(agent):
    """plots the rewards and average reward of one agent

    Args:
        agent (Agent): the trained agent for which the log needs to be plotted
    """
    
    fig, ax = plt.subplots()

    episodes_x = agent.log_episodes

    plt.xlabel('episodes')
    plt.ylabel('reward')

    ax.plot(agent.log_episodes, agent.log_rewards,label="reward",linewidth=0.6,alpha=0.6) 
    ax.plot(agent.log_episodes, agent.log_avg_rewards,label="average reward",linewidth=0.6,alpha=1,c="red") 
    ax.legend()
    plt.title(agent,fontsize=10)
    plt.show()


def plot_comparison(agent_list,save_plot=True):
    """plots the average rewards of all agents in the list

    Args:
        agent_list ([Agent]): List of agents for which to plot the average reward logging
        save_plot (bool, optional): if true the plot is saved. Defaults to True.
    """

    fig, ax = plt.subplots(figsize=(15,8))

    plt.xlabel('episodes')
    plt.ylabel('avg rewards')

    for index, agent in enumerate(agent_list):
        marker_space = len(agent.log_episodes)
        ax.plot(agent.log_episodes, agent.log_avg_rewards,label=agent.__str__(),linewidth=0.7,alpha=0.8,
                        color=colors[index]) 
    ax.legend(fontsize=8)
    
    # save plot if requested
    if save_plot:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H_%M_%S")  
        plt.savefig(f".//plots//{dt_string}.png",format="png",dpi=300)
    
    plt.show()
