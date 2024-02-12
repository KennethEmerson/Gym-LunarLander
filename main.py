"""
    Main File regarding the Lunar Lander exercise, contains startup lines for:
        * running a comparison between the training of several agents with diff. hyperparameters
        * running a training run for the proposed agent
        * running the LunarLander pretrained agent 
"""

import gym

from agent import Agent
from game import get_state_action_space, run_trial
import agent_test_lists
import plot


def train_agents(agent_list,episodes, episode_print=None, save_agents=False):
    run_trial(env,agent_list,episodes,episode_print,eval_only=False,logging=True,save_agents=save_agents)
    plot.plot_comparison(agent_list,save_plot=True)
            

def test_agent(agent,episodes, episode_print=None):
    run_trial(env,[agent],episodes,episode_print,eval_only=True)
    plot.plot_cum_rewards(test_agent)

if __name__ == '__main__':
    
    # Create the Gym environment.
    env = gym.make("LunarLander-v2")
    state_dim,action_dim = get_state_action_space(env)

    EPISODES = 5000 
    EPISODE_PRINT = 1
    
    # the list of agents tested as presented in the report
    AGENT_LIST = agent_test_lists.list1(state_dim,action_dim)
    
    # the final agent as proposed in the report
    FINAL_AGENT = [Agent(state_dim,action_dim,
                    epsilon_decay= 200,
                    repl_mem_size= 100000,
                    repl_batch_size = 64,
                    gamma = 0.99,
                    learning_rate = 0.0001,
                    loss = "SmoothL1",
                    update_target_after_episode=30)]

    ####################################################################
    # ENABLE THE FOLLOWING LINE TO TRAIN THE AGENTS FROM THE AGENT LIST
    ####################################################################
    
    #train_agents(AGENT_LIST,EPISODES, episode_print=EPISODE_PRINT, save_agents=False)
    
    ####################################################################
    # ENABLE THE FOLLOWING LINE TO TRAIN THE PROPOSED AGENT 
    ####################################################################
    
    #train_agents(FINAL_AGENT,EPISODES, episode_print=EPISODE_PRINT, save_agents=False)

    ####################################################################
    # ENABLE THE FOLLOWING LINES TO LOAD THE PROPOSED (PRETRAINED) AGENT
    # AND PERFORM AN EVALUATION RUN 
    ####################################################################
    
    test_agent = Agent.load("saved_agents/e200_mem100000_batch64_SmoothL1_g0.99_LR0.0001_TN_upd30_20210529 032209.pt")
    run_trial(env,[test_agent],1000,episode_print=1,eval_only=True,logging=True,save_agents=False)
    plot.plot_cum_rewards(test_agent)
    