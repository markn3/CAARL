# Import necessary modules
import os
import gym
from stable_baselines3 import PPO

if __name__ == "__main__":

    # Initialize base environment and wrap it
    env  = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    env.reset()

    # Initialize the agent and the adversary
    agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/agent")

    # Set the total number of episodes and the number of episodes per training round
    total_episodes = 3000
    
    # Start the training loop over the total number of episodes
    agent.learn(total_timesteps=env.steps_per_episode*total_episodes, reset_num_timesteps=False, tb_log_name="agent_no_adv_3000")
    agent.save("./models/no_adv_3000")