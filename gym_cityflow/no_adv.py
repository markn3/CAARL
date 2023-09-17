# Import necessary modules
import os
import gym
from stable_baselines3 import PPO
from custom_LSTM import CustomLSTMPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from dual_wrapper import DualEnvWrapper
from gym import spaces
from helper import load_parameters


if __name__ == "__main__":

    # Define directories for saving models and logs
    models_dir = "./models"
    logdir = os.path.join("logs")

    # Create directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logdir_agent = "./logs/agent"

    # Initialize base environment and wrap it
    env  = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    env.reset()

    # Initialize the agent and the adversary
    agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir_agent)

    # Set the total number of episodes and the number of episodes per training round
    total_episodes = 3500
    
    # Start the training loop over the total number of episodes
    for episode in range(total_episodes):
        print(f"Episode {episode}")
        obs = env.reset()
        for step in range(env.steps_per_episode):
            action, _ = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
        agent.learn(total_timesteps=env.steps_per_episode, reset_num_timesteps=False, tb_log_name="agent_no_adv_2")


 
                