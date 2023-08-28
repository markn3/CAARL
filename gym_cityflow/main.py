# Import necessary modules
import os
import gym
import copy
from stable_baselines3 import PPO
from custom_LSTM import CustomLSTMPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from adv import AdversaryEnv  # Import the AdversarialEnv class from adv.py
from agent import AgentEnv
import numpy as np



if __name__ == "__main__":

    # Define directories for saving models and logs
    models_dir = "./models"
    logdir = os.path.join("logs")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Create the environments for the agent and the adversary
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    adversary_env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

    print("1")
    # Create the adversary
    adversary = PPO("MlpPolicy", adversary_env, verbose=1, tensorboard_log=logdir)

    print("2")
    # Wrap the adversary's environment with the AdversaryEnv
    wrapped_adversary_env = AdversaryEnv(adversary_env, adversary)

    print("3")
    # Recreate the adversary with the wrapped environment
    adversary = PPO("MlpPolicy", wrapped_adversary_env, verbose=1, tensorboard_log=logdir)

    print("4")
    # Recreate the AdversaryEnv with the final adversary
    adversary_env = AdversaryEnv(adversary_env, adversary)

    print("5")
    # Wrap the agent's environment with the AgentEnv
    env = AgentEnv(env, adversary)  # Use a copy of the adversary

    # Create the main model (the traffic signal controller)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)


    total_episodes = 10

    for episode in range(total_episodes):
        obs = env.reset()
        obs = obs.reshape(1, -1)
        adversary_obs = adversary_env.reset()
        adversary_obs = adversary_obs.reshape(1, -1)
        done = False
        while not done:
            
            action, _states = model.predict(obs)
            adversary_action, _ = adversary.predict(adversary_obs)

            obs, reward, done, info = env.step(action)

            model.last_reward = reward  # Store the last reward for the adversary

            adversary_obs, _, _, _ = adversary_env.step(action)

        if episode % 2 == 0:
            print("AGENT LEARNING ", episode)
            model.learn(total_timesteps=env.steps_per_episode)
            adversary_env = AdversaryEnv(adversary_env, model)  # Use a copy of the model
        else:
            print("ADVERSARY LEARNING", episode)
            adversary.learn(total_timesteps=env.steps_per_episode)
            env = AgentEnv(env, adversary)  # Use a copy of the adversary

    print("Done")