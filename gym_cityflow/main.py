import gym
import gym_cityflow
import numpy as np
from stable_baselines3 import PPO
from adv import AdversarialEnv

# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines.common.vec_enc import DummyVecEnv
# from stable_baselines import DQN
import os

models_dir = "models/PPO"
logdir = "logs"


if __name__ == "__main__":

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    total_episodes = 1000
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    adversary = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    env = AdversarialEnv(env, adversary)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    log_interval = 5
    total_steps = total_episodes * env.steps_per_episode

    for i in range(total_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info, adversary_reward = env.step(action)
            model.learn(total_timesteps=1)
            adversary.learn(total_timesteps=1)

    print("Done")