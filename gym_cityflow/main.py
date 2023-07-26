# Import necessary modules
import os
import gym
from stable_baselines3 import PPO
from adv import AdversarialEnv  # Import the AdversarialEnv class from adv.py

if __name__ == "__main__":

    # Define directories for saving models and logs
    models_dir = "./models"
    logdir = os.path.join("logs")

    # Create directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Define the total number of episodes to train for
    total_episodes = 1000

    # Create the original Gym environment
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

    # Create the adversary using the same PPO algorithm as the main model
    adversary = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    # Wrap the original environment with the AdversarialEnv,
    # which will use the adversary to perturb the agent's observations
    env = AdversarialEnv(env, adversary)

    # Create the main model (the traffic signal controller) using the adversarial environment
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    # Define the number of steps to log results after
    log_interval = 5
    # Define the total number of steps to train for
    total_steps = total_episodes * env.steps_per_episode

    # Main training loop
    for i in range(total_episodes):
        # Reset the environment and get the initial observation
        obs = env.reset()

        done = False
        while not done:
            # The model makes a decision based on the perturbed observation
            action, _states = model.predict(obs)

            # Take a step in the environment and get the new perturbed observation and reward
            obs, reward, done, info, adversary_reward = env.step(action)

            # Both the model and the adversary learn from their experiences
            model.learn(total_timesteps=1)
            adversary.learn(total_timesteps=1)

    print("Done")
