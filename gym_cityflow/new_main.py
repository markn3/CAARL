# Import necessary modules
import os
import gym
from stable_baselines3 import PPO
from custom_LSTM import CustomLSTMPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from adv import AdversaryEnv  # Import the AdversarialEnv class from adv.py
from dual_wrapper import DualEnvWrapper
from gym import spaces


# Initialize base environment and wrap it
base_env  = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

# Create wrappers for the agent and the adversary
agent_env = DualEnvWrapper(base_env, action_space=base_env.action_space)  # or any other custom action space
adversary_env = DualEnvWrapper(base_env, action_space=spaces.MultiDiscrete([3]*33))

# Initialize the agent and the adversary
agent = PPO("MlpPolicy", agent_env, verbose=1)
adversary = PPO("MlpPolicy", adversary_env, verbose=1)

# Unified training loop
total_episodes = 100
episodes_per_round = 10  # Number of episodes to train each model per round

for episode in range(total_episodes):
    done = False
    
    # Decide who is training this episode: agent or adversary
    is_adversary_training = (episode // episodes_per_round) % 2 == 1

    if is_adversary_training:
        print("Training the adversary in episode:", episode)
    else:
        print("Training the agent in episode:", episode)
    
    # Reset the environments and get initial observations
    agent_obs = agent_env.reset()
    adversary_obs = adversary_env.reset()

    
    while not done:
        if is_adversary_training:
            # Train the adversary
            print("YES")
            action, _ = adversary.predict(adversary_obs)
            next_obs, reward, done, _ = adversary_env.step(action, is_adversary=True)
            
            # Update adversary observation
            adversary_obs = next_obs
            
            # Sync the perturbed state to the agent's environment
            agent_env.perturbed_state = adversary_env.perturbed_state
            
        else:
            # Train the agent
            action, _ = agent.predict(agent_obs)
            next_obs, reward, done, _ = agent_env.step(action, is_adversary=False)
            
            # Update agent observation
            agent_obs = next_obs
            
            # Sync the perturbed state to the adversary's environment
            adversary_env.perturbed_state = agent_env.perturbed_state
            
    # Train the models based on the collected experience
    if is_adversary_training:
        adversary.learn(total_timesteps=agent_env.steps_per_episode)
    else:
        agent.learn(total_timesteps=agent_env.steps_per_episode)