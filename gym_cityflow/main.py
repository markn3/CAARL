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
    logdir_adversary = "./logs/adversary"

    # Initialize base environment and wrap it
    base_env  = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

    # Create wrappers for the agent and the adversary
    agent_env = DualEnvWrapper(base_env, action_space=base_env.action_space)
    adversary_env = DualEnvWrapper(base_env, action_space=spaces.MultiDiscrete([3]*33))

    # Load the parameters (work in progress)
    params = load_parameters('parameters.json')

    # Initialize the agent and the adversary
    agent = PPO(CustomLSTMPolicy, agent_env, verbose=1, tensorboard_log=logdir_agent)
    adversary = PPO(CustomLSTMPolicy, adversary_env, verbose=1,tensorboard_log=logdir_adversary)

    # Set the total number of episodes and the number of episodes per training round
    total_episodes = 1000
    episodes_per_round = 10

    # Start the training loop over the total number of episodes
    for episode in range(total_episodes):

        # Reset the LSTM's internal states for both agent and adversary
        agent.policy.reset_states()
        adversary.policy.reset_states()

        done = False
        
        # Determine the training entity (agent or adversary) based on the current episode
        is_adversary_training = (episode // episodes_per_round) % 2 == 1

        # Set the training mode based on the determined training entity and display a message
        if is_adversary_training:
            print("Training the adversary in episode:", episode)
            adversary_env.set_mode(True, agent)
        else:
            print("Training the agent in episode:", episode)
            agent_env.set_mode(False)
        
        # Reset the environments and get initial observations
        agent_obs = agent_env.reset().reshape(1, 5, 33)
        adversary_obs = adversary_env.reset().reshape(1, 5, 33)

        while not done:

            # If the adversary is training, predict its action and take a step in its environment
            if is_adversary_training:
                # Train the adversary
                action, _ = adversary.predict(adversary_obs)
                next_obs, reward, done, _ = adversary_env.step(action)
                
                # Update adversary observation
                adversary_obs = next_obs
                
                # Sync the perturbed state to the agent's environment
                agent_env.perturbed_state = adversary_env.perturbed_state
            # If the agent is training, predict its action and take a step in its environment
            else:
                # Train the agent
                action, _ = agent.predict(agent_obs)
                next_obs, reward, done, _ = agent_env.step(action[0])
                # Update agent observation
                agent_obs = next_obs
                
                # Sync the perturbed state to the adversary's environment
                adversary_env.perturbed_state = agent_env.perturbed_state
                
        # Train the active model (agent or adversary) based on the experience collected during this episode
        if is_adversary_training:
            adversary_env.set_mode(True, agent)
            adversary.learn(total_timesteps=agent_env.steps_per_episode, reset_num_timesteps=False, tb_log_name="adv_1")
        else:
            agent_env.set_mode(False)
            agent.learn(total_timesteps=agent_env.steps_per_episode, reset_num_timesteps=False, tb_log_name="agent_1")