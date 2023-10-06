# Import necessary modules
import os
import gym
from stable_baselines3 import PPO
from custom_LSTM import CustomLSTMPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from dual_wrapper import DualEnvWrapper
from gym import spaces
from helper import load_parameters
from empirical_estimation import EmpiricalTransitionEstimator, train_estimator


if __name__ == "__main__":

    # Define directories for saving models, logs, and checkpoints
    models_dir = "./models"
    logdir = os.path.join("logs")
    logdir_agent = "./logs/agent"
    logdir_adversary = "./logs/adversary"
    log_agent = "agent_f"
    log_adv = "adv_f"
    params = load_parameters('parameters.json') # Load the parameters (work in progress)
    AGENT_CHECKPOINT_PATH = "./models/agent/checkpoint_agent_160" # Paths to the saved checkpoints of the agent to load
    ADVERSARY_CHECKPOINT_PATH = "./models/adv/checkpoint_adv_160" # Paths to the saved checkpoints of the adv to load
    agent_checkpoint = "./models/agent/checkpoint_agent_f" # Save checkpoint as 
    adv_checkpoint = "./models/adv/checkpoint_adv_f" # Save checkpoint as 
    current_episode = "./models/current_episode.txt" # Text file that keeps track of the latest episodes (for checkpointing)

    start_episode = 0
    total_episodes = 3000 # total number of episodes
    episodes_per_round = 10 # How many episodes to train agent/adversary before training the other
    LOAD_FROM_CHECKPOINT = False  # Flag to determine if loading from checkpoint or creating new models

    # Create directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("Training estimator")
    estimator = EmpiricalTransitionEstimator()
    transition_probs = train_estimator(estimator)
    print("done")

    # Initialize base environment and wrap it
    base_env  = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

    # Create wrappers for the agent and the adversary
    agent_env = DualEnvWrapper(base_env, action_space=base_env.action_space)
    adversary_env = DualEnvWrapper(base_env, action_space=spaces.MultiDiscrete([12]*33), tp=transition_probs)
    
    # Load or create agent and adversary
    if LOAD_FROM_CHECKPOINT:
        print("Loading checkpoints")
        agent = PPO.load(AGENT_CHECKPOINT_PATH, env=agent_env)
        adversary = PPO.load(ADVERSARY_CHECKPOINT_PATH, env=adversary_env)
        with open(current_episode, "r") as file:
            start_episode = int(file.read())
    else:
        agent = PPO(CustomLSTMPolicy, agent_env, verbose=1, tensorboard_log=logdir_agent)
        adversary = PPO(CustomLSTMPolicy, adversary_env, verbose=1, tensorboard_log=logdir_adversary)

    # Start the training loop over the total number of episodes
    for episode in range(start_episode, total_episodes):

        # Save checkpoints every 60 episodes
        if episode % 40 == 0 and episode > 0:
            agent.save(agent_checkpoint + str(episode))
            adversary.save(adv_checkpoint + str(episode))
            with open(current_episode, "w") as file:
                file.write(str(episode))

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
            agent_env.set_mode(False, agent, adversary)
        
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
                next_obs, reward, done, _ = agent_env.step()
                
                # Update observation
                adversary_obs = next_obs
                
                # Sync the perturbed state to the adversary's environment
                adversary_env.perturbed_state = agent_env.perturbed_state
                
        # Train the active model (agent or adversary) based on the experience collected during this episode
        if is_adversary_training:
            adversary.policy.reset_states()
            adversary_env.set_mode(True, agent)
            adversary.learn(total_timesteps=agent_env.steps_per_episode,reset_num_timesteps=False, tb_log_name=log_adv) # make sure to change the name
        else:            
            agent.policy.reset_states()
            agent_env.set_mode(False, agent, adversary)
            agent.learn(total_timesteps=agent_env.steps_per_episode,reset_num_timesteps=False, tb_log_name=log_agent) # make sure to change the name