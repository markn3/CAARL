import os
import gym
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from dual_wrapper import DualEnvWrapper
from custom_LSTM import CustomLSTMPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from empirical_estimation import EmpiricalTransitionEstimator, train_estimator


def evaluate_model(agent, env, perturbation_level, num_episodes):
    total_waiting_time = 0
    total_vehicles = 0
    
    # Set environment perturbation level (modify as per your environment API)
    env.set_perturbation_level(perturbation_level)

    agent_env.set_mode(False)
    
    for episode in range(num_episodes):
        obs = env.reset().reshape(1, 5, 33)  # Resetting environment and reshaping observation
        done = False

        # Reset the LSTM's internal states for both agent and adversary
        agent.policy.reset_states()
        adversary.policy.reset_states()
        
        while not done:
            action, _ = agent.predict(obs)  # Predicting action
            next_obs, reward, done, info = env.step(action[0])  # Taking a step

            # Directly retrieve average travel time
            print("info: ", info)
            
            # # Assume `info` provides waiting time and num of vehicles (modify as per your environment API)
            # waiting_time = info['waiting_time']
            # num_vehicles = info['num_vehicles']
            
            # # Update total waiting time and vehicles
            # total_waiting_time += waiting_time
            # total_vehicles += num_vehicles
            
            # Update observation
            obs = next_obs

            # Update agent observation
            agent_obs = next_obs
            
            # Sync the perturbed state to the adversary's environment
            adversary_env.perturbed_state = agent_env.perturbed_state
    
    # Calculate and return average waiting time per vehicle
    average_waiting_time_per_vehicle = total_waiting_time / total_vehicles
    return average_waiting_time_per_vehicle

if __name__ == "__main__":

    print("Training estimator")
    estimator = EmpiricalTransitionEstimator()
    transition_probs = train_estimator(estimator)
    print("done")

    # Initialize base environment and wrap it
    base_env  = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

    # Create wrappers for the agent and the adversary
    agent_env = DualEnvWrapper(base_env, action_space=base_env.action_space)
    adversary_env = DualEnvWrapper(base_env, action_space=spaces.MultiDiscrete([12]*33), tp=transition_probs)


    # Paths to trained models
    AGENT_CHECKPOINT_PATH = "./models/test_models/checkpoint_agent_2000" # Paths to the saved checkpoints of the agent to load
    ADVERSARY_CHECKPOINT_PATH = "./models/test_models/checkpoint_adv_2000" # Paths to the saved checkpoints of the adv to load

    # Load the trained model
    agent = PPO.load(AGENT_CHECKPOINT_PATH, env=agent_env)
    adversary = PPO.load(ADVERSARY_CHECKPOINT_PATH, env=adversary_env)
    
    # Define perturbation levels for testing
    perturbation_levels = ['low', 'medium', 'high']
    
    # Storage for average waiting times per perturbation level
    waiting_times = []
    
    # Main testing loop
    for level in perturbation_levels:
        # Adjust environment for the current perturbation level
        # ...
        # Evaluate the model
        avg_waiting_time = evaluate_model(model, env, num_episodes=100)
        # Store the average waiting time
        waiting_times.append(avg_waiting_time)
    
    # Create a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(perturbation_levels, waiting_times, color=['blue', 'green', 'red'])
    plt.xlabel('Perturbation Level')
    plt.ylabel('Average Waiting Time per Vehicle')
    plt.title('Sensitivity Analysis on Perturbation Budget')
    plt.show()
