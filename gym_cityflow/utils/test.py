import os
import gym
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from dual_wrapper import DualEnvWrapper
from custom_LSTM import CustomLSTMPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from empirical_estimation import EmpiricalTransitionEstimator, train_estimator


def evaluate_model(agent, adv, env, num_episodes):
    agent_env.set_mode(False, agent, adv)
    total_travel_time = 0
    avg_travel_time = 0
    
    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        agent_obs = env.reset().reshape(1, 5, 33)  # Resetting environment and reshaping observation
        done = False

        # Reset the LSTM's internal states for both agent and adversary
        agent.policy.reset_states()
        adv.policy.reset_states()
        
        while not done:
            action, _ = agent.predict(agent_obs)  # Predicting action
            agent_obs, reward, done, info = env.step(action)  # Taking a step

            # Accumulate total travel time and increment total steps
            total_travel_time += info['average_travel_time']
        avg_travel_time += total_travel_time/((episode+1)*1000)
        print("info: ", total_travel_time/((episode+1)*1000))
    
    # Calculate and return average travel time per step
    real = avg_travel_time / (num_episodes)  # Normalized by steps per episode
    print("travel_time: ", real)

    return real

def bar_graph():
    traffic_levels = ['Low', 'Medium', 'High']
    avg_travel_times = [54.67, 63.25, 110.16]

    # Use a predefined style
    plt.style.use('seaborn-darkgrid')

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Adjusted colors
    bars = plt.bar(traffic_levels, avg_travel_times, color=colors, width=0.6, edgecolor='grey')
    
    # Add labels and title with adjusted sizes
    plt.xlabel('Traffic Level', fontsize=14)
    plt.ylabel('Average Travel Time per Vehicle (seconds)', fontsize=14)
    plt.title('Average Travel Time per Vehicle for Different Traffic Levels', fontsize=16)

    plt.xticks(traffic_levels)
    plt.tight_layout()  # Adjusts subplot params for better layout

    # Add value labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, round(yval, 2), va='center', ha='center', fontsize=12)


    # Add light horizontal gridlines for easier readability
    plt.grid(axis='y', linestyle='-', alpha=0.9, color='gray')

    # Save the figure
    plt.savefig('./graphs/travel_time_comparison2.png', pad_inches=0.1)

if __name__ == "__main__":

    print("Training estimator")
    estimator = EmpiricalTransitionEstimator()
    transition_probs = train_estimator(estimator)
    print("done")

    # Initialize base environment and wrap it
    base_env  = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

    # Create wrappers for the agent and the adversary
    agent_env = DualEnvWrapper(base_env, action_space=base_env.action_space)
    adversary_env = DualEnvWrapper(base_env, action_space=spaces.MultiDiscrete([6]*33), tp=transition_probs)

    
    # Paths to trained models
    AGENT_CHECKPOINT_PATH = "./models/test_models/checkpoint_agent_newjeans1200" # Paths to the saved checkpoints of the agent to load
    ADVERSARY_CHECKPOINT_PATH = "./models/test_models/checkpoint_adv_newjeans1200" # 6

    # # # Paths to trained models
    # AGENT_CHECKPOINT_PATH = "./models/test_models/checkpoint_agent_2000" # Paths to the saved checkpoints of the agent to load
    # ADVERSARY_CHECKPOINT_PATH = "./models/test_models/checkpoint_adv_2000" # 3

    # AGENT_CHECKPOINT_PATH = "./models/test_models/checkpoint_agent_gg_Pre_trained" # Paths to the saved checkpoints of the agent to load


    # # Paths to trained models
    # AGENT_CHECKPOINT_PATH = "./models/test_models/checkpoint_agent_better1440" # Paths to the saved checkpoints of the agent to load
    # ADVERSARY_CHECKPOINT_PATH = "./models/test_models/checkpoint_adv_better1440" # 24

    # Load the trained model
    agent = PPO.load(AGENT_CHECKPOINT_PATH, env=agent_env)
    adv = PPO.load(ADVERSARY_CHECKPOINT_PATH, env=adversary_env)
    
    # Define perturbation levels for testing
    # traffic_level = 'medium'
    
    # Storage for average waiting times per perturbation level

    # Single traffic level evaluation
    avg_travel_time = evaluate_model(agent, adv, agent_env, num_episodes=10)
    
    # # Create a bar graph
    # plt.figure(figsize=(10, 6))
    # plt.bar('medium', avg_travel_time, color='green')
    # plt.xlabel('Traffic Level')
    # plt.ylabel('Average Travel Time per Vehicle')
    # plt.title('Average Travel Time per Vehicle for Medium Traffic Level')
    # plt.xticks(['medium'])  # Explicitly setting x-ticks since we have just one bar
    # print("Saving")
    # plt.savefig('./graphs/avg_travel_time_2000_no_pert.png')
    # plt.show()

    # Save the figure


    # Agent 2000
    # no pert medium level  = 63.6
    # no pert light level   = 54.18
    # no pert heavy level   = 102.46

    # Example values for the three bars
    bar_graph()

