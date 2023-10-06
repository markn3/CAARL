from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

class EmpiricalTransitionEstimator:
    def __init__(self):
        # Dictionary to hold transition counts
        # Structure: {(state, action): {next_state: count}}
        self.transition_counts = {}

    def add_transition(self, state, action, next_state):
        # Convert the state to a tuple (to be used as a key in the dictionary)
        key = (tuple(state), action)
        
        # If the (state, action) pair is not in the dictionary, add it
        if key not in self.transition_counts:
            self.transition_counts[key] = {}

        # If the next state is not in the dictionary for this (state, action) pair, add it
        if tuple(next_state) not in self.transition_counts[key]:
            self.transition_counts[key][tuple(next_state)] = 0
        
        # Increment the count for this (state, action, next_state) transition
        self.transition_counts[key][tuple(next_state)] += 1

    def get_transition_probabilities(self):
        # Dictionary to hold transition probabilities
        transition_probs = {}

        # For each (state, action) pair in the counts dictionary
        for (s, a), next_states in self.transition_counts.items():
            # Calculate the total number of transitions for this (state, action) pair
            total = sum(next_states.values())

            # For each next state, calculate the probability of transitioning to it
            # and store it in the transition_probs dictionary
            transition_probs[(s, a)] = {s_prime: count/total for s_prime, count in next_states.items()}
        return transition_probs

# Function to train the estimator
def train_estimator(estimator):
    # Create the environment
    env = DummyVecEnv([lambda: gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')])

    # Instantiate the model
    model = PPO("MlpPolicy", env, verbose=1)

    # Collect samples to estimate the transition probabilities
    num_steps = 10000
    obs = env.reset()
    for _ in range(num_steps):
        action, _ = model.predict(obs)
        next_obs, _, _, _ = env.step(action)
        
        # Store transition for empirical estimation
        for o, a, n_o in zip(obs, action, next_obs):
            estimator.add_transition(o, a, n_o)

        obs = next_obs
    return estimator.get_transition_probabilities()

# # Get the empirical transition probabilities
# transition_probs = estimator.get_transition_probabilities()
