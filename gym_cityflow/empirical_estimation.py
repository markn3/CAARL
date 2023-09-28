from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

class EmpiricalTransitionEstimator:
    def __init__(self):
        self.transition_counts = {}  # {(state, action): {next_state: count}}

    def add_transition(self, state, action, next_state):
        key = (tuple(state), action)
        if key not in self.transition_counts:
            self.transition_counts[key] = {}
        if tuple(next_state) not in self.transition_counts[key]:
            self.transition_counts[key][tuple(next_state)] = 0
        self.transition_counts[key][tuple(next_state)] += 1

    def get_transition_probabilities(self):
        transition_probs = {}
        for (s, a), next_states in self.transition_counts.items():
            total = sum(next_states.values())
            transition_probs[(s, a)] = {s_prime: count/total for s_prime, count in next_states.items()}
        return transition_probs

def train_estimator(estimator):
    # Create the environment
    env = DummyVecEnv([lambda: gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')])

    # Instantiate the model
    model = PPO("MlpPolicy", env, verbose=1)

    # Collect samples
    num_steps = 10000
    obs = env.reset()
    for _ in range(num_steps):
        action, _ = model.predict(obs)
        next_obs, _, _, _ = env.step(action)
        
        # Store transition for empirical estimation
        for o, a, n_o in zip(obs, action, next_obs):
            estimator.add_transition(o, a, n_o)

        obs = next_obs
    print("done")
    return estimator.get_transition_probabilities()

# # Get the empirical transition probabilities
# transition_probs = estimator.get_transition_probabilities()
