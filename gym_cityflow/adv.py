import gym
import numpy as np
from gym import spaces
import numpy as np

class AdversaryEnv(gym.Wrapper):
    def __init__(self, env, adversary):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([3]*33)
        self.adversary = adversary
        
    def reset(self):
        true_obs = self.env.reset()
        perturbation, _ = self.adversary.predict(true_obs)
        initial_perturbed_obs = self.perturb_observation(true_obs, perturbation)
        return initial_perturbed_obs

    def step(self, action):
        action = action[0]
        obs, reward, done, info = self.env.step(action)
        self.adversary_action = self.adversary.predict(obs)[0]
        perturbed_obs = self.perturb_observation(obs, self.adversary_action)
        perturbed_obs = perturbed_obs.reshape(1, -1)

        return perturbed_obs, -reward, done, info

    def perturb_observation(self, true_obs, perturbation):
        # p and epsilon
        p = 0
        epsilon = 0

        # Compute the lp norm of the perturbation
        lp_norm = np.linalg.norm(perturbation[:-1], ord=p)

        # If the lp norm is larger than the budget epsilon, scale down the perturbation
        if lp_norm > epsilon:
            perturbation[:-1] = perturbation[:-1] / lp_norm * epsilon

        # Subtract 2 from the action to get values of -2, -1, 0, 1, 2
        scale = 1
        perturbed_obs = true_obs.copy()  # Create a copy of the true observation
        perturbed_obs[:-1] += perturbation[:-1] - scale  # Apply the perturbation to all elements except the last one

        # Ensure that all values are >= 0
        perturbed_obs = np.maximum(perturbed_obs, 0)

        return perturbed_obs

    def get_adversary_action(self):
        return self.adversary_action