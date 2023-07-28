import gym
from gym import error, spaces, utils, logger
import numpy as np

class AgentEnv(gym.Wrapper):
    def __init__(self, env, adversary):
        super().__init__(env)
        self.adversary = adversary

    def step(self, action):
        action = action.item()
        # Get the true observation, reward, and done status from the environment
        true_obs, reward, done, info = self.env.step(action)

        # Let the adversary choose a perturbation
        perturbation, _ = self.adversary.predict(true_obs)

        # Apply the perturbation to the observation
        obs = self.perturb_observation(true_obs, perturbation)
        obs = obs.reshape(1, -1)

        return obs, reward, done, info

    def perturb_observation(self, true_obs, perturbation):
        # Subtract 2 from the action to get values of -2, -1, 0, 1, 2
        scale = 1
        perturbed_obs = true_obs.copy()  # Create a copy of the true observation
        perturbed_obs[:-1] += perturbation[:-1] - scale  # Apply the perturbation to all elements except the last one

        # Ensure that all values are >= 0
        perturbed_obs = np.maximum(perturbed_obs, 0)
        # print(f"perturbed_obs: {perturbed_obs}  | true_obs: {true_obs}")

        return perturbed_obs


