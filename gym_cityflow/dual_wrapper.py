import gym
from gym import spaces
import numpy as np

class DualEnvWrapper(gym.Wrapper):
    def __init__(self, env, action_space=None):
        super(DualEnvWrapper, self).__init__(env)
        if action_space is not None:
            self.action_space = action_space
        self.perturbed_state = None  # To store the state after adversary's perturbation
        self.is_adversary = False
        self.agent = None

    def set_mode(self, is_adversary, agent=None):
        self.is_adversary = is_adversary
        self.agent = agent

    def reset(self):
        obs = self.env.reset()
        self.perturbed_state = obs  # Initialize with the real state
        return obs  # Both agent and adversary would initially see the same observation

    def step(self, action):
        if self.is_adversary:
            # Apply the perturbation to the real state to get the perturbed state
            self.perturbed_state = self.apply_perturbation(action, self.env.get_state())

            # print("Checking state: ", self.env.get_state())
            # print("Checking perturbated state: ", self.perturbed_state)

            # get the action from the agent model. Passing the perturbed state through the predict function.
            agent_action = self.agent.predict(self.perturbed_state)

            # using the agent's action, we step in the environment to get the needed info
            self.perturbed_state, reward, done, info = self.env.step(agent_action[0])
        
            print(self.env.current_step)
            if self.env.current_step+1 == self.env.steps_per_episode:
                done = True
            else:
                done = False

            # Return the negative reward
            return self.perturbed_state, -reward, done, {}
        else:
            # Step the real environment with the agent's action
            next_state, reward, done, info = self.env.step(action)
            
            # Update the perturbed state to be the new real state (until the adversary perturbs it again)
            self.perturbed_state = next_state
            
            return next_state, reward, done, info

    def apply_perturbation(self, adversary_action, original_observation):
        """
        Modify the observation based on the adversary's action.
        The perturbation adds the adversary's actions to the observations 
        and then scales it down by 1.
        """
        # Perturb all elements except the last one
        perturbed_observation = original_observation[:-1] + adversary_action[:-1]
        perturbed_observation -= 1

        # Ensure that all values are >= 0
        perturbed_observation = np.maximum(perturbed_observation, 0)

        # Append the last, unperturbed element from the original observation
        perturbed_observation = np.append(perturbed_observation, original_observation[-1])
        return perturbed_observation


