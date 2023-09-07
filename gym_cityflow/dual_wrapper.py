# Import required libraries
import gym
from gym import spaces
import numpy as np

# Define the DualEnvWrapper class to handle both agent and adversary
class DualEnvWrapper(gym.Wrapper):

    def __init__(self, env, action_space=None):
        super(DualEnvWrapper, self).__init__(env)
        if action_space is not None:
            self.action_space = action_space
        self.perturbed_state = None  # To store the state after adversary's perturbation
        self.is_adversary = False
        self.agent = None
        self.past_observations = np.zeros((5, 33))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5, 33), dtype=np.float32)

    def update_past_observations(self, new_obs):
        self.past_observations = np.roll(self.past_observations, shift=-1, axis=0)
        self.past_observations[-1] = new_obs

    # Set the mode to either agent or adversary
    def set_mode(self, is_adversary, agent=None):
        self.is_adversary = is_adversary
        self.agent = agent

    # Reset the environment and initialize the perturbed state
    def reset(self):
        obs = self.env.reset()
        self.perturbed_state = obs  # Initialize with the real state
        # Update observation space for sequences
        self.update_past_observations(obs)
        return self.past_observations  # Both agent and adversary would initially see the same observation

    def step(self, action):
        if self.is_adversary:
            # Apply the perturbation to the real state to get the perturbed state
            self.perturbed_state = self.apply_perturbation(action, self.env.get_state())

            # get the action from the agent model. Passing the sequence of past observations through the predict function.
            agent_action = self.agent.predict(self.past_observations.reshape(1,5,33))
            print("AGENT ACITON: ", agent_action)
            print("AGENT ACITON: ", agent_action[0][0])
            # using the agent's action, we step in the environment to get the needed info
            self.perturbed_state, reward, done, info = self.env.step(agent_action[0][0])
        
            print(self.env.current_step)
            if self.env.current_step+1 == self.env.steps_per_episode:
                done = True
            else:
                done = False

            # Update the past observations with the new perturbed state
            self.update_past_observations(self.perturbed_state)
            return self.past_observations.reshape(1,5,33), -reward, done, {}
        else:
            # Step the real environment with the agent's action
            next_state, reward, done, info = self.env.step(action)
            self.update_past_observations(next_state)
            
            # Update the perturbed state to be the new real state (until the adversary perturbs it again)
            self.perturbed_state = next_state
            
            return self.past_observations.reshape(1,5,33), reward, done, info

    def apply_perturbation(self, adversary_action, original_observation, norm_p=2, budget=1.0):
        """
        Modify the observation based on the adversary's action, enforcing an L_p norm constraint.
        
        Parameters:
        - adversary_action: The perturbation added by the adversary
        - original_observation: The original state before perturbation
        - norm_p: The p value for the L_p norm
        - budget: The maximum allowable L_p norm for the perturbation
        
        Returns:
        - perturbed_observation: The perturbed state
        """

        if len(adversary_action.shape) > 1:
            adversary_action = adversary_action[0]

        # Flatten
        adversary_action_flat = adversary_action.flatten()
        # Calculate the Lp norm of the adversary_action
        lp_norm = np.linalg.norm(adversary_action[:-1], ord=norm_p)
        
        # Check if the perturbation exceeds the budget
        if lp_norm > budget:
            # Scale down the adversary_action to meet the budget constraint
            scaling_factor = budget / lp_norm
            adversary_action[:-1] = np.round(adversary_action[:-1] * scaling_factor).astype(np.int64)

        # Perturb all elements except the last one
        perturbed_observation = original_observation[:-1] + adversary_action[:-1]
        perturbed_observation -= 1

        # Ensure that all values are >= 0
        perturbed_observation = np.maximum(perturbed_observation, 0)

        # Append the last, unperturbed element from the original observation
        perturbed_observation = np.append(perturbed_observation, original_observation[-1])
        return perturbed_observation