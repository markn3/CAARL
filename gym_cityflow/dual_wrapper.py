# Import required libraries
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F


# Define the DualEnvWrapper class to handle both agent and adversary
class DualEnvWrapper(gym.Wrapper):
    def __init__(self, env, action_space=None, tp=None):
        super(DualEnvWrapper, self).__init__(env)
        if action_space is not None:
            self.action_space = action_space
        if tp is not None:
            self.transition_probs = tp
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
            self.perturbed_state, exceeded_budget = self.apply_perturbation(action, self.env.get_state())

            # get the action from the agent model. Passing the sequence of past observations through the predict function.
            agent_action = self.agent.predict(self.past_observations.reshape(1,5,33))

            # using the agent's action, we step in the environment to get the needed info
            self.perturbed_state, agent_reward, done, info = self.env.step(agent_action[0][0])    

            # action probabilities
            action_probs = self.agent.policy.action_probs

            state_key = tuple(self.env.get_state().flatten())
            perturbed_state_key = tuple(self.perturbed_state.flatten())
            agent_action, _ = agent_action  # assuming agent_action was a tuple with the actual action as the first item
            agent_action = int(agent_action)
            default_value = 1e-10
            p = self.transition_probs.get((state_key, agent_action), {}).get(perturbed_state_key, default_value)

            # Compute adversary reward
            adversary_reward = self.compute_adversary_reward(self.env.get_state(), action, self.perturbed_state, 
            action_probs, p, agent_reward, exceeded_budget)

            # Update the past observations with the new perturbed state
            self.update_past_observations(self.perturbed_state)
            
            return self.past_observations.reshape(1,5,33), adversary_reward, done, {}
        else:
            # Step the real environment with the agent's action
            next_state, reward, done, info = self.env.step(action)
            self.update_past_observations(next_state)
            
            # Update the perturbed state to be the new real state (until the adversary perturbs it again)
            self.perturbed_state = next_state
            
            return self.past_observations.reshape(1,5,33), reward, done, info

    def apply_perturbation(self, adversary_action, original_observation, budget=2.0):
        """
        Modify the observation based on the adversary's action, enforcing an L_infinity norm constraint.
        
        Parameters:
        - adversary_action: The perturbation added by the adversary
        - original_observation: The original state before perturbation
        - budget: The maximum allowable L_infinity norm for the perturbation (epsilon in the description)
            
        Returns:
        - perturbed_observation: The perturbed state
        - exceeded_budget: Boolean flag indicating if the original action was outside the budget
        """

        if len(adversary_action.shape) > 1:
            adversary_action = adversary_action[0]

        # Flatten
        adversary_action_flat = adversary_action.flatten()
        
        # Calculate the L_infinity norm of the adversary_action
        linf_norm = np.max(np.abs(adversary_action[:-1]))

        # Check if the perturbation exceeds the budget
        exceeded_budget = False
        if linf_norm > budget:
            # Set the flag to True since original action was outside the budget
            exceeded_budget = True
            # Scale down the adversary_action to meet the budget constraint
            scaling_factor = budget / linf_norm
            adversary_action[:-1] = np.sign(adversary_action[:-1]) * np.minimum(np.abs(adversary_action[:-1]), budget)

        # Perturb all elements except the last one
        perturbed_observation = original_observation[:-1] + adversary_action[:-1]
        perturbed_observation -= 1

        # Ensure that all values are >= 0
        perturbed_observation = np.maximum(perturbed_observation, 0)

        # Append the last, unperturbed element from the original observation
        perturbed_observation = np.append(perturbed_observation, original_observation[-1])
        return perturbed_observation, exceeded_budget

    def compute_adversary_reward(self, action_probs, agent_reward, transition_probs, exceeded_budget):
        """
        Compute the adversary's expected reward based on the provided inputs.
        
        Parameters:
        - C: Constant reward for cases where the adversary's action is outside the perturbation budget.
        - action_probs: Values representing the agent's policy.
        - transition_probs: Transition probabilities.
        - agent_reward: Environment reward values.
        - exceeded_budget: Boolean flag indicating if the adversary's action was outside the budget.

        Returns:
        - Adversary's expected reward.
        """
        C = -1000
        
        # Check if the adversary's action is within the perturbation budget
        if not exceeded_budget:
            # Compute the numerator and denominator of the reward formula using agent's policy, empirical transition probability, and environment reward
            numerator = -torch.sum(action_probs * transition_probs * agent_reward)
            denominator = torch.sum(action_probs * transition_probs)
            
            # Return the computed reward if the denominator is non-zero; otherwise, return a default value
            if denominator != 0:
                reward_value = (numerator / denominator).item()
                return reward_value
            else:
                return 0  # Default value when denominator is zero
        # If the adversary's action exceeds the budget, return the constant reward C
        else:
            print("Exceeded")
            return C