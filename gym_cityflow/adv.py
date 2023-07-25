import gym

class AdversarialEnv(gym.Wrapper):
    def __init__(self, env, adversary):
        super().__init__(env)
        self.adversary = adversary

        self.mode = "all_all"

        if self.mode == "all_all":
            self.observation_space = spaces.MultiDiscrete([100]*32 + [5])
        else:
            self.observation_space = spaces.MultiDiscrete([100]*8 + [5])

    def step(self, action):
        # Get the true observation, reward, and done status from the environment
        true_obs, reward, done, info = self.env.step(action)

        # Let the adversary choose a perturbation
        perturbation, _ = self.adversary.predict(true_obs)

        # Apply the perturbation to the observation
        obs = self.perturb_observation(true_obs, perturbation)

        # The adversary's reward is the negative of the primary agent's reward
        adversary_reward = -reward

        return obs, reward, done, info, adversary_reward

    def perturb_observation(self, true_obs, perturbation):
        # Modify this method as needed to apply the perturbation
        # For example, you might simply add the perturbation to the true observation
        return true_obs + perturbation

    def reset(self):
        # Reset the underlying Gym environment
        true_obs = self.env.reset()

        # Let the adversary choose an initial perturbation
        perturbation, _ = self.adversary.predict(true_obs)

        # Apply the perturbation to the initial observation
        obs = self.perturb_observation(true_obs, perturbation)

        return obs
