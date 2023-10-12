from stable_baselines3 import PPO
import gym
import numpy as np

env  = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
model = PPO("MlpPolicy", env, verbose=1)

observations = []
actions = []
next_observations = []

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    next_obs, _, _, _ = env.step(action)
    observations.append(obs)
    actions.append(int(action))
    next_observations.append(next_obs)
    
    obs = next_obs

import torch
import torch.nn as nn
import torch.optim as optim

class StochasticTransitionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StochasticTransitionModel, self).__init__()
        # Define layers for mean and variance predictions
        self.mean_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.variance_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, state, action):
        action = action.long()  # Convert action tensor to long data type
        action = torch.eye(5)[action]  # One-hot encode the action 
        x = torch.cat([state, action], dim=1)
        mean = self.mean_layer(x)
        variance = torch.exp(self.variance_layer(x))  # Ensure variance is non-negative
        return mean, variance

# Convert data to PyTorch tensors
states_tensor = torch.tensor(observations, dtype=torch.float32)
# print("ACTIONS: ", actions)
actions_tensor = torch.tensor(actions, dtype=torch.float32)
next_states_tensor = torch.tensor(next_observations, dtype=torch.float32)

# Initialize model and optimizer
predictive_model = StochasticTransitionModel(input_dim=38, output_dim=33)
optimizer = optim.Adam(predictive_model.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    
    predicted_means, predicted_variances = predictive_model(states_tensor, actions_tensor)
    
    # Gaussian Negative Log Likelihood
    loss = 0.5 * (torch.log(predicted_variances) + (next_states_tensor - predicted_means)**2 / predicted_variances).sum()
    
    loss.backward()
    optimizer.step()

# Save the trained model
save_path = "transition_model.pth"
torch.save(predictive_model.state_dict(), save_path)
print(f"Model saved to {save_path}")

with torch.no_grad():
    mean, variance = predictive_model(state_tensor, action_tensor)
    next_state_sample = torch.normal(mean, torch.sqrt(variance))

print("DONE")