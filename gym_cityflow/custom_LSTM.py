from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
from collections import deque
import torch

N = 5

class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomLSTMPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=33, hidden_size=64)
        self.features_dim = 64

        # Define a buffer to store the last N observations
        self.observation_buffer = deque(maxlen=N)  # N is the desired sequence length

    def forward(self, obs, deterministic=False):
        print("Observations: ", obs)
        # Store the current observation in the buffer
        self.observation_buffer.append(obs)

        # Convert the buffer to a PyTorch tensor and adjust dimensions
        lstm_input = torch.stack(list(self.observation_buffer)).unsqueeze(1)
        
        # Pass the batched observations through the LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        # Use the last output from the LSTM sequence for the rest of the network
        lstm_out = lstm_out[-1]

        return super(CustomLSTMPolicy, self).forward(lstm_out, deterministic)

