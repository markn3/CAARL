import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomLSTMPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # Assuming observation_space is flat: 32 elements up to 100 + 5 elements up to some small number
        super(CustomLSTMPolicy, self).__init__(observation_space, features_dim)
        
        input_dim = 33  # Based on the observation space
        hidden_dim = 128  # tune this
        num_layers = 1  # Number of LSTM layers, can also be tuned

        # Define LSTM and other layers here
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, features_dim)
        
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 128)  # initial hidden state
        c_0 = torch.zeros(1, x.size(0), 128)  # initial cell state
        x, _ = self.lstm(x, (h_0, c_0))
        return self.fc(x)