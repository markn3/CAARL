import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomLSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomLSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        
        input_dim = 33  # Based on the observation space
        hidden_dim = 128  # tune this
        num_layers = 1  # Number of LSTM layers, can also be tuned

        # Define LSTM and other layers here
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, features_dim)
        self.hidden_state = None
        self.cell_state = None
        
    
    def forward(self, x):
        # Initialize with stored states or zeros if they don't exist
        h_0 = self.hidden_state if self.hidden_state is not None else torch.zeros(self.lstm.num_layers, x.size(1), self.lstm.hidden_size).to(x.device)
        c_0 = self.cell_state if self.cell_state is not None else torch.zeros(self.lstm.num_layers, x.size(1), self.lstm.hidden_size).to(x.device)
        
        # Passing through LSTM
        x, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # Store the updated states
        self.hidden_state = h_n.detach()
        self.cell_state = c_n.detach()
        
        # We are interested in the last output for the actor-critic network
        x = x[:, -1, :]
        return self.fc(x)

        h_0 = torch.zeros(1, x.size(0), 128)  # initial hidden state
        c_0 = torch.zeros(1, x.size(0), 128)  # initial cell state
        x, _ = self.lstm(x, (h_0, c_0))
        return self.fc(x)

class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh, 
                 ortho_init=True, use_sde=False, log_std_init=0, full_std=True, use_expln=False, 
                 features_extractor_class=None, features_extractor_kwargs=None, share_features_extractor=True,
                 normalize_images=True, optimizer_class=torch.optim.Adam, optimizer_kwargs=None):
        
        # Handle the custom features_extractor_class and features_dim here
        if features_extractor_class is None:
            features_extractor_class = CustomLSTMFeatureExtractor
            features_extractor_kwargs = {"features_dim": 256}  # or adjust as needed
        
        super(CustomLSTMPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch=net_arch, 
                                               activation_fn=activation_fn, ortho_init=ortho_init, use_sde=use_sde, 
                                               log_std_init=log_std_init, full_std=full_std, use_expln=use_expln, 
                                               features_extractor_class=features_extractor_class, 
                                               features_extractor_kwargs=features_extractor_kwargs,
                                               share_features_extractor=share_features_extractor,
                                               normalize_images=normalize_images, 
                                               optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs)
    def reset_states(self):
        self.hidden_state = None
        self.cell_state = None
