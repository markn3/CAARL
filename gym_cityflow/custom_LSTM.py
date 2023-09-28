import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Tuple
from stable_baselines3.common.distributions import MultiCategoricalDistribution


class CustomLSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomLSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # # Input embedding layer projecting state dimension to 64
        self.input_embedding = nn.Linear(33, 64)

        input_dim = 64  # 
        hidden_dim = 64  # tune this
        num_layers = 1  # Number of LSTM layers, can also be tuned

        # Define LSTM and other layers here
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, features_dim)
        self.hidden_state = None
        self.cell_state = None
        
    
    def forward(self, x):
        # Apply the input embedding layer
        x = self.input_embedding(x)

        # Dynamically set the batch size based on the input
        batch_size = x.size(0)
        
        # Always initialize fresh hidden and cell states
        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        # Pass the input through the LSTM
        x, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # Store the updated states (Note: This might not be necessary during training, as the states might get overridden anyway)
        self.hidden_state = h_n.detach()
        self.cell_state = c_n.detach()
        
        # We are interested in the last output for the actor-critic network
        x = x[:, -1, :]
        return self.fc(x)

class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh, 
                 ortho_init=True, use_sde=False, log_std_init=0, full_std=True, use_expln=False, 
                 features_extractor_class=None, features_extractor_kwargs=None, share_features_extractor=True,
                 normalize_images=True, optimizer_class=torch.optim.Adam, optimizer_kwargs=None):

        self.action_probs = None
        
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

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        if not isinstance(distribution, MultiCategoricalDistribution):
            self.action_probs = distribution.probs
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob

    def reset_states(self):
        self.hidden_state = None
        self.cell_state = None
