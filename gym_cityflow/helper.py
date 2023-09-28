# parameter_loader.py

import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    #TODO log loss
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # print(self.locals)
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True


def load_parameters(json_file_path):
    """
    Load parameters for PPO from a JSON file.

    Parameters:
        json_file_path (str): The path to the JSON file containing the parameters.

    Returns:
        dict: A dictionary containing the parameters read from the JSON file.
    """
    with open(json_file_path, 'r') as f:
        params = json.load(f)
    return params

def set_checkpoints_and_directories():

    return None
