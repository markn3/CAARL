# parameter_loader.py

import json
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger

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
