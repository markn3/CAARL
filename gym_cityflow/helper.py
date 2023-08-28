# parameter_loader.py

import json

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

if __name__ == "__main__":
    # Test the function
    params = load_parameters('ppo_params.json')
    print("Loaded Parameters:")
    print(params)
