# parameter_loader.py
import os
import json
import gym
import time
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from dual_wrapper import DualEnvWrapper
from custom_LSTM import CustomLSTMPolicy
from stable_baselines3.common.callbacks import BaseCallback
from utils.empirical_estimation import EmpiricalTransitionEstimator, train_estimator

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


def get_config(json_file_path):
    with open(json_file_path, 'r') as f:
        config = json.load(f)
    return config

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{func.__name__} took {round(duration, 2)} seconds to complete")
        return result
    return wrapper

@timing_decorator
def create_models(config):

    print("Training estimator")
    estimator = EmpiricalTransitionEstimator()
    transition_probs = train_estimator(estimator)
    print("done")

    # Accessing a configuration group and value
    file_config = config["file_config"]

    if not os.path.exists(file_config["models_dir"]):
        os.makedirs(file_config["models_dir"])

    if not os.path.exists(file_config["agent_folder"]):
        os.makedirs(file_config["agent_folder"])

    if not os.path.exists(file_config["adv_folder"]):
        os.makedirs(file_config["adv_folder"])


    # Initialize base environment and wrap it
    base_env  = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

    # Create wrappers for the agent and the adversary
    agent_env = DualEnvWrapper(base_env, action_space=base_env.action_space)
    adv_env = DualEnvWrapper(base_env, action_space=spaces.MultiDiscrete([6]*33), tp=transition_probs, os=True)

    # Load or create agent and adversary
    if file_config["load_models"]:
        if file_config["load_pretrain"]:
            filepath = file_config["pretrain_checkpoint"]
            print(f"Loading {filepath}")
            agent = PPO.load(file_config["pretrain_checkpoint"], env=agent_env)
            adv = PPO("MlpPolicy", adv_env, verbose=1, n_steps=1000, tensorboard_log=file_config["adv_folder"])
        else:
            agent = PPO.load(file_config["agent_folder"] + "/" + file_config["agent_checkpoint_path"], env=agent_env)
            adv = PPO.load(file_config["adv_folder"] + "/" + file_config["adv_checkpoint_path"], env=adv_env)
            with open(file_config["current_episode"], "r") as file:
                start_episode = int(file.read())
    else:
        agent = PPO(CustomLSTMPolicy, agent_env, verbose=1, ent_coef=0.001, tensorboard_log=file_config["agent_folder"])
        adv = PPO("MlpPolicy", adv_env, verbose=1, n_steps=1000, tensorboard_log=file_config["adv_folder"])


    return agent, agent_env, adv, adv_env

def update_model(old_model, env):
    old_weights = old_model.get_parameters()

    # Create a new agent with updated hyperparameters
    new_model = PPO(
        CustomLSTMPolicy, 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=1000,
        ent_coef=0.001,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )

    # Load the old weights into the new model
    new_model.load_parameters(old_weights)
    return new_model

def save_model(agent, adv, file_config, episode=0):
    # Save model checkpoints and update current episode at specified intervals
    
    agent.save(file_config["agent_folder"] + "/" + file_config["agent_checkpoint_path"] + "_" + str(episode))
    adv.save(file_config["adv_folder"] + "/" +  file_config["adv_checkpoint_path"] + "_" + str(episode))
    with open(file_config["current_episode"], "w") as file:
        file.write(str(episode))

    return None

