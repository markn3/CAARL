import gym
import gym_cityflow
import numpy as np
from stable_baselines3 import PPO
from custom_LSTM import CustomLSTMPolicy

from empirical_estimation import EmpiricalTransitionEstimator, train_estimator

estimator = EmpiricalTransitionEstimator()
transition_probs = train_estimator(estimator)

print("Transitiion probs: ", transition_probs)