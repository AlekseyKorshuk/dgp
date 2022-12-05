import numpy as np
import random
from yaes.utils import dump_results
import gym
from stable_baselines3 import DQN, PPO, TD3
from yaes.agent import multi_tree, RLAgent
from yaes.agent.modi import ModiAgent
from yaes.environment import wrap_env
from yaes.evaluate import Evaluator
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym_name", type=str, default="CartPole-v1")
    parser.add_argument("--gym_lib", type=str, default=None)

    args = parser.parse_args()
    return args


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
