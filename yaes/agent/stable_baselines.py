from pathlib import Path

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from yaes.environment import Environment


class RLAgent:
    def __init__(self, env: Environment, agent_class, policy, train_args, verbose=False):
        self.env = env
        self.policy = policy
        self.agent_class = agent_class
        self.agent = None
        self.verbose = verbose
        self.train_args = train_args

    def train(self, log_dir="tmp/"):
        self.env.gym_env = Monitor(self.env.gym_env, log_dir)
        self.agent = self.agent_class(self.policy, self.env.gym_env, verbose=self.verbose, learning_rate=1e-3)
        self.agent.learn(**self.train_args)
        self.env.gym_env = self.env.gym_env.env
        return self.agent

    def predict(self, state):
        action, _state = self.agent.predict(state)
        return action
