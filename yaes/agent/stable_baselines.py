from pathlib import Path

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from yaes.environment import Environment
from stable_baselines3.common.callbacks import BaseCallback


class RewardLogger(BaseCallback):
    """Logs the reward accumulated in each episode"""

    def _on_step(self) -> bool:
        '''
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        '''
        print(self.locals)


class RLAgent:
    def __init__(self, env: Environment, agent_class, policy, train_args, verbose=False):
        self.env = env
        self.agent = agent_class(policy, self.env.gym_env, verbose=verbose, learning_rate=1e-3)
        self.train_args = train_args

    def train(self):
        data = self.agent.learn(**self.train_args, callback=RewardLogger())
        print(data)
        return self.agent

    def predict(self, state):
        action, _state = self.agent.predict(state)
        print(action)
        return action
