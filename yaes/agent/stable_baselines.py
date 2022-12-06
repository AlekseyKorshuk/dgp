from stable_baselines3.common.monitor import Monitor

from yaes.environment import Environment
from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np


class RLAgent:
    def __init__(self, env: Environment, agent_class: BaseAlgorithm, policy: str, train_args: dict,
                 verbose: int = 0):
        """
        This class contains functionality to train and use an RL-agent.

        :param env: an OpenAI Gym environment.
        :param agent_class: a stable_baselines agent class (not an instance)
        :param policy: training policy
        :param train_args: training arguments which will be passed to agent's 'learn' method.
        :param verbose: level of verbosity
        """
        self.env = env
        self.policy = policy
        self.agent_class = agent_class
        self.agent = None
        self.verbose = verbose
        self.train_args = train_args

    def train(self, log_dir: str = "tmp/") -> BaseAlgorithm:
        """
        Trains an RL agent in an environment and collects logs.

        :param log_dir: directory in which logs will be written.
        :return: the trained agent
        """
        self.env.gym_env = Monitor(self.env.gym_env, log_dir)
        self.agent = self.agent_class(self.policy, self.env.gym_env, verbose=self.verbose, learning_rate=1e-3)
        self.agent.learn(**self.train_args)
        self.env.gym_env = self.env.gym_env.env
        return self.agent

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        This function returns an action based on the current game state.
        :param state: current game state
        :return: action
        """
        action, _state = self.agent.predict(state)
        return action
