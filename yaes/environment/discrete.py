import numpy as np

from yaes.environment import Environment


class DiscreteEnvironment(Environment):
    def __init__(self, gym_env):
        super().__init__(gym_env)

    def check_action(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)
        assert type(action) == int, f"This environment is Discrete. Action must be an integer, got {type(action)} {action}"

    def is_discrete(self):
        return True
