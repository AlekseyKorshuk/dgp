import gym

from dgp.environment.continuous import ContinuousEnvironment
from dgp.environment.discrete import DiscreteEnvironment


def wrap_env(env: gym.Env):
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        return DiscreteEnvironment(env)
    elif type(env.action_space) == gym.spaces.box.Box:
        return ContinuousEnvironment(env)
    else:
        raise Exception(f"Unknown environment type {type(env.action_space)}")
