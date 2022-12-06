import gym

from yaes.environment.continuous import ContinuousEnvironment
from yaes.environment.discrete import DiscreteEnvironment


def wrap_env(env: gym.Env):
    """
    Wraps an OpenAI Gym environment into a YAES environment

    :param env: OpenAI Gym environment
    """
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        return DiscreteEnvironment(env)
    elif type(env.action_space) == gym.spaces.box.Box:
        return ContinuousEnvironment(env)
    else:
        raise Exception(f"Unknown environment type {type(env.action_space)}")
