import dill
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from yaes.agent import RLAgent, multi_tree
from yaes.agent.modi import ModiAgent
from yaes.environment import wrap_env
from yaes.evaluate import Evaluator


def dump_results(stats):
    for i, stat in enumerate(stats):
        best_agent = stat.pop("best_agent")
        print(type(best_agent))
        if isinstance(best_agent, BaseAlgorithm):
            best_agent.save(best_agent.__class__.__name__)
        else:
            with open(f'{best_agent.__class__.__name__}.pkl', 'wb') as f:
                dill.dump(best_agent, f)


def train_dash(gym_name, gym_lib):
    if gym_lib is not None:
        import importlib
        importlib.import_module(gym_lib)
    gym_env = gym.make(gym_name)
    env = wrap_env(gym_env)
    env.reset()
    evaluator = Evaluator(env)
    rl_agent = RLAgent(env, PPO, "MlpPolicy", {"total_timesteps": int(20000), "progress_bar": True}, verbose=1)
    multi_tree_agent = multi_tree.MultiTreeAgent(env)
    modi_agent = ModiAgent(env)
    agents = [multi_tree_agent, modi_agent, rl_agent]
    stats = evaluator.evaluate(agents)
    dump_results(stats)
    del gym_env
    del env
    del evaluator
    del rl_agent
    del multi_tree_agent
    del modi_agent
    del agents
