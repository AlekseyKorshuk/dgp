import dill
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from dgp.agent import RLAgent, multi_tree
from dgp.agent.modi import ModiAgent
from dgp.environment import wrap_env
from dgp.evaluate import Evaluator


def dump_results(stats, agent_names=None):
    if agent_names is None:
        agent_names = [str(i) for i in range(len(stats))]
    for stat, agent_name in zip(stats, agent_names):
        best_agent = stat.pop("best_agent")
        print(type(best_agent))
        if isinstance(best_agent, BaseAlgorithm):
            best_agent.save(f'./logs/monitor_stats_{agent_name}/model')
        else:
            with open(f'./logs/monitor_stats_{agent_name}/model.pkl',
                      'wb') as f:
                dill.dump(best_agent, f)


def train_dash(gym_name, gym_lib):
    if gym_lib is not None:
        import importlib
        importlib.import_module(gym_lib)
    gym_env = gym.make(gym_name)
    env = wrap_env(gym_env)
    env.reset()
    evaluator = Evaluator(env)
    rl_agent = RLAgent(env, PPO, "MlpPolicy", {"total_timesteps": int(25000), "progress_bar": True}, verbose=1)
    multi_tree_agent = multi_tree.MultiTreeAgent(env)
    modi_agent = ModiAgent(env)
    agents = [multi_tree_agent, modi_agent, rl_agent]
    stats = evaluator.evaluate(agents)
    best_agents = [stat["best_agent"] for stat in stats]
    agent_names = [agent.__class__.__name__ for agent in agents]
    dump_results(stats, agent_names)
    record_video(best_agents, agent_names, gym_env)
    del gym_env
    del env
    del evaluator
    del rl_agent
    del multi_tree_agent
    del modi_agent
    del agents


def record_video(agents, agent_names, gym_env):
    gym_env.metadata['render_fps'] = 1

    for agent, agent_name in zip(agents, agent_names):
        gym_rec = gym.wrappers.RecordVideo(gym_env, f'./logs/monitor_stats_{agent_name}/video')
        wrapped_env = wrap_env(gym_rec)
        wrapped_env.play(agent, render=False, max_duration=20, sleep=0)
        # agent.play(gym_rec, True)
    gym_env.close()
