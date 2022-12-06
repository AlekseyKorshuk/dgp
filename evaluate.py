import numpy as np
import random
import gym
from stable_baselines3 import PPO

from yaes.agent import multi_tree, RLAgent
from yaes.agent.modi import ModiAgent
from yaes.environment import wrap_env
from yaes.evaluate import Evaluator


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def main():
    set_seed(0)

    gym_env = gym.make("CartPole-v1")
    env = wrap_env(gym_env)
    evaluator = Evaluator(env)
    rl_agent = RLAgent(env, PPO, "MlpPolicy", {"total_timesteps": int(15000), "progress_bar": True}, verbose=1)
    multi_tree_agent = multi_tree.MultiTreeAgent(env)
    modi_agent = ModiAgent(env)
    agents = [rl_agent, multi_tree_agent, modi_agent]
    stats = evaluator.evaluate(agents)
    env.gym_env.metadata['render_fps'] = 1

    agent_names = [agent.__class__.__name__ for agent in agents]
    best_agents = [stat["best_agent"] for stat in stats]

    for agent, agent_name in zip(best_agents, agent_names):
        video_folder = f'./logs/monitor_stats_{agent_name}/video'
        gym_rec = gym.wrappers.RecordVideo(gym_env, video_folder)
        wrapped_env = wrap_env(gym_rec)
        wrapped_env.play(agent, render=True, max_duration=120, sleep=1 / 30)
        print(f"Video for {agent_names} saved to {video_folder}")
    gym_env.close()


if __name__ == '__main__':
    main()
