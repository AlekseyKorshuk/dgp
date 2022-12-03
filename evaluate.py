import numpy as np
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


set_seed(0)

import gym
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from yaes.agent import multi_tree, RLAgent
from yaes.agent.modi import ModiAgent
from yaes.environment import Environment, wrap_env
from yaes.evaluate import Evaluator
import dill
# First, we create our environment called LunarLander-v2
# import flappy_bird_gym

# gym_env = flappy_bird_gym.make("FlappyBird-v0")
import gym_pygame

# import gym_chrome_dino
# gym_env = gym.make("LunarLander-v2")
# gym_env = gym.make("MountainCarContinuous-v0")
# gym_env = gym.make("CartPole-v1")
print(gym.envs.registry.all())

# import highway_env
# import slimevolleygym

gym_env = gym.make("Catcher-PLE-v0")
# gym_env = gym.make("Pendulum-v1")
# gym_env = gym.make("BipedalWalker-v3")
# Then, we create our environment wrapper
env = wrap_env(gym_env)
env.reset()
print(env.state)
print(env.get_observation_space())
print(env.gym_env.action_space)
print(env.get_action_space())
# env.demo(True)
# Evaluate the agents
evaluator = Evaluator(env)
# rl_agent = RLAgent(env, DQN, "MlpPolicy", {"total_timesteps": int(2e5), "progress_bar": True}, verbose=1)
rl_agent = RLAgent(env, PPO, "MlpPolicy", {"total_timesteps": int(20000), "progress_bar": True}, verbose=1)
multi_tree_agent = multi_tree.MultiTreeAgent(env)
modi_agent = ModiAgent(env)
agents = [multi_tree_agent, modi_agent, rl_agent]
stats = evaluator.evaluate(agents)  # , rl_agent])
env.gym_env.metadata['render_fps'] = 1
env.gym_env = gym.wrappers.RecordVideo(env.gym_env, 'video')
# env.gym_env = Monitor(env.gym_env, './video', force=True)

print(stats)


def dump_stats(stats):
    for i, stat in enumerate(stats):
        best_agent = stat.pop("best_agent")
        # with open(f'stats{i}.pkl', 'wb') as f:
        #     dill.dump(stats, f)
        print(type(best_agent))
        if isinstance(best_agent, BaseAlgorithm):
            best_agent.save(f"best_agent{i}")
        else:
            with open(f'best_agent{i}.pkl', 'wb') as f:
                dill.dump(best_agent, f)


def plot_stats(stats):
    ax = plt.subplot(111)
    num_agents = len(stats)
    plt.title("Agent Comparison")
    plt.xlabel("Seconds")
    plt.ylabel("Reward")
    for i in range(num_agents):
        stat = stats[i]
        label = agents[i].__class__.__name__
        plt.plot(stat["monitor_df"]["t"], stat["monitor_df"]["r"].cummax(), label=label)
    plt.legend()
    plt.show()
    plt.savefig("agent_comparison.png")


plot_stats(stats)

dump_stats(stats)


def plot_stats2(stats):
    ax = plt.subplot(111)
    num_agents = len(stats)
    plt.title("Agent Comparison")
    plt.xlabel("Num evaluations")
    plt.ylabel("Reward")
    for i in range(num_agents):
        stat = stats[i]
        label = agents[i].__class__.__name__
        plt.plot(range(len(stat["monitor_df"])), stat["monitor_df"]["r"].cummax(), label=label)
    plt.legend()
    plt.show()
    plt.savefig("agent_comparison2.png")


plot_stats2(stats)
