import gym
from stable_baselines3 import DQN, PPO

from yaes.agent.multi_tree import Agent
from yaes.environment import Environment, wrap_env
from yaes.evaluate import Evaluator

# First, we create our environment called LunarLander-v2
import flappy_bird_gym

gym_env = flappy_bird_gym.make("FlappyBird-v0")
import gym_pygame

# import gym_chrome_dino
gym_env = gym.make("LunarLander-v2")
gym_env = gym.make("MountainCarContinuous-v0")
print(gym.envs.registry.all())
gym_env = gym.make("Swimmer-v3")
# gym_env = gym.make("BipedalWalker-v3")
# Then, we create our environment wrapper
env = wrap_env(gym_env)
env.reset()
print(env.state)
print(env.get_observation_space())
print(env.gym_env.action_space)
print(env.get_action_space())

# Evaluate the agents
evaluator = Evaluator(env)
# rl_agent = RLAgent(env, DQN, "MlpPolicy", {"total_timesteps": int(2e5), "progress_bar": True}, verbose=1)
# rl_agent = RLAgent(env, PPO, "MlpPolicy", {"total_timesteps": int(10000), "progress_bar": True}, verbose=1)
es_agent = Agent(env)
stats = evaluator.evaluate([es_agent])  # , rl_agent])
print(stats)
from gym.wrappers import Monitor

env.gym_env = Monitor(env.gym_env, './video', force=True)
for i in range(1):
    env.play(stats[0]["best_agent"], render=True)

import pdb;

pdb.set_trace()
# for i in range(1):
#     env.play(stats[1]["best_agent"], render=True)
