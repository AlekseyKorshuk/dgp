import dill
import gym
import highway_env
from yaes.environment import wrap_env
from deap import *
import gym_pygame

gym_env = gym.make("Pixelcopter-PLE-v0")
env = wrap_env(gym_env)

with open('best_agent.pkl', 'rb') as f:
    best_agent = dill.load(f, fix_imports=False, encoding="ASCII", errors="")

print(best_agent.__dict__)

env.gym_env = gym.wrappers.RecordVideo(env.gym_env, 'agent_preview_copter')
print(env.play(best_agent, False, max_duration=100))

print(env.play(best_agent, False, max_duration=100))

print(env.play(best_agent, False, max_duration=100))
