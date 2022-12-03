import dill
import gym
from stable_baselines3 import DQN, PPO

from yaes.agent.modi import Agent
from yaes.environment import Environment, wrap_env
from yaes.evaluate import Evaluator

# First, we create our environment called LunarLander-v2
import flappy_bird_gym

# gym_env = flappy_bird_gym.make("FlappyBird-v0")
import highway_env
import gym_pygame

gym_env = gym.make("Pixelcopter-PLE-v0")

env = wrap_env(gym_env)
env.reset()
print(env.state_)
print(env.state)
print(env.get_observation_space())
print(env.gym_env.action_space)
print(env.get_action_space())
print(env.step(env.gym_env.action_space.sample()))

# Evaluate the agents
evaluator = Evaluator(env)
# rl_agent = RLAgent(env, DQN, "MlpPolicy", {"total_timesteps": int(2e5), "progress_bar": True}, verbose=1)
# rl_agent = RLAgent(env, PPO, "MlpPolicy", {"total_timesteps": int(10000), "progress_bar": True}, verbose=1)
es_agent = Agent(env)
stats = evaluator.evaluate([es_agent])  # , rl_agent])
print(stats)
env.gym_env = gym.wrappers.RecordVideo(env.gym_env, 'video')
env.gym_env.metadata['render_fps'] = 1
# env.gym_env = Monitor(env.gym_env, './video', force=True)
best_agent = stats[0]["best_agent"]
with open('stats.pkl', 'wb') as f:
    dill.dump(stats, f)
with open('best_agent.pkl', 'wb') as f:
    dill.dump(best_agent, f)

print(best_agent)
print(best_agent.__dict__)

# for i in range(1):
#     env.play(stats[1]["best_agent"], render=True)
