import dill
import gym
from stable_baselines3 import DQN, PPO

from yaes.agent import RLAgent
from yaes.agent.modi import Agent as ModiAgent
from yaes.agent.multi_tree import Agent as MultiTreeAgent
from yaes.environment import Environment, wrap_env
from yaes.evaluate import Evaluator

gym_env = gym.make("CartPole-v1")
gym_env = gym.wrappers.Monitor(gym_env, "videos", force=True)

env = wrap_env(gym_env)

evaluator = Evaluator(env)
# rl_agent = RLAgent(env, DQN, "MlpPolicy", {"total_timesteps": int(2e5), "progress_bar": True}, verbose=1)
# rl_agent = RLAgent(env, PPO, "MlpPolicy", {"total_timesteps": int(10000), "progress_bar": True}, verbose=1)
modi_agent = ModiAgent(env)
multi_tree_agent = MultiTreeAgent(env)
stats = evaluator.evaluate([modi_agent, multi_tree_agent])  # , rl_agent])
print(stats)





# env.gym_env = gym.wrappers.RecordVideo(env.gym_env, 'video')
# env.gym_env.metadata['render_fps'] = 1
#
# best_agent = stats[0]["best_agent"]
# with open('stats.pkl', 'wb') as f:
#     dill.dump(stats, f)
# with open('best_agent.pkl', 'wb') as f:
#     dill.dump(best_agent, f)
#
# print(best_agent)
# print(best_agent.__dict__)
