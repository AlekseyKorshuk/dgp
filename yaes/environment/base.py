import itertools
import time
from collections import OrderedDict

import gym.spaces
import numpy as np


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.state_ = self.gym_env.reset()
        self.done = False
        self.rewards = []
        self.sum_reward = True
        self.info = None
        self.truncated = False

    def reset(self):
        self.state_ = self.gym_env.reset()
        self.done = False
        self.rewards = []

    def check_action(self, action):
        pass

    def step(self, action):
        self.check_action(action)
        try:
            self.state_, reward, self.done, self.info = self.gym_env.step(action)
        except Exception as e:
            try:
                self.state_, reward, self.done, self.truncated, self.info = self.gym_env.step(action)
            except Exception as e:
                print(self.gym_env.step(action))
                raise e
        self.rewards.append(reward)
        return self.state, self.reward, self.done, self.info

    def render(self):
        self.gym_env.render()

    def close(self):
        self.gym_env.close()

    def play(self, agent_class, render=False, sleep=1 / 30, max_duration=10):
        if type(agent_class) == type:
            agent = agent_class(self.gym_env.observation_space.shape[0], self.gym_env.action_space.n,
                                is_discrete=True)
        else:
            agent = agent_class
        self.reset()
        start_time = time.time()
        while not (self.done or self.truncated or (time.time() - start_time > max_duration)):
            action = agent.predict(self.state)
            if type(action) == tuple:
                action = action[0]
            self.step(action)
            if render:
                self.render()
                time.sleep(sleep)
        return {
            "reward": self.reward,
            "steps": self.get_elapsed_steps(),
            "agent": agent,
            "info": self.info,
        }

    def demo(self, render=False, sleep=0):
        self.reset()
        while not (self.done or self.truncated):
            action = self.gym_env.action_space.sample()
            if type(action) == tuple:
                action = action[0]
            data = self.step(action)
            if render:
                print("here", action, (self.done or self.truncated))
                self.render()
                time.sleep(sleep)

        return {
            "reward": self.reward,
            "steps": self.get_elapsed_steps(),
            "info": self.info,
        }

    def get_elapsed_steps(self):
        if "score" in self.info:
            return self.info["score"]
        try:
            return self.gym_env._elapsed_steps
        except:
            return 0

    def get_observation_space(self):
        if type(self.gym_env.observation_space) == gym.spaces.Discrete:
            return 1
        if type(self.gym_env.observation_space) == gym.spaces.dict.Dict:
            num_inputs = 0
            for key in self.gym_env.observation_space.spaces:
                num_inputs += self.gym_env.observation_space.spaces[key].shape[0]
            return num_inputs
        return np.prod(self.gym_env.observation_space.shape)

    def get_action_space(self):
        if type(self.gym_env.action_space) == gym.spaces.Discrete:
            return self.gym_env.action_space.n
        return self.gym_env.action_space.shape[0]

    def is_discrete(self):
        return True

    @property
    def state(self):
        if type(self.gym_env.observation_space) == gym.spaces.box.Box and type(self.state_) != tuple:
            if type(self.state_[0]) == float or type(self.state_[0]) == np.float64:
                return self.state_
            try:
                state = np.concatenate(self.state_)
                while len(state.shape) > 1:
                    state = np.concatenate(state)
                return list(state)
            except:
                return self.state_

        if type(self.gym_env.observation_space) == gym.spaces.Discrete:
            return list([self.state_])
        if type(self.gym_env.observation_space) == gym.spaces.dict.Dict or type(
                self.gym_env.observation_space) == gym.spaces.box.Box:
            state = []
            if type(self.state_) == list or type(self.state_) == tuple:
                self.state_ = self.state_[0]
            if type(self.state_) == OrderedDict:
                for key in self.state_:
                    state += self.state_[key].tolist()
            else:
                try:
                    state = np.concatenate(self.state_)
                    while len(state.shape) > 1:
                        state = np.concatenate(state)
                    return list(state)
                except:
                    return self.state_
            return list(state)
        if type(self.state_[0]) != list:
            return list(self.state_)
        state = np.concatenate(self.state_)
        while len(state.shape) > 1:
            state = np.concatenate(state)
        return list(state)

    @property
    def reward(self):
        if self.sum_reward:
            return sum(self.rewards)
        return self.rewards[-1]
