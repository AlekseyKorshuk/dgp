import random
from yaes.environment import Environment
from deap import gp
from deap.gp import PrimitiveSetTyped, compile
from deap import creator, base, tools, algorithms
import operator
import numpy as np
import random


def if_then_else(input, output1, output2):
    return output1 if input else output2


def safe_div(x1, x2):
    return 0 if x2 < 1e-15 else x1 / x2


def create_primitive_set(num_inputs):
    pset = PrimitiveSetTyped("main", [float]*num_inputs, float)
    pset.addPrimitive(operator.xor, [bool, bool], bool)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(safe_div, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)

    pset.addPrimitive(if_then_else, [bool, float, float], float)
    # pset.addTerminal(3.0, float)
    pset.addTerminal(1, bool)
    pset.addEphemeralConstant('rn', lambda: random.uniform(-1, 1), float)

    return pset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class AgentHelper:
    def __init__(self, func, pset):
        self.func = func
        self.pset = pset

    def predict(self, state):
        state = list(map(float, state))
        # print(compile(self.func, self.pset)(*state))
        # import pdb; pdb.set_trace()
        # print()
        p = sigmoid(compile(self.func, self.pset)(*state))

        return np.clip(int(p.round()), 0, 3)


class Agent:
    def __init__(self, env: Environment):
        self.env = env
        self.num_states = self.env.get_observation_space()
        self.num_actions = self.env.get_action_space()
        self.is_discrete = self.env.is_discrete()

        self.pset = create_primitive_set(self.num_states)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('evaluate', self._fitness)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

    def predict(self, state):
        if self.is_discrete:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = random.uniform(0, 1)
        return action

    def _fitness(self, agent):
        result = self.env.play(AgentHelper(agent, self.pset), render=False)
        reward, steps = result['reward'], result['steps']
        return reward,  #, steps

    def train(self):
        pop = self.toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(pop, self.toolbox, 0.5, 0.1, 10, stats=self.stats,
                                       halloffame=hof, verbose=True)

        print(hof[0])
        return AgentHelper(hof[0], self.pset)




