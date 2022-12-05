from yaes.environment import Environment
from deap import gp
from deap import creator, base, tools, algorithms
import operator
import numpy as np


class AgentHelper:
    def __init__(self, func, bounds=None, is_continuous=False):
        self.func = func
        self.bounds = bounds
        self.is_continuous = is_continuous

    def predict(self, state):
        state = list(map(float, state))
        output = self.func(*state)
        if self.is_continuous and self.bounds is not None:
            output = np.clip(output, *self.bounds).tolist()
        else:
            output = int(np.argmax(output))
        return output


class Agent:
    def __init__(self, env: Environment):
        self.env = env
        self.num_states = self.env.get_observation_space()
        self.num_actions = self.env.get_action_space()
        self.is_discrete = self.env.is_discrete()

        self.pset = self._create_primitive_set(self.num_states, self.num_actions)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset,
                       num_outputs=self.num_actions)

        self.agent_helper = self._get_agent_helper

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=3)
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

    def _create_primitive_set(self, num_inputs, num_outputs):
        raise NotImplementedError

    def _get_agent_helper(self, func):
        if self.is_discrete:
            return AgentHelper(func)
        else:
            return AgentHelper(func, bounds=self.env.get_bounds(), is_continuous=True)

    def _fitness(self, agent):
        result = self.env.play(self.agent_helper(agent), render=False)
        reward, steps = result["reward"], result["steps"]
        return reward,  # , steps

    def train(self, n_pop=30, cxpb=0.9, mutpb=0.5, n_gens=10):
        pop = self.toolbox.population(n=n_pop)
        hof = tools.HallOfFame(1)
        log = None
        try:
            for _ in range(n_gens):
                pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb, mutpb, ngen=1, stats=self.stats,
                                               halloffame=hof, verbose=True)
                print("modi", len(pop))

                min_fitness_index = np.argmin(list(map(lambda x: x.fitness.values[0], pop)))
                pop[min_fitness_index] = hof[0]
        except KeyboardInterrupt:
            pass
        finally:
            training_stats = {
                "log": log,
            }
            return self.agent_helper(hof[0]), training_stats
