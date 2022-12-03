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
    pset = PrimitiveSetTyped("main", [float] * num_inputs, float)
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


sigmoid_v = np.vectorize(sigmoid)


class AgentHelper:
    def __init__(self, func, pset):
        self.func = [compile(func_, pset) for func_ in func]

    def predict(self, state):
        # print(state)
        state = list(map(float, state))
        # print(compile(self.func, self.pset)(*state))
        # import pdb; pdb.set_trace()
        # print()
        output = [func_(*state) for func_ in self.func]
        p = sigmoid_v(output)
        p = p.argmax()
        return int(p.round())


class ContinuousAgentHelper:
    def __init__(self, func, pset, bounds):
        self.func = [compile(func_, pset) for func_ in func]
        self.bounds = bounds

    def predict(self, state):
        # print(state)
        state = list(map(float, state))
        # print(compile(self.func, self.pset)(*state))
        # import pdb; pdb.set_trace()
        # print()
        output = [func_(*state) for func_ in self.func]
        # print(output)
        for i in range(len(output)):
            data = np.clip(output[i], *self.bounds)
            if type(data) == np.ndarray:
                output[i] = float(data[0])
            else:
                output[i] = float(data)
        # print(output)
        return output


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

        for i in range(self.num_actions):
            self.toolbox.register(f"individual{i + 1}", tools.initIterate, creator.Individual, self.toolbox.expr)
            self.toolbox.register("population1", tools.initRepeat, list, eval(f"self.toolbox.individual{i + 1}"))

        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        # self.stats.register("avg", np.mean, axis=0)
        # self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

    def predict(self, state):
        if self.is_discrete:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = random.uniform(0, 1)
        return action

    def get_agent_helper(self):
        if self.is_discrete:
            return AgentHelper
        else:
            return ContinuousAgentHelper

    def get_agent_args(self, agent):
        if self.is_discrete:
            return agent, self.pset
        else:
            return agent, self.pset, self.env.get_bounds()

    def _fitness(self, agent):
        result = self.env.play(self.get_agent_helper()(*self.get_agent_args(agent)), render=False)
        # print(result)
        reward, steps = result['reward'], result['steps']
        if reward is None:
            reward = 0
        if steps is None:
            steps = 0
        if "rewards" in result['info']:
            reward += sum(result['info']['rewards'].values())
        return reward,  # , steps

    def train(self):
        pops = [self.toolbox.population(n=300) for _ in range(self.num_actions)]
        hofs = [tools.HallOfFame(1) for _ in range(self.num_actions)]

        log = None
        generations = 20

        try:
            pop, log = Evolve(pops, self.toolbox, 0.9, 0.5, generations,
                              halloffame=hofs,
                              verbose=True, stats=self.stats)
        except KeyboardInterrupt:
            pass

        # individual = [self.toolbox.compile(expr=hof_[0]) for hof_ in hofs]
        individual = [hof[0] for hof in hofs]
        print("Best individual is:")
        for i in range(self.num_actions):
            print(i, individual[i])

        training_stats = {
            "log": log,
        }

        if self.is_discrete:
            return AgentHelper(individual, self.pset), training_stats
        else:
            return ContinuousAgentHelper(individual, self.pset, self.env.get_bounds()), training_stats


# Re-write the eaSimple() function to evolve the 4 individuals w.r.t to the cost returned by the python function: cost_function
def Evolve(pop, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbooks = [tools.Logbook() for _ in range(len(pop))]
    for i in range(len(pop)):
        logbooks[i].header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = []
    for i in range(len(pop)):
        invalid_ind.append([ind for ind in pop[i] if not ind.fitness.valid])

    fitnesses = []
    for i in range(len(pop)):
        fitnesses_ = toolbox.map(toolbox.evaluate, zip(*invalid_ind))
        fitnesses.append(fitnesses_)
        for ind, fit in zip(invalid_ind[i], fitnesses[i]):
            ind.fitness.values = fit

    for i in range(len(pop)):
        if halloffame is not None:
            halloffame[i].update(pop[i])

    record = []
    for i in range(len(pop)):
        record.append(stats.compile(pop[i]) if stats else {})

    for i in range(len(pop)):
        logbooks[i].record(gen=0, nevals=len(invalid_ind[i]), **record[i])

    for gen in range(1, ngen + 1):
        offspring = []
        for i in range(len(pop)):
            offspring_ = toolbox.select(pop[i], len(pop[i]))
            offspring_ = algorithms.varAnd(offspring_, toolbox, cxpb, mutpb)
            offspring.append(offspring_)

        invalid_ind = []
        for i in range(len(pop)):
            invalid_ind.append([ind for ind in offspring[i] if not ind.fitness.valid])

        fitnesses = []
        for i in range(len(pop)):
            fitnesses_ = toolbox.map(toolbox.evaluate, zip(*invalid_ind))
            fitnesses.append(fitnesses_)
            for ind, fit in zip(invalid_ind[i], fitnesses[i]):
                ind.fitness.values = fit

        for i in range(len(pop)):
            if halloffame is not None:
                halloffame[i].update(pop[i])

        for i in range(len(pop)):
            pop[i][:] = offspring[i]

        record = []
        for i in range(len(pop)):
            # print(i, stats.compile(pop[i]))
            record.append(stats.compile(pop[i]) if stats else {})

        for i in range(len(pop)):
            logbooks[i].record(gen=gen, nevals=len(invalid_ind[i]), **record[i])

        if verbose:
            # print(logbooks[0].stream)
            print(*[logbook.stream for logbook in logbooks], sep='\n')

    pop = [pop[i] for i in range(len(pop))]
    log = [logbooks[i] for i in range(len(logbooks))]
    return pop, log
