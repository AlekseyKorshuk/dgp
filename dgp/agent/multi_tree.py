import uuid

from dgp.environment import Environment
from deap import tools
from .deap_primitives import basic_primitive_set
from .base import Agent
from deap.gp import compile
import numpy as np
from deap import algorithms


def get_scores(agent):
    return lambda *state: list(map(lambda func: func(*state), agent))


class MultiTreeAgent(Agent):
    def __init__(self, env: Environment):
        super().__init__(env)

    def _create_primitive_set(self, num_inputs, _):
        random_uuid = uuid.uuid4().hex
        pset = basic_primitive_set(num_inputs, random_uuid)
        return pset

    def _get_agent_helper(self, agent):
        formula = [str(func) for func in agent]
        agent = tuple(map(lambda x: compile(x, self.pset), agent))

        return super()._get_agent_helper(get_scores(agent), formula=formula)

    def train(self, n_pop=30, cxpb=0.9, mutpb=0.5, n_gens=10):
        pops = [self.toolbox.population(n=n_pop) for _ in range(self.num_actions)]
        hofs = [tools.HallOfFame(1) for _ in range(self.num_actions)]

        log = None

        try:
            for _ in range(n_gens):
                pops, log = Evolve(pops, self.toolbox, cxpb, mutpb, 1,
                                   halloffame=hofs,
                                   verbose=True, stats=self.stats)
                fitnesses = self.toolbox.map(lambda x: x[0].fitness.values[0], zip(*pops))
                min_fitness_index = np.argmin(fitnesses)
                for i in range(len(pops)):
                    pops[i][min_fitness_index] = hofs[i][0]

        except KeyboardInterrupt:
            pass

        individual = [hof[0] for hof in hofs]

        training_stats = {
            "log": log,
        }
        return self.agent_helper(individual), training_stats


def Evolve(pop, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = []
    for i in range(len(pop[0])):
        individ = [pop[j][i] for j in range(len(pop))]
        is_it_valid = np.all([sub_ind.fitness.valid for sub_ind in individ])
        if not is_it_valid:
            invalid_ind.append(individ)
    invalid_ind = list(zip(*invalid_ind))

    fitnesses = toolbox.map(toolbox.evaluate, zip(*invalid_ind))
    fitnesses = list(fitnesses)
    for i in range(len(fitnesses)):
        for j in range(len(invalid_ind)):
            invalid_ind[j][i].fitness.values = fitnesses[i]

    for i in range(len(pop)):
        if halloffame is not None:
            halloffame[i].update(pop[i])

    record = stats.compile(pop[0]) if stats else {}

    nevals = 0 if invalid_ind == [] else len(invalid_ind[0])
    logbook.record(gen=0, nevals=nevals, **record)

    for gen in range(1, ngen + 1):
        offspring = []
        for i in range(len(pop)):
            offspring_ = toolbox.select(pop[i], len(pop[i]))
            offspring_ = algorithms.varAnd(offspring_, toolbox, cxpb, mutpb)
            offspring.append(offspring_)

        invalid_ind = []
        for i in range(len(pop[0])):
            individ = [offspring[j][i] for j in range(len(offspring))]
            is_it_valid = np.all([sub_ind.fitness.valid for sub_ind in individ])
            if not is_it_valid:
                invalid_ind.append(individ)
        invalid_ind = list(zip(*invalid_ind))

        fitnesses = toolbox.map(toolbox.evaluate, zip(*invalid_ind))
        fitnesses = list(fitnesses)
        for i in range(len(fitnesses)):
            for j in range(len(invalid_ind)):
                invalid_ind[j][i].fitness.values = fitnesses[i]

        for i in range(len(pop)):
            if halloffame is not None:
                halloffame[i].update(pop[i])

        for i in range(len(pop)):
            pop[i][:] = offspring[i]

        record = stats.compile(pop[0]) if stats else {}

        nevals = 0 if invalid_ind == [] else len(invalid_ind[0])
        logbook.record(gen=0, nevals=nevals, **record)

        if verbose:
            print(*logbook, sep='\n')

    pop = [pop[i] for i in range(len(pop))]
    log = logbook
    return pop, log
