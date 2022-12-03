from yaes.environment import Environment
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
        return basic_primitive_set(num_inputs, "multi_tree_pset")

    def _get_agent_helper(self, agent):
        agent = tuple(map(lambda x: compile(x, self.pset), agent))

        return super()._get_agent_helper(get_scores(agent))

    def train(self, n_pop=30, cxpb=0.9, mutpb=0.5, n_gens=3):
        pops = [self.toolbox.population(n=n_pop) for _ in range(self.num_actions)]
        hofs = [tools.HallOfFame(1) for _ in range(self.num_actions)]

        log = None

        try:
            for _ in range(n_gens):
                pops, log = Evolve(pops, self.toolbox, 0.9, 0.5, 1,
                                  halloffame=hofs,
                                  verbose=True, stats=self.stats)

                fitnesses = self.toolbox.map(self.toolbox.evaluate, zip(*pops))
                min_fitness_index = np.argmin(fitnesses)
                for i in range(len(pops)):
                    pops[i][min_fitness_index] = hofs[i][0]

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        finally:
            individual = [hof[0] for hof in hofs]

            training_stats = {
                "log": log,
            }
            return self.agent_helper(individual), training_stats


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
            print('sldfjlas;dkfj', len(pop), type(pop[i]), len(pop[i]))
            record.append(stats.compile(pop[i]) if stats else {})

        for i in range(len(pop)):
            logbooks[i].record(gen=gen, nevals=len(invalid_ind[i]), **record[i])

        if verbose:
            # print(logbooks[0].stream)
            print(*[logbook.stream for logbook in logbooks], sep='\n')

    pop = [pop[i] for i in range(len(pop))]
    log = [logbooks[i] for i in range(len(logbooks))]
    return pop, log
