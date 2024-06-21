import copy
import random
from tqdm import tqdm
import numpy as np

__all__ = ["EvolutionFinder"]


class ArchManager:
    def __init__(self):

        # Configurations for NAS and MBconv block
        self.num_blocks = 20
        self.num_stages = 5  # Multiple blocks are repeated in one stage
        self.kernel_sizes = [3, 5, 7]
        self.expand_ratios = [3, 4, 6]
        self.depths = [2, 3, 4]
        self.resolutions = [224]

    # Randomly sample an architecture
    def random_sample(self):
        sample = {}
        d = []
        e = []
        ks = []

        # Randomly selecting different parameters
        for i in range(self.num_stages):
            d.append(random.choice(self.depths))

        for i in range(self.num_blocks):
            e.append(random.choice(self.expand_ratios))
            ks.append(random.choice(self.kernel_sizes))

        # This defines an architecutre.
        sample = {
            "wid": None,
            "ks": ks,
            "e": e,
            "d": d,
            "r": [random.choice(self.resolutions)],
        }

        return sample

    # Randomly resample a specific block in the architecture, kernel and expand ratio
    def random_resample(self, sample, i):
        assert i >= 0 and i < self.num_blocks
        sample["ks"][i] = random.choice(self.kernel_sizes)
        sample["e"][i] = random.choice(self.expand_ratios)

    # Randomly resample depth of a specific stage in the architecture
    def random_resample_depth(self, sample, i):
        assert i >= 0 and i < self.num_stages
        sample["d"][i] = random.choice(self.depths)

    # Randomly resample res
    def random_resample_resolution(self, sample):
        sample["r"][0] = random.choice(self.resolutions)


class EvolutionFinder:

    # do define the type of constraint
    valid_constraint_range = {
        "arthemetic_intensity": [10, 35],
        "latency": [10, 45],
        "efficient_arthemetic_intensity": [0.4, 10],
        "cli": [0, 50],
    }

    # need to add MAC and total size of model
    def __init__(
        self,
        constraint_type,  # for now is single constraint
        efficiency_predictor,
        **kwargs
    ):
        self.constraint_type = constraint_type

        if not constraint_type in self.valid_constraint_range.keys():
            self.invite_reset_constraint_type()
        self.efficiency_predictor = efficiency_predictor
        self.arch_manager = ArchManager()
        self.num_blocks = self.arch_manager.num_blocks
        self.num_stages = self.arch_manager.num_stages

        self.mutate_prob = kwargs.get(
            "mutate_prob", 0.1
        )  # prob that any mbconvblock to get mutate
        self.population_size = kwargs.get("population_size", 100)
        self.max_time_budget = kwargs.get("max_time_budget", 500)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)

    # Reseting the constraint if invalid
    def invite_reset_constraint_type(self):
        print(
            "Invalid constraint type! Please input one of:",
            list(self.valid_constraint_range.keys()),
        )
        new_type = input()
        while new_type not in self.valid_constraint_range.keys():
            print(
                "Invalid constraint type! Please input one of:",
                list(self.valid_constraint_range.keys()),
            )
            new_type = input()
        self.constraint_type = new_type

    # Sample an architecutre parameter and then check for constraint
    def random_sample(self):
        while True:
            sample = self.arch_manager.random_sample()
            efficiency = self.efficiency_predictor.predict_efficiency(sample)

            return sample, efficiency

    def mutate_sample(self, sample):
        while True:
            new_sample = copy.deepcopy(sample)

            if random.random() < self.mutate_prob:
                self.arch_manager.random_resample_resolution(new_sample)

            for i in range(self.num_blocks):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample(
                        new_sample, i
                    )  # kernel and expand_ratio

            for i in range(self.num_stages):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample_depth(new_sample, i)

            efficiency = self.efficiency_predictor.predict_efficiency(new_sample)

            return new_sample, efficiency

    def crossover_sample(self, sample1, sample2):
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    continue
                for i in range(len(new_sample[key])):
                    new_sample[key][i] = random.choice(
                        [sample1[key][i], sample2[key][i]]
                    )

            efficiency = self.efficiency_predictor.predict_efficiency(new_sample)

            return new_sample, efficiency

    def run_evolution_search(self, verbose=False):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(
            round(self.mutation_ratio * population_size)
        )  # how many to mutate from population
        parents_size = int(
            round(self.parent_ratio * population_size)
        )  # how many parents from the population for next generation

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples
        child_pool = []  # store samples
        efficiency_pool = []  # store efficiency
        best_info = None  # (validation, sample, latency) of best accuracy sample

        print("Generate random population...")
        for _ in range(population_size):
            sample, efficiency = self.random_sample()  # all pass efficiency creteria
            child_pool.append(sample)
            efficiency_pool.append(efficiency)

        # entire set of sample predict
        for i in range(population_size):
            population.append(
                (
                    efficiency_pool[i],
                    child_pool[i],
                )
            )

        # Seeding completed

        print("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        for iter in tqdm(
            range(max_time_budget),
            desc="Searching with %s" % (self.constraint_type),
        ):
            parents = sorted(population, key=lambda x: x[0])[::-1][
                :parents_size
            ]  # sorting based on cli
            cli = parents[0][0]
            if verbose:
                print("Iter: {} CLI: {}".format(iter - 1, parents[0][0]))

            if cli > best_valids[-1]:
                best_valids.append(cli)
                best_info = (parents[0][0], parents[0][1])
            else:
                best_valids.append(best_valids[-1])

            # Empety child pool to store new generation
            population = parents
            child_pool = []
            efficiency_pool = []

            for i in range(mutation_numbers):  # 50
                par_sample = population[np.random.randint(parents_size)][1]
                # Mutate
                new_sample, efficiency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            for i in range(population_size - mutation_numbers):  # 50
                par_sample1 = population[np.random.randint(parents_size)][1]
                par_sample2 = population[np.random.randint(parents_size)][1]
                # Crossover
                new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            for i in range(population_size):
                population.append(
                    (
                        efficiency_pool[i],
                        child_pool[i],
                    )
                )

        return best_valids, best_info
