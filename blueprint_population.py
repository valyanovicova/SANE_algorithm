from typing import List
import random
from .blueprint import Blueprint
from .neuron_population import NeuronPopulation


class BlueprintPopulation(object):
    def __init__(self,
                 population_size:
                 int, blueprint_size: int):
        self.population_size = population_size
        self.blueprint_size = blueprint_size
        self.neuron_population = None
        self.blueprints = []

    def init(self,
             neuron_population: NeuronPopulation):
        self.neuron_population = neuron_population
        for _ in range(self.population_size):
            selected_neurons = self.select_neurons()
            self.blueprints.append(Blueprint(
                neurons=selected_neurons,
                neuron_population=self.neuron_population))

    def select_neurons(self) -> List[int]:
        result = []
        while True:
            for _ in range(self.blueprint_size):
                result.append(random.randrange(len(self.neuron_population)))
            if len(set(result)) == self.blueprint_size:
                break
            result.clear()
        return result

    def mutation(self):
        for blueprint in self.blueprints:
            blueprint.mutation()

    def crossover(self):
        self.blueprints.sort(key=lambda x: x.fitness)
        selected_blueprint_count = int(len(self.blueprints) / 4)
        selected_blueprint_count -= selected_blueprint_count % 2
        for i in range(0, selected_blueprint_count, 2):
            parent1 = self.blueprints[i]
            parent2 = self.blueprints[i + 1]
            child1, child2 = Blueprint.crossover(
                parent1=parent1,
                parent2=parent2)
            child1.neuron_population = self.neuron_population
            child2.neuron_population = self.neuron_population
            selected1 = parent1 if random.randrange(2) == 0 else parent2
            selected2 = child1 if random.randrange(2) == 0 else child2
            self.blueprints[-selected_blueprint_count + i] = selected1
            self.blueprints[-selected_blueprint_count + i + 1] = selected2

    def __getitem__(self, key: int) -> Blueprint:
        return self.blueprints[key]

    def __len__(self):
        return len(self.blueprints)

    def __iter__(self):
        return iter(self.blueprints)
