import random
from .neuron import Neuron


class NeuronPopulation(object):
    def __init__(self,
                 population_size: int,
                 connections_count: int):
        self.neurons = []
        for i in range(population_size):
            self.neurons.append(Neuron(
                connections_count=connections_count))

    def init(self,
             min_value: float,
             max_value: float):
        for neuron in self.neurons:
            neuron.init(
                min_value=min_value,
                max_value=max_value)

    def crossover(self):
        self.neurons.sort(key=lambda x: x.fitness)
        selected_neuron_count = int(len(self.neurons) / 4)
        selected_neuron_count -= selected_neuron_count % 2
        for i in range(0, selected_neuron_count, 2):
            parent1 = self.neurons[i]
            parent2 = self.neurons[i + 1]
            child1, child2 = Neuron.crossover(
                parent1=parent1,
                parent2=parent2)
            selected1 = parent1 if random.randrange(2) == 0 else parent2
            selected2 = child1 if random.randrange(2) == 0 else child2
            self.neurons[-selected_neuron_count + i] = selected1
            self.neurons[-selected_neuron_count + i + 1] = selected2

    def mutation(self):
        for i in range(len(self.neurons)):
            self.neurons[i].mutation()

    def __getitem__(self, key: int) -> Neuron:
        return self.neurons[key]

    def __len__(self):
        return len(self.neurons)
