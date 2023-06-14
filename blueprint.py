from typing import List
import random
from .neuron_population import NeuronPopulation


class Blueprint(object):
    def __init__(self,
                 neurons: List[int],
                 neuron_population: NeuronPopulation):
        self.neurons = neurons
        self.neuron_population = neuron_population
        self.fitness = 0.0

    def mutation(self):
        if random.random() <= 0.01:
            new_neuron_index = random.randrange(len(self.neuron_population))
            neuron_index = random.randrange(len(self.neurons))
            self.neurons[neuron_index] = new_neuron_index

    @staticmethod
    def crossover(parent1, parent2):
        neurons_count = len(parent1.neurons)
        crossover_point = random.randrange(neurons_count)
        child1_neurons = parent1.neurons[:crossover_point] + parent2.neurons[crossover_point:]
        child2_neurons = parent2.neurons[:crossover_point] + parent1.neurons[crossover_point:]
        return Blueprint(child1_neurons, None), Blueprint(child2_neurons, None)
