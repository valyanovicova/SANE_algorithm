from typing import List
import numpy as np
import platform
import os
from copy import deepcopy
from .neuron_population import NeuronPopulation
from .blueprint_population import BlueprintPopulation
from .neural_network import NeuralNetwork
from .utils import mse


CRLF = '\r\x1B[K' if platform.system() != 'Windows' else '\r'


class SANEAlgorithm(object):
    def __init__(self,
                 blueprints_population_size: int,
                 neuron_population_size: int,
                 hidden_layer_size: int,
                 connections_count: int):
        self.neuron_population = NeuronPopulation(
            population_size=neuron_population_size,
            connections_count=connections_count)
        self.blueprint_population = BlueprintPopulation(
            population_size=blueprints_population_size,
            blueprint_size=hidden_layer_size)
        self.best_nn = None

    def init(self,
             min_value: float,
             max_value: float):
        self.neuron_population.init(
            min_value=min_value,
            max_value=max_value)
        self.blueprint_population.init(
            neuron_population=self.neuron_population)

    def train(self,
              generations_count: int,
              x_train: np.array,
              y_train: np.array):
        if x_train.shape[0] != y_train.shape[0]:
            raise Exception()
        result = []
        for generation in range(generations_count):
            inputs_count = x_train[0].size
            outputs_count = y_train[0].size
            neural_networks = self.create_neural_networks(
                inputs_count=inputs_count,
                outputs_count=outputs_count)
            self.forward_train(
                neural_networks=neural_networks,
                x_train=x_train,
                y_train=y_train)
            neural_networks.sort(key=lambda x: x.fitness)
            best_nn = neural_networks[0]
            if self.best_nn is None:
                self.best_nn = deepcopy(best_nn)
            if best_nn.fitness < self.best_nn.fitness:
                self.best_nn = deepcopy(best_nn)
            result.append(self.best_nn.fitness)
            self.update_neuron_fitness()
            self.neuron_population.crossover()
            self.neuron_population.mutation()
            self.blueprint_population.crossover()
            self.blueprint_population.mutation()
            print('{}{}/{} best fitness = {}, current fitness = {}'
                  .format(CRLF, generation, generations_count, self.best_nn.fitness, best_nn.fitness), end='')
        print(os.linesep)
        return result

    def test(self,
             x_test: np.array,
             y_test: np.array):
        if x_test.shape[0] != y_test.shape[0]:
            raise Exception()
        dataset_size = x_test.shape[0]
        result = []
        for i in range(dataset_size):
            error = self.forward(x=x_test[i], y=y_test[i])
            result.append(error)
        return result

    def forward(self, x: np.array, y: np.array):
        output = self.best_nn.forward(x)
        return mse(y, output)

    def forward_train(self,
                      neural_networks: List[NeuralNetwork],
                      x_train: np.array,
                      y_train: np.array):
        dataset_size = x_train.shape[0]
        for i in range(len(neural_networks)):
            errors = []
            for j in range(dataset_size):
                output = neural_networks[i].forward(input_data=x_train[j])
                error = mse(y_true=y_train[j], y_pred=output)
                errors.append(error)
            avg_error = np.array(errors).mean()
            neural_networks[i].fitness = avg_error
            self.blueprint_population[i].fitness = avg_error

    def create_neural_networks(self,
                               inputs_count: int,
                               outputs_count: int) -> List[NeuralNetwork]:
        result = []
        for population in self.blueprint_population:
            hidden_neurons = population.neurons
            result.append(NeuralNetwork(
                hidden_neurons=hidden_neurons,
                inputs_count=inputs_count,
                outputs_count=outputs_count,
                neuron_population=self.neuron_population))
        return result

    def update_neuron_fitness(self):
        for neuron in self.neuron_population:
            fitness_list = []
            for population in self.blueprint_population:
                if neuron in population.neurons:
                    fitness_list.append(population.fitness)
                if len(fitness_list) == 5:
                    break
            neuron.fitness = np.array(fitness_list).mean() if len(fitness_list) > 0 else 0.0
