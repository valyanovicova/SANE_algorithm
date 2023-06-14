from typing import List
import numpy as np
from .layer import Layer
from .activations import Sigmoid
from .neuron_population import NeuronPopulation


class NeuralNetwork(object):
    def __init__(self,
                 hidden_neurons: List[int],
                 inputs_count: int,
                 outputs_count: int,
                 neuron_population: NeuronPopulation):
        self.fitness = 0.0
        self.input_weights = np.zeros((len(hidden_neurons), inputs_count))
        output_weights = np.zeros((len(hidden_neurons), outputs_count))
        for i in range(len(hidden_neurons)):
            self.input_weights[i] = neuron_population[hidden_neurons[i]].get_input_weights(inputs_count)
            output_weights[i] = neuron_population[hidden_neurons[i]].get_output_weights(outputs_count)
        self.output_weights = np.zeros((outputs_count, len(hidden_neurons)))
        for i in range(outputs_count):
            self.output_weights[i] = output_weights[:, i]
        self.layers = []
        self.layers.append(Layer(
            weights=self.input_weights,
            activation=Sigmoid()))
        self.layers.append(Layer(
            weights=self.output_weights,
            activation=Sigmoid()))

    def forward(self,
                input_data: np.array) -> np.array:
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
