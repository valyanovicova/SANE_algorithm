import numpy as np
from .gene import Gene, ConnectionType


class Neuron(object):
    def __init__(self,
                 connections_count: int):
        self.genes = []
        self.connections_count = connections_count
        self.fitness = 0.0

    def init(self,
             min_value: float,
             max_value: float):
        while True:
            for i in range(self.connections_count):
                self.genes.append(Gene(
                    min_value=min_value,
                    max_value=max_value))
            for i in range(self.connections_count):
                self.genes[i].init()
            connection_types = [connection.get_connection_type().value for connection in self.genes]
            if len(set(connection_types)) > 1:
                break
            self.genes.clear()

    def get_weights(self,
                    neurons_count: int,
                    connection: ConnectionType) -> np.array:
        result = np.zeros(neurons_count)
        for gene in self.genes:
            if gene.get_connection_type() == connection:
                result[gene.get_index(neurons_count)] = gene.get_weight()
        return result

    def get_input_weights(self,
                          neurons_count: int) -> np.array:
        return self.get_weights(
            neurons_count=neurons_count,
            connection=ConnectionType.INPUT)

    def get_output_weights(self,
                           neurons_count: int) -> np.array:
        return self.get_weights(
            neurons_count=neurons_count,
            connection=ConnectionType.OUTPUT)

    def mutation(self):
        for i in range(len(self.genes)):
            self.genes[i].mutation()

    @staticmethod
    def crossover(parent1, parent2):
        genes_count = len(parent1.genes)
        connections_count = parent1.connections_count
        child1 = Neuron(
            connections_count=connections_count)
        child2 = Neuron(
            connections_count=connections_count)
        for i in range(genes_count):
            child1_gene, child2_gene = Gene.crossover(
                parent1=parent1.genes[i],
                parent2=parent2.genes[i])
            child1.genes.append(child1_gene)
            child2.genes.append(child2_gene)
        return child1, child2
