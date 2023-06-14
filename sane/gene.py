from enum import Enum
import random


class ConnectionType(Enum):
    INPUT = 1
    OUTPUT = 2

    @classmethod
    def from_label(cls, label: int):
        return cls(cls.OUTPUT if label > (2**7 - 1) else cls.INPUT)


def crossover_integer(parent1: int, parent2: int, precision: int):
    crossover_point = random.randrange(precision)
    mask1 = ((2 ** precision - 1) << crossover_point) & (2 ** precision - 1)
    mask2 = ((2 ** precision - 1) >> (precision - crossover_point)) & (2 ** precision - 1)
    child1 = (parent1 & mask1) | (parent2 & mask2)
    child2 = (parent2 & mask1) | (parent1 & mask2)
    return child1, child2


def crossover_real(parent1: float, parent2: float, blend=0.1):
    child1 = parent1 - blend * (parent2 - parent1)
    child2 = parent2 + blend * (parent2 - parent1)
    return child1, child2


def invert_bit(value: int, bit: int, precision: int):
    mask = (1 << bit) & (2 ** precision - 1)
    return value ^ mask


class IntegerGene(object):
    def __init__(self, value: float, min_value: float, max_value: float, precision: int):
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.precision = precision

    def init(self):
        self.value = random.uniform(self.min_value, self.max_value)

    def get_int_value(self) -> int:
        return int((self.value - self.min_value) * (2**self.precision - 1) / (self.max_value - self.min_value))

    def set_int_value(self, int_value: int):
        self.value = int_value * (self.max_value - self.min_value) / (2**self.precision - 1) + self.min_value
    int_value = property(get_int_value, set_int_value)

    def mutation(self):
        mutation_bit = random.randrange(self.precision)
        self.int_value = invert_bit(self.int_value, mutation_bit, self.precision)

    @staticmethod
    def crossover(parent1, parent2):
        min_value = parent1.min_value
        max_value = parent1.max_value
        precision = parent1.precision
        child1_gene, child2_gene = crossover_integer(
            parent1=parent1.int_value,
            parent2=parent2.int_value,
            precision=precision)
        child1 = IntegerGene(
            value=0.0,
            min_value=min_value,
            max_value=max_value,
            precision=precision)
        child2 = IntegerGene(
            value=0.0,
            min_value=min_value,
            max_value=max_value,
            precision=precision)
        child1.int_value = child1_gene
        child2.int_value = child2_gene
        return child1, child2


class RealGene(object):
    def __init__(self, value: float, min_value: float, max_value: float):
        self.value = value
        self.min_value = min_value
        self.max_value = max_value

    def init(self):
        self.value = random.uniform(self.min_value, self.max_value)

    def mutation(self):
        self.value = random.gauss(mu=self.value, sigma=0.1)

    @staticmethod
    def crossover(parent1, parent2):
        min_value = parent1.min_value
        max_value = parent1.max_value
        child1_gene, child2_gene = crossover_real(
            parent1=parent1.value,
            parent2=parent2.value)
        child1 = RealGene(
            value=child1_gene,
            min_value=min_value,
            max_value=max_value)
        child2 = RealGene(
            value=child2_gene,
            min_value=min_value,
            max_value=max_value)
        return child1, child2


class Gene(object):
    def __init__(self, min_value: float, max_value: float):
        self.label = 0
        #self.weight = IntegerGene(
        #    value=0.0,
        #    min_value=min_value,
        #    max_value=max_value,
        #    precision=32)
        self.weight = RealGene(
            value=0.0,
            min_value=min_value,
            max_value=max_value)

    def get_connection_type(self) -> ConnectionType:
        return ConnectionType.from_label(self.label)

    def get_index(self, neurons_count) -> int:
        return self.label % neurons_count

    def get_weight(self) -> float:
        return self.weight.value

    def init(self):
        self.label = int(random.random() * (2**8 - 1))
        self.weight.init()

    def mutation(self):
        if random.random() <= 0.01:
            mutation_bit = random.randrange(8)
            self.label = invert_bit(self.label, mutation_bit, 8)
            self.weight.mutation()

    @staticmethod
    def crossover(parent1, parent2):
        min_value = parent1.weight.min_value
        max_value = parent1.weight.max_value
        child1 = Gene(
            min_value=min_value,
            max_value=max_value)
        child2 = Gene(
            min_value=min_value,
            max_value=max_value)
        child1.label, child2.label = crossover_integer(
            parent1=parent1.label,
            parent2=parent2.label,
            precision=8)
        #child1.weight, child2.weight = IntegerGene.crossover(
        #    parent1=parent1.weight,
        #    parent2=parent2.weight)
        child1.weight, child2.weight = RealGene.crossover(
            parent1=parent1.weight,
            parent2=parent2.weight)
        return child1, child2
