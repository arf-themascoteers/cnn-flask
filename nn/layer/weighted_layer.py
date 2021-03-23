import numpy as np
import math
from abc import ABC, abstractmethod

from nn.layer.layer import Layer


class WeightedLayer(Layer):
    def __init__(self, count_neuron, activation, previous_layer, next_layer):
        super().__init__(count_neuron, activation, previous_layer, next_layer)
        self.weights = np.ones((previous_layer.count_neuron, count_neuron))
        self.biases = np.zeros(count_neuron)

    def get_weighted_sums(self, input):
        return np.dot(input, self.weights)

    def get_biased_sums(self, input):
        weighted_sums = self.get_weighted_sums(input)
        return list(map(sum, zip(weighted_sums, self.biases)))

    def get_activation(self, input):
        biased_sums = self.get_biased_sums(input)
        return self.activation.forward(biased_sums)

    def get_output(self, input):
        return self.get_activation(input)
