import numpy as np
from abc import ABC, abstractmethod

class Layer:
    def __init__(self, count_neuron, activation, previous_layer, next_layer):
        self.count_neuron = count_neuron
        self.activation = activation
        self.previous_layer = previous_layer
        self.next_layer = next_layer

    def forward(self, input):
        output = self.get_output(input)
        if self.next_layer != None:
            self.next_layer.forward(output)

    def backward(self, output):
        pass

    @abstractmethod
    def get_output(self, input):
        pass

    def print(self, level):
        print(f'({level} - {self.count_neuron})')
        if self.next_layer != None:
            self.next_layer.print(level+1)

