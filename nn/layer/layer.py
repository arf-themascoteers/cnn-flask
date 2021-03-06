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
        if self.activation != None:
            output = self.activation.forward(output)

        if self.next_layer != None:
            self.next_layer.forward(output)

    def backward(self, back_input):
        dinput = None
        if self.activation != None:
            dinput = self.activation.backward(back_input)

        dinput = self.backprop(back_input)

        if self.previous_layer != None:
            self.previous_layer.backward(dinput)

    @abstractmethod
    def backprop(self, back_input):
        pass

    @abstractmethod
    def get_output(self, input):
        pass

    def print(self, level):
        print(f'(Layer - {level} - Neurons {self.count_neuron})')
        if self.next_layer != None:
            self.next_layer.print(level + 1)

    def print_backward(self, level):
        print(f'(Layer (from back) - {level} - Neurons {self.count_neuron})')
        if self.previous_layer != None:
            self.previous_layer.print_backward(level + 1)
