from abc import ABC, abstractmethod
from nn.layer import Layer


class OutputLayer(Layer):
    def __init__(self, count_neuron, previous_layer, loss_layer):
        super().__init__(count_neuron, None, None, loss_layer)
        self.input = input

    def start_forward(self):
        self.next_layer.forward(self.input)
