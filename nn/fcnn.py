import numpy as np

from nn.input_layer import InputLayer
from nn.output_layer import OutputLayer


class FCNN:
    def __init__(self, count_input, count_output):
        self.count_input = count_input
        self.count_output = count_output
        self.input_layer = InputLayer()
        self.output_layer = OutputLayer()
        self.loss_layer = LossLayer()

    def add_layer(self, count_neuron, activation):
        pass


