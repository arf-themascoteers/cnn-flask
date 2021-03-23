import numpy as np

class FCNN:
    def __init__(self, count_input, count_output):
        self.count_input = count_input
        self.count_output = count_output
        self.layers = []

    def add_layer(self, count_neuron, activation):

