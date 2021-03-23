from nn.layer.hidden_layer import HiddenLayer
from nn.layer.input_layer import InputLayer
from nn.layer.loss_layer import LossLayer
from nn.layer.output_layer import OutputLayer


class FCNN:
    def __init__(self, count_input, count_output):
        self.count_input = count_input
        self.count_output = count_output
        self.input_layer = InputLayer(count_input)
        self.output_layer = OutputLayer(count_output, self.input_layer)
        self.loss_layer = LossLayer(self.output_layer)
        self.input_layer.next_layer = self.output_layer
        self.output_layer.previous_layer = self.input_layer
        self.output_layer.next_layer = self.loss_layer

    def add_hidden_layer(self, count_neuron, activation):
        source_layer = self.output_layer.previous_layer
        new_layer = HiddenLayer(count_neuron, activation, source_layer, self.output_layer)
        source_layer.next_layer = new_layer
        new_layer.previous_layer = source_layer
        new_layer.next_layer = self.output_layer
        self.output_layer.previous_layer = new_layer

    def print(self):
        self.input_layer.print(1)




