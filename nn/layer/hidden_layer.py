from nn.activation.relu import ReLU
from nn.layer.weighted_layer import WeightedLayer


class HiddenLayer(WeightedLayer):
    def __init__(self, count_neuron, activation, previous_layer, next_layer):
        super().__init__(count_neuron, activation, previous_layer, next_layer)
        self.input = input
