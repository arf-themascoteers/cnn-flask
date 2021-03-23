from nn.activation.relu import ReLU
from nn.activation.softmax import SoftMax
from nn.layer.weighted_layer import WeightedLayer


class HiddenLayer(WeightedLayer):
    def __init__(self, count_neuron, previous_layer, loss_layer):
        super().__init__(count_neuron, ReLU(), previous_layer, loss_layer)
        self.input = input
