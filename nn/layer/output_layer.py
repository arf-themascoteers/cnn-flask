from nn.activation.softmax import SoftMax
from nn.layer.weighted_layer import WeightedLayer

class OutputLayer(WeightedLayer):
    def __init__(self, count_neuron, previous_layer):
        super().__init__(count_neuron, SoftMax(), previous_layer, None)
