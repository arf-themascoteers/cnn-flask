from nn.layer.layer import Layer


class InputLayer(Layer):
    def __init__(self, count_neuron):
        super().__init__(count_neuron, None, None, None)

    def get_output(self, input):
        return input