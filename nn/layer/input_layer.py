from nn.layer.layer import Layer


class InputLayer(Layer):
    def __init__(self, count_neuron, next_layer, input):
        super().__init__(count_neuron, None, None, next_layer)
        self.input = input

    def start_forward(self):
        self.next_layer.forward(self.input)

    def get_output(self, input):
        return input