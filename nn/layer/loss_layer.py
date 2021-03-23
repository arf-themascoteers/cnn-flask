from nn.layer.layer import Layer
import math


class LossLayer(Layer):
    def __init__(self, output_layer):
        super().__init__(1, None, output_layer, None)
        self.expected_output = None

    def get_output(self, input):
        logs = list(map(math.log, input))
        products = list(map(math.prod, zip(self.expected_output, logs)))
        return -sum(products)

    def set_expected_output(self, expected_output):
        self.expected_output = expected_output




